import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

class DataLoader:
    TCODE_FORMULAS = {
        1: "x(t)",
        2: "x(t) - x(t-1)",
        3: "(x(t) - x(t-1)) - (x(t-1) - x(t-2))",
        4: "log(x(t))",
        5: "log(x(t)) - log(x(t-1))",
        6: "(log(x(t)) - log(x(t-1))) - (log(x(t-1)) - log(x(t-2)))",
        7: "(x(t)/x(t-1) - 1) - (x(t-1)/x(t-2) - 1)"
    }

    def __init__(self, data_path, appendix_path):
        self.data_path = data_path
        self.appendix_path = appendix_path
        self.raw_df = None
        self.appendix_df = None
        self.df = None
        self.scaler = StandardScaler()
        
        # Metadata Tracking
        self.excluded_series = {} # {name: reason}
        self.included_series = {} # {group: [cols]}
        self.excluded_months = []
        self.train_range = (None, None)
        self.data_range = (None, None)

    def load_data(self):
        # Load Appendix with latin1 encoding to avoid Unicode errors
        self.appendix_df = pd.read_csv(self.appendix_path, encoding='latin1')
        
        # Load Main Data - Read as object/strings first to handle the 'Transform:' row safely
        # Row 0 is headers, Row 1 is Transform codes
        self.raw_df = pd.read_csv(self.data_path)
        
        # Extract t-codes from the first row of data (index 0)
        # The 'sasdate' column in this row contains "Transform:"
        self.t_codes = self.raw_df.iloc[0].iloc[1:].to_dict() 
        
        # Drop the Transform row (index 0)
        self.raw_df = self.raw_df.iloc[1:].reset_index(drop=True)
        
        # Convert sasdate to datetime
        self.raw_df['sasdate'] = pd.to_datetime(self.raw_df['sasdate'])
        self.raw_df.set_index('sasdate', inplace=True)

        return self

    def filter_groups(self):
        # Groups to KEEP: 1, 2, 3, 4, 5, 7
        # Groups to EXCLUDE: 6, 8
        valid_groups = [1, 2, 3, 4, 5, 7]
        
        # Map series ID to group
        series_to_group = dict(zip(self.appendix_df['fred'], self.appendix_df['group']))
        
        # Identify columns to keep
        cols_to_keep = []
        for col in self.raw_df.columns:
            if col in series_to_group:
                group_id = series_to_group[col]
                if group_id in valid_groups:
                    cols_to_keep.append(col)
                else:
                    self.excluded_series[col] = f"Group {group_id} (Excluded)"
            else:
                self.excluded_series[col] = "Unknown Group"
        
        self.df = self.raw_df[cols_to_keep].astype(float)
        return self

    def clean_data(self):
        # 1. Historical Gaps: Drop columns that don't go back to 1960-01-01
        start_date = pd.Timestamp('1960-01-01')
        
        valid_cols = []
        for col in self.df.columns:
            # Check if likely valid at start_date (allowing for transformations losing first few rows)
            # Actually, raw data should be present.
            if not pd.isna(self.df.loc[start_date, col]):
                valid_cols.append(col)
            else:
                self.excluded_series[col] = "Missing History (Pre-1960)"
        
        self.df = self.df[valid_cols]
        self.df = self.df[self.df.index >= start_date]

        # 2. Intermediate Gaps: Linear Interpolation (Limit 2)
        self.df = self.df.interpolate(method='linear', limit=2)
        
        # 3. Ragged Edge: Drop row if any remaining NaNs
        self.df.dropna(axis=0, how='any', inplace=True)
        
        return self

    def _apply_tcode(self, series, code):
        """
        1: No transformation
        2: First difference
        3: Second difference
        4: Log
        5: First difference of log
        6: Second difference of log
        7: Delta (xt/xt-1 - 1)
        """
        x = series
        code = int(code)
        
        if code == 1:
            return x
        elif code == 2:
            return x.diff()
        elif code == 3:
            return x.diff().diff()
        elif code == 4:
            return np.log(x)
        elif code == 5:
            return np.log(x).diff()
        elif code == 6:
            return np.log(x).diff().diff()
        elif code == 7:
            return (x / x.shift(1) - 1).diff()
        else:
            return x

    def transform_data(self):
        transformed_data = {}
        for col in self.df.columns:
            # Get t-code, default to 1 if missing (unlikely given logic)
            code = self.t_codes.get(col, 1)
            transformed_data[col] = self._apply_tcode(self.df[col], code)
        
        self.df = pd.DataFrame(transformed_data, index=self.df.index)
        
        # Transformations create NaNs directly at the start. Drop them.
        self.df.dropna(inplace=True)
        
        return self

    def winsorize(self):
        # Clip at 0.5% and 99.5% quantiles
        lower_q = self.df.quantile(0.005)
        upper_q = self.df.quantile(0.995)
        
        self.df = self.df.clip(lower=lower_q, upper=upper_q, axis=1)
        return self

    def smooth(self):
        # 3-Month Rolling Mean
        self.df = self.df.rolling(window=3).mean()
        # Rolling mean creates NaNs at start
        self.df.dropna(inplace=True)
        return self

    def normalize(self):
        # Fit scaler on Pre-2020 data ONLY
        train_mask = self.df.index < '2020-01-01'
        train_data = self.df[train_mask]
        
        # Fit
        self.scaler.fit(train_data)
        
        # Transform ALL data
        self.df = pd.DataFrame(
            self.scaler.transform(self.df),
            index=self.df.index,
            columns=self.df.columns
        )
        return self

    def get_feature_series(self, feature_names):
        """
        Retrieve specific series with transformations applied, 
        regardless of group filtering or cleaning.
        Useful for visualization variables (e.g. S&P 500) excluded from PCA.
        """
        result = {}
        for col in feature_names:
            if col in self.raw_df.columns:
                # Ensure numeric
                try:
                    series = self.raw_df[col].astype(float)
                    code = self.t_codes.get(col, 1)
                    transformed = self._apply_tcode(series, code)
                    result[col] = transformed
                except Exception as e:
                    print(f"Error processing {col}: {e}")
        
        # Return dataframe aligned with original index (handling dropped rows from transform)
        return pd.DataFrame(result, index=self.raw_df.index)

    def run_pipeline(self):
        self.load_data()
        
        # Track initial dates
        initial_dates = set(self.raw_df.index)
        
        self.filter_groups()
        self.clean_data()
        self.transform_data()
        self.winsorize()
        self.smooth()
        self.normalize()
        
        # Final Metadata
        final_dates = set(self.df.index)
        self.excluded_months = sorted(list(initial_dates - final_dates))
        
        if not self.df.empty:
            self.data_range = (self.df.index.min(), self.df.index.max())
            # Train range is everything before 2020
            train_mask = self.df.index < '2020-01-01'
            if train_mask.any():
                self.train_range = (self.df[train_mask].index.min(), self.df[train_mask].index.max())
        
        # Populate Included Series (Final Check)
        series_to_group = dict(zip(self.appendix_df['fred'], self.appendix_df['group']))
        for col in self.df.columns:
            if col in series_to_group:
                grp_id = series_to_group[col]
                if grp_id not in self.included_series:
                    self.included_series[grp_id] = []
                self.included_series[grp_id].append(col)
        
        return self.df
