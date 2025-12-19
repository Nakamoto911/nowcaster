import streamlit as st
import pandas as pd
import numpy as np
from src.data import DataLoader
from src.models import RegimeModel, RegimeHMM
from src.plots import plot_regime_probabilities, plot_pca_phase_diagram, plot_multi_series_with_recessions, plot_economic_health_index, plot_regime_heatmap, plot_regime_probability_subplots, plot_pca_components, plot_regime_timeline, plot_series_comparison


st.set_page_config(layout="wide", page_title="US Economic Regime Nowcaster")

st.title("US Economic Regime Nowcaster")

@st.cache_data
def load_and_process_data_v4():
    loader = DataLoader(
        data_path='2025-11-MD.csv',
        appendix_path='FRED-MD_updated_appendix.csv'
    )
    df = loader.run_pipeline()
    return df, loader

@st.cache_resource
def train_regime_model(df, inflation_series, model_type="GMM"):
    if model_type == "HMM (Experimental)":
        model = RegimeHMM(n_components=6)
    else:
        model = RegimeModel()
        
    model.fit(df, inflation_series=inflation_series)
    return model

def check_nber_alignment(probs_df):
    """
    Check Acceptance Criteria 2 & 3
    """
    tests = []
    
    # helper
    def check_period(start, end, regime, threshold, name):
        mask = (probs_df.index >= start) & (probs_df.index <= end)
        subset = probs_df.loc[mask, regime]
        max_prob = subset.max()
        pass_test = max_prob > threshold
        return {
            "Test": name,
            "Target Regime": regime,
            "Max Prob Observed": f"{max_prob:.2f}",
            "Threshold": f"> {threshold}",
            "Result": "PASS" if pass_test else "FAIL"
        }

    # 1. GFC (2008-2009)
    tests.append(check_period('2008-09-01', '2009-06-01', 'Contraction', 0.5, "GFC (2008)"))
    
    # 2. Dotcom (2001)
    tests.append(check_period('2001-03-01', '2001-11-01', 'Contraction', 0.5, "Dotcom (2001)"))
    
    # 3. 1990 Recession
    tests.append(check_period('1990-07-01', '1991-03-01', 'Contraction', 0.5, "1990 Recession"))
    
    # 4. COVID (2020)
    tests.append(check_period('2020-03-01', '2020-04-01', 'Contraction', 0.9, "COVID Shock (2020)"))
    
    # 5. Stagflation (1974 or 1980 or 2022)
    # Check 2022 for Stagflation dominance
    tests.append(check_period('2022-01-01', '2022-12-01', 'Stagflation', 0.5, "2022 Inflation"))

    return pd.DataFrame(tests)

# --- Main Execution ---

try:
    # Sidebar
    st.sidebar.header("Configuration")
    model_choice = st.sidebar.selectbox("Model Type", ["GMM (Standard)", "HMM (Experimental)"])

    with st.spinner("Loading and Processing Data..."):
        df_processed, loader = load_and_process_data_v4()
    

    with st.spinner(f"Fitting {model_choice}..."):
        # Calculate Inflation Proxy (CPI YoY) for Feature Augmentation
        # We need raw CPI. Re-load raw data quickly.
        tmp_loader = DataLoader('2025-11-MD.csv', 'FRED-MD_updated_appendix.csv')
        tmp_loader.load_data()
        cpi_raw = tmp_loader.raw_df['CPIAUCSL']
        inflation_yoy = cpi_raw.pct_change(12, fill_method=None) * 100
        inflation_yoy = inflation_yoy.reindex(df_processed.index).ffill()

        model = train_regime_model(df_processed, inflation_yoy, model_type=model_choice)
        regime_probs, pca_df = model.transform(df_processed, inflation_series=inflation_yoy)
    
    # Create Tabs
    tab_main, tab_diag, tab_series = st.tabs(["Dashboard", "Diagnostics", "Series"])

    with tab_main:


        # --- Visualizations ---
        #st.header("6. Visualization Requirements")
        
        # --- NEW ADDITION START ---
        st.subheader("Regime Stability (Smoothed Timeline)")

        # 1. Calculate Blocks (Smooth out < 3 month flickers)
        regime_blocks = model.get_regime_blocks(pca_df, min_duration=3)

        # 2. Display the Plot
        fig_timeline = plot_regime_timeline(regime_blocks)
        st.plotly_chart(fig_timeline, width="stretch")

        # 3. (Optional) Show the raw table in an expander for inspection
        with st.expander("View Regime Block Details"):
            st.dataframe(regime_blocks)
        # --- NEW ADDITION END ---

        st.subheader("Regime Probability Time Series")
        fig_ts = plot_regime_probabilities(regime_probs)
        st.plotly_chart(fig_ts, width="stretch")



        st.subheader("Regime Probabilities (Detailed)")
        fig_subplots = plot_regime_probability_subplots(regime_probs)
        st.plotly_chart(fig_subplots, width="stretch")

        st.subheader("Economic Health Index")
        fig_index = plot_economic_health_index(regime_probs)
        st.plotly_chart(fig_index, width="stretch")
        
        st.subheader("PCA Phase Diagram")
        fig_phase = plot_pca_phase_diagram(pca_df, regime_probs)
        st.plotly_chart(fig_phase, width="stretch")

        st.subheader("Regime Statistics (Normalized)")
        # Load descriptions for tooltip
        try:
            desc_df = pd.read_csv('FRED-MD_updated_appendix.csv', encoding='latin-1')
            # Ensure we use 'fred' column as key
            desc_map = desc_df.set_index('fred')['description'].to_dict()
        except Exception as e:
            st.warning(f"Could not load series descriptions: {e}")
            desc_map = {}
            
        # 1. Get auxiliary variables that might be missing from PCA model data
        # (e.g., S&P 500, FEDFUNDS were excluded from PCA but needed for Viz)
        needed_vars = ['S&P 500', 'FEDFUNDS', 'UMCSENTx']
        aux_vars = [v for v in needed_vars if v not in df_processed.columns]
        
        # We need the loader instance for aux vars.
        # Hack for now: Re-create loader to fetch aux vars.
        aux_loader = DataLoader('2025-11-MD.csv', 'FRED-MD_updated_appendix.csv')
        aux_loader.load_data() 
        aux_df = aux_loader.get_feature_series(aux_vars)
        
        # Join with processed data for visualization
        # Use left join on df_processed to align with valid model timeline
        df_viz = df_processed.join(aux_df, how='left')

        fig_stats = plot_regime_heatmap(df_viz, pca_df['Regime'], desc_map)
        st.plotly_chart(fig_stats, width="stretch")

        st.markdown("#### Detailed Statistics- Original Values (Transformed)")
        
        def calculate_regime_stats(df, regime_col, key_stats):
            """
            Calculate Mean and Quintiles for key stats per regime.
            """
            data = df[key_stats].copy()
            data['Regime'] = regime_col
            
            stats_list = []
            
            # Quartiles to calculate
            quantiles = [0.25, 0.5, 0.75]
            q_names = ['25%', 'Median', '75%']
            
            for regime, group in data.groupby('Regime'):
                if regime not in ['Expansion', 'Recovery', 'Stagflation', 'Contraction']:
                    continue
                    
                for col in key_stats:
                    series = group[col]
                    row = {
                        'Regime': regime,
                        'Variable': col,
                        'Mean': series.mean(),
                    }
                    # Add Quantiles
                    qs = series.quantile(quantiles)
                    for q_val, q_name in zip(qs, q_names):
                        row[q_name] = q_val
                        
                    stats_list.append(row)
            
            res_df = pd.DataFrame(stats_list)
            # Sort by Regime Order
            regime_order = {'Contraction': 0, 'Stagflation': 1, 'Recovery': 2, 'Expansion': 3}
            res_df['RegimeVal'] = res_df['Regime'].map(regime_order)
            res_df = res_df.sort_values(['RegimeVal', 'Variable']).drop(columns=['RegimeVal'])
            
            return res_df.set_index(['Regime', 'Variable'])

        # Variables to show (same as heatmap)
        vars_to_show = ['RPI', 'UNRATE', 'UMCSENTx', 'FEDFUNDS', 'CPIAUCSL', 'S&P 500']
        # Filter existing
        vars_to_show = [v for v in vars_to_show if v in df_viz.columns]
        
        stats_df = calculate_regime_stats(df_viz, pca_df['Regime'], vars_to_show)
        st.dataframe(stats_df, width=1200)
    
    with tab_diag:
        st.header("Model Diagnostics")
        
        # 1. Acceptance Criteria Metrics
        st.subheader("Acceptance Criteria Metrics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Variance Explained (PCA)**")
            explained_variance = model.pca.explained_variance_ratio_
            total_var = np.sum(explained_variance) * 100
            
            st.metric(label="Total Variance (PC1 + PC2)", value=f"{total_var:.2f}%")
            if total_var >= 40:
                st.success("PASSED (>40%)")
            else:
                st.error("FAILED (<40%)")
                
            st.write(f"PC1 (Growth): {explained_variance[0]*100:.2f}%")
            st.write(f"PC2 (Inflation): {explained_variance[1]*100:.2f}%")

        with col2:
            st.markdown("**Historical Alignment Checks**")
            test_results = check_nber_alignment(regime_probs)
            st.dataframe(test_results, hide_index=True)

        st.divider()

        # New: PCA Components Time Series
        st.subheader("Principal Components Time Series")
        fig_pca = plot_pca_components(pca_df)
        st.plotly_chart(fig_pca, width="stretch")

        st.divider()
        
        # 2. Inspect Cluster Centroids
        st.subheader("Cluster Centroids")
        
        if hasattr(model, 'gmm'):
            means = model.gmm.means_
            n_comp = model.gmm.n_components
            st.caption("Model: GMM")
        elif hasattr(model, 'model'): # HMM
             means = model.model.means_
             n_comp = model.model.n_components
             st.caption("Model: HMM")
        else:
             means = []
             n_comp = 0
             
        centroid_data = []
        for i in range(n_comp):
            label = model.regime_map.get(i, f"Cluster {i}")
            # The means are in the transformed PCA space (plus augmented dims)
            # Display first 2 dims (Growth, Inflation)
            centroid_data.append({
                "Cluster ID": i,
                "Regime Label": label,
                "Growth (PC1)": means[i, 0],
                "Inflation (PC2)": means[i, 1] if means.shape[1] > 1 else 0
            })
        
        st.dataframe(pd.DataFrame(centroid_data))
        st.info("Expected: Expansion (+,+), Stagflation (-,+), Contraction (-,-), Recovery (+,-)")

        # 3. PCA Loadings (Top Drivers)
        st.subheader("3. PCA Loadings (Top Drivers)")
        
        # Get components (2, n_features)
        components = model.pca.components_
        feature_names = df_processed.columns
        
        # Adjust for signs used in model
        pc1_loadings = components[0] * model.pc1_sign
        pc2_loadings = components[1] * model.pc2_sign
        
        obs_df = pd.DataFrame({
            "Feature": feature_names,
            "PC1_Loading": pc1_loadings,
            "PC2_Loading": pc2_loadings
        }).set_index("Feature")
        
        c1, c2 = st.columns(2)
        
        with c1:
            st.markdown("### PC1 (Growth)")
            st.markdown("**Top Positive** (Should be Production/Employment)")
            st.dataframe(obs_df['PC1_Loading'].sort_values(ascending=False).head(10))
            st.markdown("**Top Negative**")
            st.dataframe(obs_df['PC1_Loading'].sort_values(ascending=True).head(10))
            
        with c2:
            st.markdown("### PC2 (Inflation)")
            st.markdown("**Top Positive** (Should be Prices)")
            st.dataframe(obs_df['PC2_Loading'].sort_values(ascending=False).head(10))
            st.markdown("**Top Negative**")
            st.dataframe(obs_df['PC2_Loading'].sort_values(ascending=True).head(10))

        # 4. Data Hygiene
        st.subheader("4. Processed Data Inspection")
        st.markdown("Checking normalization of key indicators. values should be centered around 0.")
        
        key_vars = ['INDPRO', 'CPIAUCSL', 'UNRATE', 'PAYEMS']
        valid_vars = [v for v in key_vars if v in df_processed.columns]
        
        if valid_vars:
            fig_data = plot_multi_series_with_recessions(df_processed, valid_vars)
            st.plotly_chart(fig_data, width="stretch")
        else:
            st.warning("Key variables (INDPRO, CPIAUCSL...) not found in processed data!")

        # 5. Data Information (Detailed)
        st.subheader("5. Data Information")
        
        # Ranges
        train_start = loader.train_range[0].strftime('%Y-%m') if loader.train_range[0] else "N/A"
        train_end = loader.train_range[1].strftime('%Y-%m') if loader.train_range[1] else "N/A"
        data_start = loader.data_range[0].strftime('%Y-%m') if loader.data_range[0] else "N/A"
        data_end = loader.data_range[1].strftime('%Y-%m') if loader.data_range[1] else "N/A"
        
        st.markdown(f"**Training Period:** {train_start} to {train_end} (Pre-2020)")
        st.markdown(f"**Full Data Period:** {data_start} to {data_end}")
        st.markdown(f"**Series Count:** {df_processed.shape[1]} (Included) / {len(loader.excluded_series) + df_processed.shape[1]} (Total)")

        # Expanders
        with st.expander("Excluded Months (Transform/Cleaning Drop)"):
            if loader.excluded_months:
                st.write(", ".join([d.strftime('%Y-%m') for d in loader.excluded_months]))
                st.info(f"Total Dropped: {len(loader.excluded_months)}")
            else:
                st.success("No months dropped.")

        with st.expander("Excluded Series (Reason)"):
            if loader.excluded_series:
                excl_df = pd.DataFrame(list(loader.excluded_series.items()), columns=['Series', 'Reason'])
                st.dataframe(excl_df)
            else:
                st.success("No series excluded.")

        with st.expander("Included Series per Group"):
            # We need description map
            try:
                desc_df_info = pd.read_csv('FRED-MD_updated_appendix.csv', encoding='latin-1')
                desc_map_info = desc_df_info.set_index('fred')['description'].to_dict()
            except:
                desc_map_info = {}
            
            for grp_id, cols in loader.included_series.items():
                st.markdown(f"**Group {grp_id}** ({len(cols)} series)")
                
                # Create mini dataframe for cleaner display
                grp_data = []
                for c in cols:
                    grp_data.append({
                        "Series ID": c,
                        "Description": desc_map_info.get(c, "N/A")
                    })
                
                st.dataframe(pd.DataFrame(grp_data), hide_index=True)

    with tab_series:
        st.header("Series Analysis")
        scaler_name = loader.scaler.__class__.__name__
        st.write(f"Visualizing raw, transformed, and normalized data for all available series. Scaler used: **{scaler_name}**")

        # Group Names Mapping
        GROUP_NAMES = {
            1: "Output and Income",
            2: "Labor Market",
            3: "Housing",
            4: "Consumption, Orders, and Inventories",
            5: "Money and Credit",
            6: "Interest and Exchange Rates",
            7: "Prices",
            8: "Stock Market"
        }

        # Use the loader already loaded in the main execution
        raw_full = loader.raw_df
        
        # 1. Get t-code transformed data
        all_cols = [c for c in raw_full.columns if c != 'sasdate']
        transformed_all = loader.get_feature_series(all_cols)
        
        # 2. Get normalized data from df_processed (which has all cleaning applied)
        # For series not in df_processed, we won't show the normalized line
        normalized_all = df_processed
        
        # Load appendix for grouping
        try:
            appendix = pd.read_csv('FRED-MD_updated_appendix.csv', encoding='latin1')
            series_to_group = appendix.set_index('fred')['group'].to_dict()
            series_to_desc = appendix.set_index('fred')['description'].to_dict()
        except:
            series_to_group = {}
            series_to_desc = {}

        # Group columns
        grouped_cols = {}
        for col in all_cols:
            grp_id = series_to_group.get(col, "Other")
            if grp_id not in grouped_cols:
                grouped_cols[grp_id] = []
            grouped_cols[grp_id].append(col)

        # Iterate by Group ID
        sorted_groups = sorted([g for g in grouped_cols.keys() if isinstance(g, (int, float))])
        if "Other" in grouped_cols:
            sorted_groups.append("Other")

        for grp_id in sorted_groups:
            grp_name = GROUP_NAMES.get(grp_id, f"Group {grp_id}")
            st.subheader(f"Group {grp_id}: {grp_name}")
            
            for col in grouped_cols[grp_id]:
                desc = series_to_desc.get(col, col)
                with st.expander(f"{col}: {desc}"):
                    try:
                        r_series = raw_full[col].astype(float)
                        t_series = transformed_all[col]
                        
                        # Get formula
                        t_code = loader.t_codes.get(col, 1)
                        formula = loader.TCODE_FORMULAS.get(int(t_code), "Unknown")
                        
                        # Get normalized series if available
                        n_series = normalized_all[col] if col in normalized_all.columns else pd.Series(dtype=float)
                        
                        fig = plot_series_comparison(
                            r_series, 
                            t_series, 
                            n_series,
                            title=f"{col} Analysis", 
                            description=desc,
                            formula=formula,
                            scaler_name=scaler_name
                        )
                        st.plotly_chart(fig, width="stretch")
                        
                        if col not in normalized_all.columns:
                            st.info(f"Note: {col} was excluded from the normalization pipeline (likely due to missing history or group filtering).")
                            
                    except Exception as e:
                        st.error(f"Error plotting {col}: {e}")

except Exception as e:
    st.error(f"An error occurred: {e}")
    st.exception(e)
