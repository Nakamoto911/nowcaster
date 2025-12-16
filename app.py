import streamlit as st
import pandas as pd
import numpy as np
from src.data import DataLoader
from src.models import RegimeModel
from src.plots import plot_regime_probabilities, plot_pca_phase_diagram, plot_multi_series_with_recessions, plot_economic_health_index, plot_regime_heatmap

st.set_page_config(layout="wide", page_title="US Economic Regime Nowcaster")

st.title("US Economic Regime Nowcaster")

@st.cache_data
def load_and_process_data():
    loader = DataLoader(
        data_path='2025-11-MD.csv',
        appendix_path='FRED-MD_updated_appendix.csv'
    )
    df = loader.run_pipeline()
    return df

@st.cache_resource
def train_model(df):
    model = RegimeModel()
    model.fit(df)
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
    with st.spinner("Loading and Processing Data..."):
        df_processed = load_and_process_data()
    
    st.success(f"Data Loaded: {df_processed.shape[0]} months, {df_processed.shape[1]} series")
    
    with st.spinner("Fitting Model..."):
        model = train_model(df_processed)
        regime_probs, pca_df = model.transform(df_processed)
    
    # Create Tabs
    tab_main, tab_diag = st.tabs(["Main Dashboard", "Model Diagnostics"])

    with tab_main:


        # --- Visualizations ---
        #st.header("6. Visualization Requirements")
        
        st.subheader("Regime Probability Time Series")
        fig_ts = plot_regime_probabilities(regime_probs)
        st.plotly_chart(fig_ts, width="stretch")

        st.subheader("Economic Health Index")
        fig_index = plot_economic_health_index(regime_probs)
        st.plotly_chart(fig_index, width="stretch")
        
        st.subheader("PCA Phase Diagram")
        fig_phase = plot_pca_phase_diagram(pca_df)
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
        aux_vars = ['S&P 500', 'FEDFUNDS', 'UMCSENTx']
        
        # We need the loader instance. 'load_and_process_data' is a cache wrapper that assumes loader is internal or recreated.
        # But 'load_and_process_data' currently returns just 'df'. Hiding the loader.
        # We can re-instantiate loader just to get raw series (cheap) or refactor 'load_and_process_data' to return loader/extra.
        # Since 'load_and_process_data' is cached, refactoring return type changes cache.
        # Let's instantiate a new loader for aux vars (lightweight read if cached by OS, or just read CSV).
        # Better: Do this INSIDE load_and_process_data?
        # No, 'load_and_process_data' is for the MODEL.
        
        # Hack for now: Re-create loader to fetch aux vars.
        # (In production, we'd refactor logic to avoid double read, but this is fast enough).
        loader = DataLoader('2025-11-MD.csv', 'FRED-MD_updated_appendix.csv')
        loader.load_data() # Loads raw_df
        aux_df = loader.get_feature_series(aux_vars)
        
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

        # 2. Inspect GMM Centroids
        st.subheader("GMM Cluster Centroids")
        means = model.gmm.means_
        # To make sense of means, we need to know which cluster ID maps to what
        # model.regime_map maps ID -> Label
        
        centroid_data = []
        for i in range(5):
            label = model.regime_map.get(i, f"Cluster {i}")
            # The means are in the transformed PCA space
            centroid_data.append({
                "Cluster ID": i,
                "Regime Label": label,
                "Growth (PC1)": means[i, 0],
                "Inflation (PC2)": means[i, 1]
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

except Exception as e:
    st.error(f"An error occurred: {e}")
    st.exception(e)
