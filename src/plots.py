import plotly.graph_objects as go
import pandas as pd
from plotly.subplots import make_subplots

# ... (Existing NBER_RECESSIONS) ...


# Hardcoded NBER Recessions
NBER_RECESSIONS = [
    ('1960-04-01', '1961-02-01'),
    ('1969-12-01', '1970-11-01'),
    ('1973-11-01', '1975-03-01'),
    ('1980-01-01', '1980-07-01'),
    ('1981-07-01', '1982-11-01'),
    ('1990-07-01', '1991-03-01'),
    ('2001-03-01', '2001-11-01'),
    ('2007-12-01', '2009-06-01'),
    ('2020-02-01', '2020-04-01')
]

def plot_regime_probabilities(probs_df):
    """
    Multi-line Time Series with Scatter overlay.
    """
    fig = go.Figure()
    
    colors = {
        'Expansion': '#90EE90',   # Light Green
        'Recovery': '#008000',    # Dark Green
        'Stagflation': '#FF4500', # Orange-Red
        'Contraction': '#8B0000'  # Dark Red (Worst)
    }
    
    # 1. Add NBER Recession Bands
    shapes = []
    for start, end in NBER_RECESSIONS:
        # Check if range is within data bounds
        if pd.Timestamp(start) <= probs_df.index.max() and pd.Timestamp(end) >= probs_df.index.min():
            shapes.append(dict(
                type="rect",
                xref="x",
                yref="paper",
                x0=start,
                y0=0,
                x1=end,
                y1=1,
                fillcolor="rgba(128, 128, 128, 0.3)",
                layer="below",
                line_width=0,
            ))
            
    fig.update_layout(shapes=shapes)
    
    # 2. Add Lines and Markers
    for col in probs_df.columns:
        color = colors.get(col, '#000000')
        
        # Merged Line + Markers
        # Y-axis is now the Regime Name (categorical)
        # We plot a horizontal line for each regime
        
        # Scale sizes: 0 probability = 0 size. Max probability 1.0 = 6 size.
        marker_sizes = probs_df[col] * 6
        
        # Create a constant series for the Y-axis values
        y_values = [col] * len(probs_df)
        
        fig.add_trace(go.Scatter(
            x=probs_df.index,
            y=y_values,
            mode='markers',
            name=col,
            line=dict(width=1, color=color), # Thinner line for the "track"
            marker=dict(
                size=marker_sizes,
                color=color,
                opacity=0.8,
                sizemode='diameter',
                sizemin=0,
                line=dict(width=0) # Remove white stroke
            ),
            hovertemplate=f"Date: %{{x|%Y-%m}}<br>Regime: {col}<br>Prob: %{{marker.size:.2f}}<extra></extra>" # Hacky way to get prob back roughly or we can add customdata
        ))
        
        # Note: recovering original probability for tooltip is better with customdata
        fig.data[-1].customdata = probs_df[col]
        fig.data[-1].hovertemplate = "%{y}: %{customdata:.0%}<extra></extra>"

    fig.update_layout(
        title="US Economic Regime Probabilities (1960 - Present)",
        yaxis_title="Regime",
        xaxis_title="Date",
        hovermode="x unified",
        template="plotly_white",
        height=200,
        margin=dict(l=20, r=20, t=30, b=20),
        xaxis=dict(showgrid=False, zeroline=False, showline=False),
        yaxis=dict(
            showgrid=False, 
            zeroline=False, 
            showline=False, 
            showticklabels=True,
            categoryarray=['Contraction', 'Stagflation', 'Recovery', 'Expansion'] # Bottom to Top
        )
    )
    
    return fig

def plot_pca_phase_diagram(pca_df):
    """
    2D Scatter Plot of PC1 (Growth) vs PC2 (Inflation).
    Points colored by assigned Cluster/Regime.
    """
    fig = go.Figure()
    
    # Define quadrant background colors
    # Q1 (Exp): Top Right (+, +)
    # Q2 (Stag): Top Left (-, +)
    # Q3 (Con): Bottom Left (-, -)
    # Q4 (Rec): Bottom Right (+, -)
    
    # We can add shapes for quadrants or annotations
    
    # Scatter points
    # Group by Regime to get legend
    for regime in pca_df['Regime'].unique():
        subset = pca_df[pca_df['Regime'] == regime]
        
        fig.add_trace(go.Scatter(
            x=subset['Growth'],
            y=subset['Inflation'],
            mode='markers',
            name=regime,
            text=subset.index.strftime('%Y-%m'),
            marker=dict(size=6, opacity=0.7)
        ))
        
    # Add axes lines
    fig.add_hline(y=0, line_width=1, line_color="black")
    fig.add_vline(x=0, line_width=1, line_color="black")
    
    # Annotations for Quadrants
    max_x = pca_df['Growth'].abs().max()
    max_y = pca_df['Inflation'].abs().max()
    
    fig.add_annotation(x=max_x/2, y=max_y/2, text="Expansion", showarrow=False, font=dict(size=14, color="gray"))
    fig.add_annotation(x=-max_x/2, y=max_y/2, text="Stagflation", showarrow=False, font=dict(size=14, color="gray"))
    fig.add_annotation(x=-max_x/2, y=-max_y/2, text="Contraction", showarrow=False, font=dict(size=14, color="gray"))
    fig.add_annotation(x=max_x/2, y=-max_y/2, text="Recovery", showarrow=False, font=dict(size=14, color="gray"))

    fig.update_layout(
        title="Economic Phase Diagram (PCA)",
        xaxis_title="Growth (PC1)",
        yaxis_title="Inflation (PC2)",
        template="plotly_white",
        height=600,
        width=800
    )
    
    return fig

def plot_multi_series_with_recessions(df, columns):
    """
    Vertically stacked line charts for multiple series with NBER bounds.
    """
    n = len(columns)
    fig = make_subplots(
        rows=n, 
        cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.05,
        subplot_titles=columns
    )
    
    # Add Traces
    for i, col in enumerate(columns):
        row_idx = i + 1
        
        # Line Trace
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df[col],
            mode='lines',
            name=col,
            line=dict(width=1.5),
            showlegend=False
        ), row=row_idx, col=1)
        
        # Add NBER Recession Bands for this subplot
        # We need to add them as shapes to the specific row's xaxis/yaxis or strictly via layout shapes with xref/yref
        # Plotly makes this tricky with subplots + shapes.
        # Best approach: Add shapes with xref='x', yref='paper' but clipped? No.
        # We must add per-subplot shapes or use specific references 'x1', 'y1', 'x2', 'y2'...
        
        # Let's iterate and add shapes for each subplot
        # Note: In make_subplots, axes are usually x1, y1 for row 1...
        # But shared_xaxes=True means they might share x axis anchor? 
        # Actually shared_xaxes means they are linked, but they have distinct axis objects (x, x2, x3...).
        
        # Simpler approach: Use V-Rects (Shapes) with yref="paper" spanning the whole height?
        # But that covers titles too.
        # Better: Add shape for each subplot.
        pass

    # Add Shapes globally (easier if we use yref='paper' and just let it span everything, user won't mind recession crossing spacing)
    shapes = []
    for start, end in NBER_RECESSIONS:
        if pd.Timestamp(start) <= df.index.max() and pd.Timestamp(end) >= df.index.min():
            shapes.append(dict(
                type="rect",
                xref="x", # Shared X-axis?
                yref="paper",
                x0=start,
                y0=0,
                x1=end,
                y1=1,
                fillcolor="rgba(128, 128, 128, 0.3)",
                layer="below",
                line_width=0,
            ))
    
    fig.update_layout(
        shapes=shapes,
        height=150 * n,
        margin=dict(l=20, r=20, t=40, b=20),
        template="plotly_white",
        hovermode="x unified"
    )
    
    return fig

def plot_economic_health_index(probs_df):
    """
    Line chart of Net Expansion Score = P(Expansion) - P(Contraction).
    Range: [-1, 1].
    """
    df = probs_df.copy()
    df['Net Score'] = df['Expansion'] - df['Contraction']
    
    fig = go.Figure()
    
    # Add Area Trace
    # We want color to be dynamic? Plotly 'bar' or 'scatter' with fill.
    # Easiest is to add two traces: Positive (Green) and Negative (Red).
    
    # Multi-threshold interpolation for 4-color zones
    # Thresholds: 0.5 (Green/SoftGreen), 0.0 (SoftGreen/Orange), -0.5 (Orange/Red)
    thresholds = [0.5, 0.0, -0.5]
    
    y = df['Net Score']
    new_points = []
    
    # Convert to numeric for calculation (nanoseconds)
    t_numeric = df.index.view('int64')
    y_values = y.values
    n = len(df)
    
    for thresh in thresholds:
        pass
        # Vectorized or loop approach. Loop is safer for logic clarity.
        # We can do one pass over data and check all thresholds? Or loop thresholds?
        # Loop thresholds is easier to implement.
        
        # Detect crossings of 'thresh'
        # (y1 - thresh) and (y2 - thresh) have different signs
        
        for i in range(1, n):
            y1 = y_values[i-1]
            y2 = y_values[i]
            
            if (y1 < thresh < y2) or (y1 > thresh > y2):
                # Crossing
                t1 = t_numeric[i-1]
                t2 = t_numeric[i]
                
                # Interpolate timestamp where y = thresh
                # slope = (y2 - y1) / (t2 - t1)
                # thresh = y1 + slope * (tx - t1)
                # tx = t1 + (thresh - y1) / slope
                
                slope = (y2 - y1) / (t2 - t1)
                tx = t1 + (thresh - y1) / slope
                
                new_ts = pd.Timestamp(int(tx))
                new_points.append({'date': new_ts, 'Net Score': thresh})

    # Add new points and resort
    if new_points:
        df_new = pd.DataFrame(new_points).set_index('date')
        df_augmented = pd.concat([df[['Net Score']], df_new]).sort_index()
        df_augmented = df_augmented[~df_augmented.index.duplicated(keep='first')]
    else:
        df_augmented = df[['Net Score']]
        
    # Create 4 masked series
    # Using small epsilon for float comparison safety or just <= / >= logic
    # We want inclusivity at boundaries so lines connect
    
    s = df_augmented['Net Score']
    
    # 1. Strong Expansion (> 0.5)
    s1 = s.copy()
    s1[s1 < 0.5] = None
    
    # 2. Mild Expansion (0.0 to 0.5)
    s2 = s.copy()
    s2[(s2 < 0.0) | (s2 > 0.5)] = None
    
    # 3. Mild Contraction (-0.5 to 0.0)
    s3 = s.copy()
    s3[(s3 < -0.5) | (s3 > 0.0)] = None
    
    # 4. Deep Contraction (< -0.5)
    s4 = s.copy()
    s4[s4 > -0.5] = None
    
    # Plot Traces
    fig.add_trace(go.Scatter(x=s.index, y=s1, mode='lines', name='Strong Expansion', line=dict(color='#90EE90', width=1.5), showlegend=False)) # LightGreen (Expansion)
    fig.add_trace(go.Scatter(x=s.index, y=s2, mode='lines', name='Mild Expansion', line=dict(color='#006400', width=1.5), showlegend=False))   # DarkGreen (Recovery)
    fig.add_trace(go.Scatter(x=s.index, y=s3, mode='lines', name='Mild Contraction', line=dict(color='orange', width=1.5), showlegend=False))  # Orange
    fig.add_trace(go.Scatter(x=s.index, y=s4, mode='lines', name='Deep Contraction', line=dict(color='red', width=1.5), showlegend=False))     # Red
    
    # Add NBER Bands
    shapes = []
    for start, end in NBER_RECESSIONS:
        if pd.Timestamp(start) <= df.index.max() and pd.Timestamp(end) >= df.index.min():
            shapes.append(dict(
                type="rect",
                xref="x",
                yref="paper",
                x0=start,
                y0=0,
                x1=end,
                y1=1,
                fillcolor="rgba(128, 128, 128, 0.3)",
                layer="below",
                line_width=0,
            ))
            
    fig.update_layout(
        title="Economic Health Index (Net Expansion - Contraction)",
        yaxis_title="Score",
        xaxis_title="Date",
        shapes=shapes,
        template="plotly_white",
        height=300,
        yaxis=dict(range=[-1.1, 1.1], zeroline=True, zerolinecolor='black'),
        margin=dict(l=20, r=20, t=40, b=20),
        hovermode="x unified"
    )
    
    return fig

def plot_regime_heatmap(df, regime_col, description_map=None):
    """
    Heatmap of Min-Max Normalized average statistics for each Regime.
    X-axis: Regimes
    Y-axis: FRED Statistics
    """
    # Key Statistics
    key_stats = ['RPI', 'UNRATE', 'UMCSENTx', 'FEDFUNDS', 'CPIAUCSL', 'S&P 500']
    
    # 1. Prepare Data
    joined = df.copy()
    joined['Regime'] = regime_col
    
    # Filter for key stats that exist in data
    valid_stats = [c for c in key_stats if c in joined.columns]
    
    # Group by Regime and Mean
    # Note: 'Regime' column might be categorical (strings)
    grouped = joined.groupby('Regime')[valid_stats].mean()
    
    # Transpose: Rows = Stats, Cols = Regimes
    heatmap_data = grouped.T
    
    # 2. Min-Max Normalization per Row (Statistic)
    # (x - min) / (max - min)
    row_mins = heatmap_data.min(axis=1)
    row_maxs = heatmap_data.max(axis=1)
    
    heatmap_norm = heatmap_data.sub(row_mins, axis=0).div(row_maxs - row_mins, axis=0)
    
    # Ensure columns order: Contraction, Stagflation, Recovery, Expansion (if present)
    desired_order = ['Contraction', 'Stagflation', 'Recovery', 'Expansion']
    existing_cols = [c for c in desired_order if c in heatmap_norm.columns]
    heatmap_norm = heatmap_norm[existing_cols]
    
    # 3. Create Heatmap
    # Prepare text for cells (Original values or Normalized? Request said "display... numbers" usually means norm values or just color)
    # Re-reading: "Display the name of our 4 regimes... instead of the numbers" -> X-axis labels.
    # The reference image has numbers in cells. Let's show normalized values formatted.
    
    text_values = heatmap_norm.round(2).astype(str)
    
    # Prepare Custom Data for Tooltip (Descriptions)
    if description_map:
        # Create a matching matrix of descriptions
        # Rows = Stats, Cols = Regimes (Repeated)
        desc_matrix = []
        for stat in heatmap_norm.index:
            desc = description_map.get(stat, stat) # Default to ID if no desc
            row_descs = [desc] * len(heatmap_norm.columns)
            desc_matrix.append(row_descs)
    else:
        desc_matrix = None

    fig = go.Figure(data=go.Heatmap(
        z=heatmap_norm.values,
        x=heatmap_norm.columns,
        y=heatmap_norm.index,
        text=text_values,
        texttemplate="%{text}",
        textfont={"size": 12},
        colorscale="YlGnBu",
        showscale=True,
        customdata=desc_matrix,
        hovertemplate=(
            "<b>Regime:</b> %{x}<br>"
            "<b>Stat:</b> %{y}<br>"
            "<b>Normalized Value:</b> %{z:.2f}<br>"
            "<b>Description:</b> %{customdata}<extra></extra>"
        ) if desc_matrix else None
    ))
    
    fig.update_layout(
        title="Min-Max Normalized Regime Statistics",
        yaxis_title="FRED Statistic",
        xaxis_title="Regime",
        template="plotly_white",
        height=400,
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis=dict(side="bottom")
    )
    
    return fig
