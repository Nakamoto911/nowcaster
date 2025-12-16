# US Economic Regime Nowcaster

This application implements the "Teacher" model for identifying US Economic Regimes using PCA and GMM, based on FRED-MD data.

## Setup

1.  **Environment**: It is recommended to use a virtual environment.
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

2.  **Dependencies**: Install the required packages.
    ```bash
    pip install -r requirements.txt
    ```

3.  **Data**: Ensure `2025-11-MD.csv` and `FRED-MD_updated_appendix.csv` are in the root directory.

## Running the App

To start the Streamlit dashboard:

```bash
streamlit run app.py
```

## Project Structure

- `app.py`: Main application entry point.
- `src/data.py`: Handles data loading, cleaning, transformation (t-codes, winsorization, smoothing), and normalization.
- `src/models.py`: Contains the `RegimeModel` class (PCA + GMM) with semantic labeling logic.
- `src/plots.py`: Visualization functions using Plotly.
- `requirements.txt`: Python dependencies.
