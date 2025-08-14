import pandas as pd
def heuristics_v2(df: pd.DataFrame, sentiment: pd.Series, signals: pd.Series, macro_indicators: pd.Series) -> pd.Series:
    # Calculate daily return
