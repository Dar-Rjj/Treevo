import pandas as pd
def heuristics_v2(df: pd.DataFrame, sentiment_df: pd.DataFrame, economic_df: pd.DataFrame) -> pd.Series:
    # Calculate daily return
