import pandas as pd
def heuristics_v2(df: pd.DataFrame, macro_df: pd.DataFrame, cross_asset_df: pd.DataFrame) -> pd.Series:
    # Momentum calculation - Adaptive window return based on 30, 60, and 90 periods
