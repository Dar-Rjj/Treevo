import pandas as pd
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    # Calculate the multi-period log returns for a 10-day and 30-day window
