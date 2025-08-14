import pandas as pd
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    # Calculate the momentum factor using the log return over a dynamic lookback period (10 to 30 days)
