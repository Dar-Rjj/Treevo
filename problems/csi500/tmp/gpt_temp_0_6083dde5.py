import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate Intraday Return
    intraday_return = (df['high'] - df['low']) / df['open']

    # Adjust for Volume
    volume_ma_20 = df['volume'].rolling(window=20).mean()
    volume_diff = df['volume'] - volume_ma_20
    adjusted_intraday_return = intraday_return * volume_diff

    # Incorporate Price Volatility
    close_std_10 = df['close'].rolling(window=10).std()
    volatility_adjustment = 1.5 if close_std_10 > close_std_10.mean() else 0.7
    final_factor = adjusted_intraday_return * volatility_adjustment

    return final_factor
