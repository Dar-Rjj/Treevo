import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Determine intervals
    short_term_interval = 5
    long_term_interval = 20

    # Calculate Short-Term Price Momentum (ROC)
    short_term_roc = df['close'].pct_change(periods=short_term_interval)

    # Calculate Long-Term Price Momentum (ROC)
    long_term_roc = df['close'].pct_change(periods=long_term_interval)

    # Compute Volatility
    short_term_volatility = df['close'].rolling(window=short_term_interval).std()
    long_term_volatility = df['close'].rolling(window=long_term_interval).std()

    # Calculate Average True Range (ATR) for Short-Term and Long-Term
    tr = df[['high', 'low', 'close']].apply(lambda x: np.max(x) - np.min(x), axis=1)
    short_term_atr = tr.rolling(window=short_term_interval).mean()
    long_term_atr = tr.rolling(window=long_term_interval).mean()

    # Compute Volume Ratio
    short_term_avg_volume = df['volume'].rolling(window=short_term_interval).mean()
    long_term_avg_volume = df['volume'].rolling(window=long_term_interval).mean()
    volume_ratio = short_term_avg_volume / long_term_avg_volume

    # Adjust Short-Term ROC by Multiplying with Short-Term ATR
    adjusted_short_term_roc = short_term_roc * short_term_atr

    # Adjust Long-Term ROC by Multiplying with Long-Term ATR
    adjusted_long_term_roc = long_term_roc * long_term_atr

    # Multiply Adjusted Short-Term ROC by Volume Ratio
    adjusted_short_term_roc *= volume_ratio

    # Multiply Adjusted Long-Term ROC by Volume Ratio
    adjusted_long_term_roc *= volume_ratio

    # Final Factor
    final_factor = adjusted_short_term_roc + adjusted_long_term_roc

    return final_factor
