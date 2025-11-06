import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate daily returns
    daily_returns = df['close'].pct_change()
    
    # Calculate intraday volatility (high-low range normalized by open)
    intraday_vol = (df['high'] - df['low']) / df['open']
    
    # Calculate volume-weighted average price
    vwap = (df['amount'] / df['volume']).replace([np.inf, -np.inf], np.nan)
    
    # Calculate price position relative to daily range
    price_position = (df['close'] - df['low']) / (df['high'] - df['low']).replace(0, np.nan)
    
    # Calculate 5-day rolling volume momentum
    volume_momentum = df['volume'] / df['volume'].rolling(window=5, min_periods=1).mean()
    
    # Calculate price deviation from VWAP
    price_vwap_dev = (df['close'] - vwap) / vwap
    
    # Calculate 3-day rolling correlation between returns and volume
    corr_window = 3
    corr_ret_vol = pd.Series(index=df.index, dtype=float)
    for i in range(len(df)):
        if i >= corr_window - 1:
            window_returns = daily_returns.iloc[i-corr_window+1:i+1]
            window_volume = df['volume'].iloc[i-corr_window+1:i+1]
            if len(window_returns) >= 2 and window_volume.std() > 0 and window_returns.std() > 0:
                corr_ret_vol.iloc[i] = window_returns.corr(window_volume)
            else:
                corr_ret_vol.iloc[i] = 0
        else:
            corr_ret_vol.iloc[i] = 0
    
    # Calculate normalized price momentum (5-day vs 10-day)
    mom_5 = df['close'].pct_change(periods=5)
    mom_10 = df['close'].pct_change(periods=10)
    normalized_momentum = mom_5 / mom_10.replace(0, np.nan)
    
    # Combine factors with appropriate weights
    factor = (
        0.3 * daily_returns.rolling(window=5, min_periods=1).mean() +  # Short-term return momentum
        0.2 * intraday_vol.rolling(window=5, min_periods=1).mean() +   # Recent volatility
        0.15 * price_position.rolling(window=3, min_periods=1).mean() +  # Price positioning
        0.15 * volume_momentum +  # Volume momentum
        0.1 * price_vwap_dev.rolling(window=3, min_periods=1).mean() +  # Price-VWAP deviation
        0.05 * corr_ret_vol +  # Return-volume correlation
        0.05 * normalized_momentum.rolling(window=3, min_periods=1).mean()  # Normalized momentum
    )
    
    # Handle any remaining NaN values
    factor = factor.fillna(0)
    
    return factor
