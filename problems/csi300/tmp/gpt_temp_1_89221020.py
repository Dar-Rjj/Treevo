import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(data):
    """
    Volume-Weighted Momentum Decay Factor
    Combines multi-timeframe momentum with volume confirmation signals
    """
    df = data.copy()
    
    # Momentum Calculation
    # Short-Term Momentum (5-day)
    mom_short = (df['close'] / df['close'].shift(5) - 1)
    mom_short_decayed = mom_short.ewm(alpha=0.2, adjust=False).mean()  # λ=0.8 decay
    
    # Medium-Term Momentum (10-day)
    mom_medium = (df['close'] / df['close'].shift(10) - 1)
    mom_medium_decayed = mom_medium.ewm(alpha=0.1, adjust=False).mean()  # λ=0.9 decay
    
    # Long-Term Momentum (20-day)
    mom_long = (df['close'] / df['close'].shift(20) - 1)
    mom_long_decayed = mom_long.ewm(alpha=0.05, adjust=False).mean()  # λ=0.95 decay
    
    # Volume Signal Processing
    # Volume Momentum (5-day)
    vol_mom = (df['volume'] / df['volume'].shift(5) - 1)
    vol_mom_decayed = vol_mom.ewm(alpha=0.2, adjust=False).mean()
    
    # Volume Acceleration (2nd derivative using 5-day and 10-day difference)
    vol_mom_10 = (df['volume'] / df['volume'].shift(10) - 1)
    vol_acceleration = vol_mom_decayed - vol_mom_10.ewm(alpha=0.1, adjust=False).mean()
    
    # Volume-Price Divergence (rolling correlation)
    vol_price_corr = pd.Series(index=df.index)
    for i in range(10, len(df)):
        window_data = df.iloc[i-9:i+1]
        if len(window_data) >= 5:  # Ensure enough data for correlation
            corr = window_data['close'].pct_change().corr(window_data['volume'].pct_change())
            vol_price_corr.iloc[i] = corr if not np.isnan(corr) else 0
        else:
            vol_price_corr.iloc[i] = 0
    vol_price_corr = vol_price_corr.fillna(0)
    
    # Factor Combination
    # Weighted Momentum Score
    weighted_momentum = (
        0.4 * mom_short_decayed + 
        0.35 * mom_medium_decayed + 
        0.25 * mom_long_decayed
    )
    
    # Volume Confirmation Score
    volume_confirmation = (vol_mom_decayed + vol_acceleration) * (1 + vol_price_corr)
    
    # Final Alpha Factor
    alpha_factor = weighted_momentum * volume_confirmation
    
    # Volatility adjustment using 20-day standard deviation
    volatility = df['close'].pct_change().rolling(window=20, min_periods=10).std()
    volatility_adj = volatility.replace(0, np.nan).fillna(method='ffill')
    
    # Scale factor by inverse volatility
    final_factor = alpha_factor / volatility_adj
    
    return final_factor
