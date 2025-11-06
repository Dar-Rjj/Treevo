import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Momentum Component
    close = df['close']
    
    # Short-term momentum (5-day)
    mom_5d = close / close.shift(5) - 1
    
    # Medium-term momentum (20-day)
    mom_20d = close / close.shift(20) - 1
    
    # Volatility Scaling
    high = df['high']
    low = df['low']
    close_prev = close.shift(1)
    
    # True Range calculation
    tr1 = high - low
    tr2 = abs(high - close_prev)
    tr3 = abs(low - close_prev)
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # 20-day Average True Range
    atr_20d = true_range.rolling(window=20).mean()
    
    # Volume Asymmetry Analysis
    volume = df['volume']
    
    # 5-day price change for market regime
    price_change_5d = close / close.shift(5) - 1
    
    # Classify uptrend/downtrend (positive/negative 5-day change)
    uptrend = price_change_5d > 0
    downtrend = price_change_5d <= 0
    
    # Calculate volume ratio (up-day volume / down-day volume)
    # Create boolean masks for up and down days in past 20 days
    daily_returns = close.pct_change()
    up_days = daily_returns > 0
    down_days = daily_returns <= 0
    
    # Rolling average volume on up days and down days
    up_volume_avg = volume.rolling(window=20).apply(
        lambda x: x[up_days.loc[x.index].values].mean() if up_days.loc[x.index].sum() > 0 else np.nan, 
        raw=False
    )
    down_volume_avg = volume.rolling(window=20).apply(
        lambda x: x[down_days.loc[x.index].values].mean() if down_days.loc[x.index].sum() > 0 else np.nan, 
        raw=False
    )
    
    volume_ratio = up_volume_avg / down_volume_avg
    
    # Factor Combination
    # Scale momentum by volatility
    mom_5d_scaled = mom_5d / atr_20d
    mom_20d_scaled = mom_20d / atr_20d
    
    # Average volatility-scaled momentums
    combined_momentum = (mom_5d_scaled + mom_20d_scaled) / 2
    
    # Apply volume asymmetry adjustment
    alpha_factor = pd.Series(index=df.index, dtype=float)
    
    # For uptrend: multiply by volume ratio
    alpha_factor[uptrend] = combined_momentum[uptrend] * volume_ratio[uptrend]
    
    # For downtrend: divide by volume ratio
    alpha_factor[downtrend] = combined_momentum[downtrend] / volume_ratio[downtrend]
    
    return alpha_factor
