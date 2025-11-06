import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Raw Momentum Components
    close = df['close']
    
    # Short-term Momentum (5-day)
    mom_5d = (close / close.shift(5) - 1)
    
    # Medium-term Momentum (20-day)
    mom_20d = (close / close.shift(20) - 1)
    
    # Momentum Acceleration
    mom_accel = mom_5d - mom_20d
    
    # Volatility Assessment and Adjustment
    # Daily Price Range
    daily_range = (df['high'] - df['low']) / close.shift(1)
    
    # Rolling Volatility Estimation (20-day std of daily returns)
    daily_returns = close.pct_change()
    vol_20d = daily_returns.rolling(window=20).std()
    
    # Volatility-Adjusted Momentum
    vol_adj_mom_5d = mom_5d / vol_20d.replace(0, np.nan)
    vol_adj_mom_20d = mom_20d / vol_20d.replace(0, np.nan)
    vol_adj_mom_accel = mom_accel / vol_20d.replace(0, np.nan)
    
    # Asymmetric Volume Analysis
    # Classify Market Regime
    price_change = close.diff()
    bullish_days = price_change > 0
    bearish_days = price_change < 0
    
    volume = df['volume']
    
    # Calculate Volume Asymmetry (20-day rolling)
    up_volume_avg = volume.rolling(window=20).apply(
        lambda x: x[bullish_days.iloc[-len(x):].values].mean() if len(x[bullish_days.iloc[-len(x):].values]) > 0 else np.nan,
        raw=False
    )
    
    down_volume_avg = volume.rolling(window=20).apply(
        lambda x: x[bearish_days.iloc[-len(x):].values].mean() if len(x[bearish_days.iloc[-len(x):].values]) > 0 else np.nan,
        raw=False
    )
    
    volume_ratio = up_volume_avg / down_volume_avg.replace(0, np.nan)
    
    # Volume-Confirmed Momentum
    volume_confirmation = np.where(bullish_days, volume_ratio, 
                                  np.where(bearish_days, 1/volume_ratio.replace(0, np.nan), 1))
    
    vol_vol_adj_mom_5d = vol_adj_mom_5d * volume_confirmation
    vol_vol_adj_mom_20d = vol_adj_mom_20d * volume_confirmation
    
    # Signal Integration and Enhancement
    # Regime-Based Signal Amplification
    volatility_regime = daily_range.rolling(window=5).mean() / daily_range.rolling(window=20).mean()
    
    # Weight short-term vs medium-term momentum based on volatility regime
    short_term_weight = np.where(volatility_regime > 1, 0.7, 0.3)
    medium_term_weight = 1 - short_term_weight
    
    # Combine Adjusted Components
    combined_momentum = (short_term_weight * vol_vol_adj_mom_5d + 
                        medium_term_weight * vol_vol_adj_mom_20d + 
                        0.2 * vol_adj_mom_accel)
    
    # Apply volume confirmation multiplier
    final_alpha = combined_momentum * volume_confirmation
    
    # Handle NaN values
    final_alpha = final_alpha.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    return pd.Series(final_alpha, index=df.index)
