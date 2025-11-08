import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    data = df.copy()
    
    # Price-Volume Divergence Momentum
    price_return_5 = data['close'] / data['close'].shift(5) - 1
    volume_return_5 = data['volume'] / data['volume'].shift(5) - 1
    
    bullish_div = (price_return_5 < 0) & (volume_return_5 > 0)
    bearish_div = (price_return_5 > 0) & (volume_return_5 < 0)
    
    divergence_momentum = np.zeros(len(data))
    divergence_momentum[bullish_div] = (np.abs(price_return_5[bullish_div]) * np.abs(volume_return_5[bullish_div]))
    divergence_momentum[bearish_div] = -(np.abs(price_return_5[bearish_div]) * np.abs(volume_return_5[bearish_div]))
    
    # Gap Filling Propensity
    overnight_gap = data['open'] / data['close'].shift(1) - 1
    
    gap_filling = np.zeros(len(data))
    gap_up = overnight_gap > 0
    gap_down = overnight_gap < 0
    
    # Gap up filling: distance from low to previous close
    gap_filling[gap_up] = -overnight_gap[gap_up] * ((data['close'].shift(1)[gap_up] - data['low'][gap_up]) / 
                                                   (data['open'][gap_up] - data['close'].shift(1)[gap_up])).clip(0, 1)
    
    # Gap down filling: distance from high to previous close  
    gap_filling[gap_down] = -overnight_gap[gap_down] * ((data['high'][gap_down] - data['close'].shift(1)[gap_down]) / 
                                                       (data['close'].shift(1)[gap_down] - data['open'][gap_down])).clip(0, 1)
    
    # Regime-Sensitive Mean Reversion
    mean_20 = data['close'].rolling(window=20, min_periods=10).mean()
    std_20 = data['close'].rolling(window=20, min_periods=10).std()
    z_score = (data['close'] - mean_20) / std_20
    
    # Market regime identification
    trend_50 = data['close'].rolling(window=50, min_periods=25).apply(lambda x: (x[-1] - x[0]) / x[0] if len(x) == 50 else np.nan)
    volatility_50 = data['close'].rolling(window=50, min_periods=25).std() / data['close'].rolling(window=50, min_periods=25).mean()
    
    mean_reverting_regime = (np.abs(trend_50) < 0.05) & (volatility_50 > 0.02)
    regime_adjustment = np.where(mean_reverting_regime, 2.0, 1.0)
    
    mean_reversion = -z_score * regime_adjustment
    
    # Volume-Weighted Price Acceleration
    close_change = data['close'].diff()
    price_acceleration = close_change.diff()
    volume_weight = data['volume'] / data['volume'].rolling(window=5, min_periods=3).mean()
    price_accel_weighted = price_acceleration * volume_weight
    
    # Support/Resistance Breakout Efficiency
    resistance_10 = data['high'].rolling(window=10, min_periods=5).max().shift(1)
    support_10 = data['low'].rolling(window=10, min_periods=5).min().shift(1)
    
    volume_mean_10 = data['volume'].rolling(window=10, min_periods=5).mean().shift(1)
    volume_ratio = data['volume'] / volume_mean_10
    
    # Breakout detection
    resistance_breakout = (data['close'] > resistance_10) & (volume_ratio > 1.2)
    support_breakout = (data['close'] < support_10) & (volume_ratio > 1.2)
    
    breakout_strength = np.zeros(len(data))
    breakout_strength[resistance_breakout] = ((data['close'][resistance_breakout] - resistance_10[resistance_breakout]) / 
                                             resistance_10[resistance_breakout]) * volume_ratio[resistance_breakout]
    breakout_strength[support_breakout] = ((support_10[support_breakout] - data['close'][support_breakout]) / 
                                          support_10[support_breakout]) * volume_ratio[support_breakout]
    
    # Breakout persistence (3-day efficiency)
    breakout_persistence = np.zeros(len(data))
    for i in range(2, len(data)):
        if resistance_breakout.iloc[i] or support_breakout.iloc[i]:
            # Check if breakout persists for next 2 days (using only past data)
            if i + 2 < len(data):
                persistence_count = 0
                for j in range(1, 3):
                    if resistance_breakout.iloc[i]:
                        if data['close'].iloc[i+j] > resistance_10.iloc[i]:
                            persistence_count += 1
                    elif support_breakout.iloc[i]:
                        if data['close'].iloc[i+j] < support_10.iloc[i]:
                            persistence_count += 1
                breakout_persistence[i] = persistence_count / 2.0
    
    breakout_efficiency = breakout_strength * breakout_persistence
    
    # Combine all factors with equal weights
    factor = (divergence_momentum + gap_filling + mean_reversion + 
              price_accel_weighted + breakout_efficiency) / 5
    
    return pd.Series(factor, index=data.index)
