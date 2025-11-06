import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Price Momentum with Volatility Adjustment
    # Raw momentum: (close_t - close_t-1) / (high_t - low_t)
    raw_momentum = (data['close'] - data['close'].shift(1)) / (data['high'] - data['low'])
    
    # Momentum persistence: count of positive raw momentum in last 3 days
    momentum_persistence = pd.Series(index=data.index, dtype=float)
    for i in range(len(data)):
        if i >= 3:
            window = raw_momentum.iloc[i-3:i]
            momentum_persistence.iloc[i] = (window > 0).sum()
        else:
            momentum_persistence.iloc[i] = np.nan
    
    # Volatility-scaled momentum: raw momentum * (high_t-1 - low_t-1) / (high_t - low_t)
    vol_ratio = (data['high'].shift(1) - data['low'].shift(1)) / (data['high'] - data['low'])
    volatility_scaled_momentum = raw_momentum * vol_ratio
    
    # Volume-Weighted Price Action
    # Volume intensity: volume_t / (volume_t-1 + volume_t-2 + volume_t-3)
    volume_intensity = data['volume'] / (data['volume'].shift(1) + data['volume'].shift(2) + data['volume'].shift(3))
    
    # Price-volume correlation: sign(close_t - close_t-1) * sign(volume_t - volume_t-1)
    price_volume_corr = np.sign(data['close'] - data['close'].shift(1)) * np.sign(data['volume'] - data['volume'].shift(1))
    
    # Weighted price move: (close_t - close_t-1) * volume_t / amount_t
    weighted_price_move = (data['close'] - data['close'].shift(1)) * data['volume'] / data['amount']
    
    # Regime-Dependent Scaling Factors
    # Volatility regime: (high_t - low_t) / (high_t-5 - low_t-5)
    volatility_regime = (data['high'] - data['low']) / (data['high'].shift(5) - data['low'].shift(5))
    
    # Volume regime: volume_t / (volume_t-5 + volume_t-6 + volume_t-7 + volume_t-8 + volume_t-9)
    volume_regime = data['volume'] / (data['volume'].shift(5) + data['volume'].shift(6) + 
                                    data['volume'].shift(7) + data['volume'].shift(8) + data['volume'].shift(9))
    
    # Regime multiplier: volatility regime * volume regime
    regime_multiplier = volatility_regime * volume_regime
    
    # Multi-Timeframe Factor Integration
    # Short-term component: volatility-scaled momentum * price-volume correlation
    short_term_component = volatility_scaled_momentum * price_volume_corr
    
    # Medium-term component: momentum persistence * weighted price move
    medium_term_component = momentum_persistence * weighted_price_move
    
    # Final factor: (short-term component + medium-term component) * regime multiplier
    final_factor = (short_term_component + medium_term_component) * regime_multiplier
    
    return final_factor
