import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Multi-Timeframe Momentum Analysis
    df['momentum_10d'] = (df['close'] - df['close'].shift(10)) / df['close'].shift(10)
    df['momentum_20d'] = (df['close'] - df['close'].shift(20)) / df['close'].shift(20)
    df['momentum_acceleration'] = df['momentum_10d'] - df['momentum_20d']
    
    # Volume-Price Alignment System
    df['volume_trend'] = (df['volume'] - df['volume'].shift(5)) / df['volume'].shift(5)
    
    def get_volume_confirmation_score(volume_trend, momentum_20d):
        vol_sign = np.sign(volume_trend)
        mom_sign = np.sign(momentum_20d)
        vol_mag = abs(volume_trend)
        mom_mag = abs(momentum_20d)
        
        # Strong alignment: same signs and both magnitudes > 0.05
        if vol_sign == mom_sign and vol_mag > 0.05 and mom_mag > 0.05:
            return 1.3
        # Divergence: opposite signs and both magnitudes > 0.1
        elif vol_sign != mom_sign and vol_mag > 0.1 and mom_mag > 0.1:
            return 0.5
        # Weak alignment: all other cases
        else:
            return 0.9
    
    df['volume_confirmation'] = df.apply(
        lambda x: get_volume_confirmation_score(x['volume_trend'], x['momentum_20d']), 
        axis=1
    )
    
    # Robust Trend Persistence Framework
    def calculate_direction_persistence(momentum_20d):
        persistence = 0
        current_sign = np.sign(momentum_20d)
        count = 0
        
        for i in range(len(momentum_20d)):
            if i == 0 or np.sign(momentum_20d.iloc[i]) != current_sign:
                current_sign = np.sign(momentum_20d.iloc[i])
                count = 1
            else:
                count += 1
            persistence += (0.8 ** (count - 1))
        
        return persistence
    
    # Calculate rolling direction persistence
    df['direction_persistence'] = df['momentum_20d'].rolling(window=20, min_periods=1).apply(
        lambda x: calculate_direction_persistence(pd.Series(x)), raw=False
    )
    
    # Momentum stability assessment
    df['momentum_iqr'] = df['momentum_10d'].rolling(window=15).apply(
        lambda x: np.percentile(x, 75) - np.percentile(x, 25), raw=True
    )
    df['momentum_stability'] = 1 / (df['momentum_iqr'] + 0.0001)
    
    # Combined persistence metric
    df['persistence_metric'] = (
        df['direction_persistence'] * 
        df['momentum_stability'] * 
        abs(df['momentum_20d'])
    )
    
    # Volatility-Adjusted Range Analysis
    df['daily_range_volatility'] = (df['high'] - df['low']) / df['close']
    df['range_efficiency'] = (df['close'] - df['open']) / (df['high'] - df['low'])
    
    def get_efficiency_multiplier(range_efficiency):
        eff_abs = abs(range_efficiency)
        if eff_abs > 0.7:
            return 1.2
        elif eff_abs >= 0.3:
            return 1.0
        else:
            return 0.8
    
    df['efficiency_multiplier'] = df['range_efficiency'].apply(get_efficiency_multiplier)
    
    # Final Alpha Construction
    # Core momentum base with acceleration modifier
    core_momentum = df['momentum_20d'] * (1 + df['momentum_acceleration'])
    
    # Volume-confirmed signal
    volume_confirmed = core_momentum * df['volume_confirmation']
    
    # Persistence-enhanced signal
    persistence_enhanced = volume_confirmed * df['persistence_metric']
    
    # Volatility-optimized alpha factor
    alpha_factor = (persistence_enhanced * df['efficiency_multiplier']) / (df['daily_range_volatility'] + 0.0001)
    
    return alpha_factor
