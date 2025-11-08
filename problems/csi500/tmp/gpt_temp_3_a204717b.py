import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(data):
    """
    Adaptive Momentum-Volume Regime Factor
    Combines multi-timeframe momentum with volume persistence analysis
    and adaptive exponential weighting based on volatility regimes
    """
    df = data.copy()
    epsilon = 1e-8
    
    # Multi-Timeframe Momentum Extraction
    # Intraday Momentum Component
    df['intraday_return'] = (df['close'] - df['open']) / (df['high'] - df['low'] + epsilon)
    df['momentum_direction'] = np.sign(df['close'] - df['open'])
    df['momentum_strength'] = np.abs(df['intraday_return'])
    
    # Multi-Day Momentum Component
    df['momentum_3d'] = (df['close'] - df['close'].shift(3)) / (
        df['high'].rolling(4).max() - df['low'].rolling(4).min() + epsilon
    )
    df['momentum_5d'] = (df['close'] - df['close'].shift(5)) / (
        df['high'].rolling(6).max() - df['low'].rolling(6).min() + epsilon
    )
    
    # Momentum Consistency
    df['momentum_sign_3d'] = np.sign(df['momentum_3d'])
    df['momentum_sign_5d'] = np.sign(df['momentum_5d'])
    df['momentum_consistency'] = (
        (df['momentum_direction'] == df['momentum_sign_3d']).astype(int) + 
        (df['momentum_direction'] == df['momentum_sign_5d']).astype(int)
    )
    
    # Volume Persistence Analysis
    df['volume_ratio'] = df['volume'] / (df['volume'].shift(1) + epsilon)
    df['volume_direction'] = np.sign(df['volume'] - df['volume'].shift(1))
    
    # Volume persistence (consecutive days with same direction)
    volume_dir_changes = df['volume_direction'] != df['volume_direction'].shift(1)
    df['volume_persistence'] = volume_dir_changes.groupby(volume_dir_changes.cumsum()).cumcount() + 1
    
    # Volume-Momentum Alignment
    df['directional_alignment'] = df['momentum_direction'] * df['volume_direction']
    df['strength_alignment'] = df['momentum_strength'] * df['volume_persistence']
    df['confirmation_score'] = df['directional_alignment'] * df['strength_alignment']
    
    # Volatility Regime Detection
    df['recent_volatility'] = (df['high'] - df['low']).rolling(5).mean()
    volatility_median = df['recent_volatility'].median()
    df['volatility_regime'] = np.where(df['recent_volatility'] > volatility_median, 'high', 'low')
    
    # Adaptive Exponential Weighting
    def apply_adaptive_weighting(series, regime_col, lookback=10):
        weighted_values = []
        for i in range(len(series)):
            if i < lookback:
                weighted_values.append(np.nan)
                continue
                
            window_data = series.iloc[i-lookback:i]
            window_regime = regime_col.iloc[i-lookback:i]
            
            weights = []
            values = []
            
            for j, (val, regime) in enumerate(zip(window_data, window_regime)):
                if pd.isna(val):
                    continue
                    
                days_ago = lookback - j - 1
                if regime == 'high':
                    decay_rate = 0.15
                else:
                    decay_rate = 0.05
                    
                weight = np.exp(-decay_rate * days_ago)
                weights.append(weight)
                values.append(val)
            
            if weights:
                weighted_avg = np.average(values, weights=weights)
                weighted_values.append(weighted_avg)
            else:
                weighted_values.append(np.nan)
        
        return pd.Series(weighted_values, index=series.index)
    
    df['weighted_momentum'] = apply_adaptive_weighting(df['momentum_strength'], df['volatility_regime'])
    df['weighted_volume'] = apply_adaptive_weighting(df['volume_persistence'], df['volatility_regime'])
    
    # Range-Based Scaling
    df['intraday_range'] = df['high'] - df['low']
    df['multi_day_range'] = df['high'].rolling(6).max() - df['low'].rolling(6).min()
    
    # Directional Alignment Filtering
    df['cross_timeframe_agreement'] = (
        (df['momentum_direction'] == df['momentum_sign_3d']) & 
        (df['momentum_direction'] == df['momentum_sign_5d'])
    ).astype(int)
    
    df['volume_confirmation'] = (df['directional_alignment'] > 0).astype(int)
    df['filter_strength'] = df['cross_timeframe_agreement'] + df['volume_confirmation'] + df['momentum_consistency']
    
    # Final Factor Construction
    df['base_factor'] = df['weighted_momentum'] * df['confirmation_score']
    
    # Regime adjustment
    regime_multiplier = np.where(df['volatility_regime'] == 'high', 1.2, 0.8)
    df['regime_adjusted'] = df['base_factor'] * regime_multiplier
    
    # Alignment boost
    alignment_multiplier = 1 + (df['filter_strength'] * 0.1)
    df['final_factor'] = df['regime_adjusted'] * alignment_multiplier
    
    return df['final_factor']
