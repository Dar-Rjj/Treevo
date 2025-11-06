import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(data):
    """
    Regime-Adaptive Momentum & Mean Reversion Composite factor
    """
    df = data.copy()
    
    # Momentum Acceleration Component
    # Multi-timeframe Acceleration
    df['ret_1d'] = df['close'] / df['close'].shift(1) - 1
    df['ret_2d'] = df['close'] / df['close'].shift(2) - 1
    df['ret_5d'] = df['close'] / df['close'].shift(5) - 1
    df['ret_8d'] = df['close'] / df['close'].shift(8) - 1
    
    df['accel_ultra'] = df['ret_2d'] - df['ret_1d']
    df['accel_short'] = df['ret_5d'] - df['ret_2d']
    df['accel_medium'] = df['ret_8d'] - df['ret_5d']
    
    # Volume-Weighted Momentum
    df['vol_3d_sum'] = df['volume'].rolling(window=3).sum()
    df['vol_7d_sum'] = df['volume'].rolling(window=7).sum()
    df['vol_intensity'] = df['vol_3d_sum'] / df['vol_7d_sum']
    
    df['accel_ultra_vol'] = df['accel_ultra'] * df['vol_intensity']
    df['accel_short_vol'] = df['accel_short'] * df['vol_intensity']
    df['accel_medium_vol'] = df['accel_medium'] * df['vol_intensity']
    
    # Composite momentum score
    momentum_score = (0.4 * df['accel_ultra_vol'] + 
                     0.35 * df['accel_short_vol'] + 
                     0.25 * df['accel_medium_vol'])
    
    # Mean Reversion Component
    # Price Extremes Detection
    df['high_8d'] = df['high'].rolling(window=8).max()
    df['low_8d'] = df['low'].rolling(window=8).min()
    df['position'] = (df['close'] - df['low_8d']) / (df['high_8d'] - df['low_8d'])
    
    # Volume-Confirmed Reversion
    df['vol_median_8d'] = df['volume'].rolling(window=8).median()
    df['vol_spike'] = df['volume'] > (1.5 * df['vol_median_8d'])
    
    df['midpoint'] = (df['high_8d'] + df['low_8d']) / 2
    df['norm_distance'] = (df['close'] - df['midpoint']) / (df['high_8d'] - df['low_8d'])
    
    df['vol_multiplier'] = np.minimum(df['volume'] / df['vol_median_8d'], 2.0)
    base_reversion = -df['norm_distance']
    df['reversion_vol'] = base_reversion * df['vol_multiplier']
    
    # Time-Decay Reversion
    df['days_since_high'] = (df['high'] == df['high_8d']).astype(int)
    df['days_since_low'] = (df['low'] == df['low_8d']).astype(int)
    
    # Calculate rolling days since extreme
    def days_since_extreme(series):
        days = 0
        result = []
        for val in series:
            if val == 1:
                days = 0
            else:
                days += 1
            result.append(days)
        return pd.Series(result, index=series.index)
    
    df['days_since_high_roll'] = days_since_extreme(df['days_since_high'])
    df['days_since_low_roll'] = days_since_extreme(df['days_since_low'])
    
    df['decay_weight_high'] = np.exp(-df['days_since_high_roll'] / 3)
    df['decay_weight_low'] = np.exp(-df['days_since_low_roll'] / 3)
    
    # Apply decay weights to reversion signals
    high_extreme_reversion = df['reversion_vol'] * (df['position'] > 0.8) * df['decay_weight_high']
    low_extreme_reversion = df['reversion_vol'] * (df['position'] < 0.2) * df['decay_weight_low']
    
    mean_reversion_score = high_extreme_reversion + low_extreme_reversion
    
    # Volatility Conditioning
    # True Range and ATR
    df['tr1'] = df['high'] - df['low']
    df['tr2'] = abs(df['high'] - df['close'].shift(1))
    df['tr3'] = abs(df['low'] - df['close'].shift(1))
    df['true_range'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
    df['atr_5d'] = df['true_range'].rolling(window=5).mean()
    
    # Risk-adjusted factors
    vol_adj = df['atr_5d'] / df['close']
    momentum_vol_adj = momentum_score / vol_adj
    mean_reversion_vol_adj = mean_reversion_score / vol_adj
    
    # Volatility Regime Detection
    df['atr_median_15d'] = df['atr_5d'].rolling(window=15).median()
    high_vol_regime = df['atr_5d'] > (1.3 * df['atr_median_15d'])
    
    # Apply volatility regime adjustments
    momentum_base = momentum_vol_adj.copy()
    mean_reversion_base = mean_reversion_vol_adj.copy()
    
    momentum_base[high_vol_regime] = momentum_base[high_vol_regime] * 0.7
    mean_reversion_base[high_vol_regime] = mean_reversion_base[high_vol_regime] * 1.2
    
    # Dynamic Regime Integration
    # Market State Classification
    df['range_ratio'] = (df['high_8d'] - df['low_8d']) / df['close']
    
    # Trend consistency measure
    df['ret_sign'] = np.sign(df['ret_1d'])
    df['consistency_5d'] = df['ret_sign'].rolling(window=5).apply(
        lambda x: sum(x == x.iloc[-1]) if len(x) == 5 else np.nan
    )
    
    strong_trend = (df['consistency_5d'] >= 4) & (df['range_ratio'] > df['range_ratio'].rolling(window=20).quantile(0.7))
    
    # Volume regime
    df['vol_median_15d'] = df['volume'].rolling(window=15).median()
    high_volume = df['volume'] > (1.4 * df['vol_median_15d'])
    low_volume = df['volume'] < (0.7 * df['vol_median_15d'])
    normal_volume = ~high_volume & ~low_volume
    
    # Component Weighting based on regime
    momentum_weight = pd.Series(0.5, index=df.index)
    mean_reversion_weight = pd.Series(0.5, index=df.index)
    regime_multiplier = pd.Series(1.0, index=df.index)
    
    # Trending market regime
    trend_mask = strong_trend & high_volume
    momentum_weight[trend_mask] = 0.7
    mean_reversion_weight[trend_mask] = 0.3
    
    # Mean-reverting market regime
    mean_rev_mask = (~strong_trend) & (high_volume | normal_volume)
    momentum_weight[mean_rev_mask] = 0.3
    mean_reversion_weight[mean_rev_mask] = 0.7
    
    # Low-volume regime
    low_vol_mask = low_volume
    momentum_weight[low_vol_mask] = 0.4
    mean_reversion_weight[low_vol_mask] = 0.6
    regime_multiplier[low_vol_mask] = 0.75
    
    # Final Alpha Factor
    final_factor = (momentum_base * momentum_weight + 
                   mean_reversion_base * mean_reversion_weight) * regime_multiplier
    
    return final_factor
