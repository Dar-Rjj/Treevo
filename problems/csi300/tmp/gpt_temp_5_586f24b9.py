import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Daily Price Efficiency
    df['daily_efficiency'] = (df['close'] - df['open']) / (df['high'] - df['low'])
    df['daily_efficiency'] = df['daily_efficiency'].replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Multi-Timeframe Efficiency Sums
    df['eff_sum_3d'] = df['daily_efficiency'].rolling(window=3, min_periods=1).sum()
    df['eff_sum_8d'] = df['daily_efficiency'].rolling(window=8, min_periods=1).sum()
    df['eff_sum_15d'] = df['daily_efficiency'].rolling(window=15, min_periods=1).sum()
    
    # Efficiency Momentum Divergence
    df['eff_div_short'] = df['eff_sum_3d'] / df['eff_sum_8d']
    df['eff_div_medium'] = df['eff_sum_8d'] / df['eff_sum_15d']
    df['eff_div_short'] = df['eff_div_short'].replace([np.inf, -np.inf], np.nan).fillna(0)
    df['eff_div_medium'] = df['eff_div_medium'].replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Efficiency Signal (sign-adjusted product)
    df['efficiency_signal'] = df['eff_div_short'] * df['eff_div_medium'] * np.sign(df['eff_div_short'])
    
    # Volume Distribution Confirmation
    # Volume Asymmetry Ratio (5-day window)
    df['up_day'] = df['close'] > df['open']
    df['down_day'] = df['close'] < df['open']
    
    up_volume = df['volume'].rolling(window=5, min_periods=1).apply(
        lambda x: np.sum(x[df['up_day'].iloc[-len(x):].values]), raw=False
    )
    down_volume = df['volume'].rolling(window=5, min_periods=1).apply(
        lambda x: np.sum(x[df['down_day'].iloc[-len(x):].values]), raw=False
    )
    
    df['volume_asymmetry'] = up_volume / down_volume
    df['volume_asymmetry'] = df['volume_asymmetry'].replace([np.inf, -np.inf], np.nan).fillna(1)
    
    # Volume Efficiency Momentum
    eff_volume_5d = (df['daily_efficiency'] * df['volume']).rolling(window=5, min_periods=1).sum()
    eff_volume_15d = (df['daily_efficiency'] * df['volume']).rolling(window=15, min_periods=1).sum()
    df['volume_eff_momentum'] = eff_volume_5d / eff_volume_15d
    df['volume_eff_momentum'] = df['volume_eff_momentum'].replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Volume Multiplier
    conditions = [
        (df['volume_asymmetry'] > 1.2) & (df['volume_eff_momentum'] > 0),
        (df['volume_asymmetry'] < 0.8) & (df['volume_eff_momentum'] < 0),
        ((df['volume_asymmetry'] > 1.2) & (df['volume_eff_momentum'] < 0)) | 
        ((df['volume_asymmetry'] < 0.8) & (df['volume_eff_momentum'] > 0))
    ]
    choices = [1.4, 1.4, 0.6]
    df['volume_multiplier'] = np.select(conditions, choices, default=1.0)
    
    # Price Momentum Integration
    df['price_trend_3d'] = df['close'] / df['close'].shift(3) - 1
    df['price_trend_8d'] = df['close'] / df['close'].shift(8) - 1
    df['price_trend_15d'] = df['close'] / df['close'].shift(15) - 1
    
    # Handle NaN values from shifts
    df['price_trend_3d'] = df['price_trend_3d'].fillna(0)
    df['price_trend_8d'] = df['price_trend_8d'].fillna(0)
    df['price_trend_15d'] = df['price_trend_15d'].fillna(0)
    
    # Momentum Divergence
    df['momentum_div_short'] = (df['price_trend_3d'] - df['price_trend_8d']) / (df['price_trend_3d'].abs() + 1e-8)
    df['momentum_div_medium'] = (df['price_trend_8d'] - df['price_trend_15d']) / (df['price_trend_8d'].abs() + 1e-8)
    
    df['momentum_div_short'] = df['momentum_div_short'].replace([np.inf, -np.inf], np.nan).fillna(0)
    df['momentum_div_medium'] = df['momentum_div_medium'].replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Consistency Multiplier
    same_sign = (
        (df['price_trend_3d'] > 0) & (df['price_trend_8d'] > 0) & (df['price_trend_15d'] > 0) |
        (df['price_trend_3d'] < 0) & (df['price_trend_8d'] < 0) & (df['price_trend_15d'] < 0)
    )
    df['consistency_multiplier'] = np.where(same_sign, 1.3, 0.8)
    
    # Composite Alpha Factor
    avg_momentum_div = (df['momentum_div_short'] + df['momentum_div_medium']) / 2
    df['core_signal'] = df['efficiency_signal'] * avg_momentum_div
    
    df['volume_enhanced'] = df['core_signal'] * df['volume_multiplier'] * df['consistency_multiplier']
    
    # Final normalization by average volume
    avg_volume_5d = df['volume'].rolling(window=5, min_periods=1).mean()
    df['final_factor'] = df['volume_enhanced'] / (avg_volume_5d + 1e-8)
    
    # Clean up intermediate columns
    result = df['final_factor'].copy()
    
    return result
