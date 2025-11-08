import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Intraday Momentum Decay Factor with regime-aware components
    """
    data = df.copy()
    
    # Calculate Intraday Momentum Strength
    # Price-based Momentum
    data['intraday_return'] = (data['close'] - data['close'].shift(1)) / data['close'].shift(1)
    data['normalized_range'] = (data['high'] - data['low']) / data['close'].shift(1)
    
    # Momentum Persistence
    # 3-day Momentum Consistency
    data['momentum_3d'] = data['close'].pct_change(3)
    data['momentum_sign_agreement'] = (
        (data['intraday_return'].shift(1) > 0).astype(int) + 
        (data['intraday_return'].shift(2) > 0).astype(int) + 
        (data['intraday_return'] > 0).astype(int)
    ) / 3.0
    
    # Exponential decay weighting for recent momentum
    weights = np.array([0.5, 0.3, 0.2])  # Recent emphasis
    data['decay_adjusted_momentum'] = (
        weights[0] * data['intraday_return'] + 
        weights[1] * data['intraday_return'].shift(1) + 
        weights[2] * data['intraday_return'].shift(2)
    )
    
    # Volume Acceleration Component
    # Volume Trend Strength
    data['volume_ratio'] = data['volume'] / data['volume'].shift(1)
    data['volume_slope_5d'] = data['volume'].rolling(5).apply(
        lambda x: np.polyfit(range(5), x, 1)[0] if not x.isnull().any() else np.nan
    )
    
    # Liquidity Efficiency
    data['amount_per_volume'] = data['amount'] / data['volume']
    data['efficiency_trend'] = data['amount_per_volume'].pct_change(3)
    
    # Volume-Price Correlation (10-day)
    data['volume_price_corr'] = data['close'].rolling(10).corr(data['volume'])
    data['direction_alignment'] = np.sign(data['intraday_return']) * np.sign(data['volume_ratio'])
    
    # Volatility-Scaled Signal
    # Adaptive Volatility Measure
    data['vol_5d'] = data['close'].pct_change().rolling(5).std()
    data['vol_20d'] = data['close'].pct_change().rolling(20).std()
    data['vol_ratio'] = data['vol_5d'] / data['vol_20d']
    
    # Volatility regime classification
    data['vol_regime'] = np.where(data['vol_ratio'] > 1.2, 'high', 
                                 np.where(data['vol_ratio'] < 0.8, 'low', 'normal'))
    
    # Recent Volatility Trend
    data['vol_acceleration'] = data['vol_5d'].pct_change(3)
    data['vol_mean_reversion'] = (data['vol_5d'] - data['vol_20d']) / data['vol_20d']
    
    # Final Factor Construction
    # Momentum-Volume Blend with regime-dependent weights
    momentum_component = (
        0.6 * data['decay_adjusted_momentum'] + 
        0.4 * data['momentum_sign_agreement']
    )
    
    volume_component = (
        0.5 * data['volume_slope_5d'] + 
        0.3 * data['direction_alignment'] + 
        0.2 * data['efficiency_trend']
    )
    
    # Regime-dependent weights
    high_vol_weight = 0.3
    normal_vol_weight = 0.5
    low_vol_weight = 0.7
    
    data['momentum_volume_blend'] = np.where(
        data['vol_regime'] == 'high', 
        high_vol_weight * momentum_component + (1 - high_vol_weight) * volume_component,
        np.where(
            data['vol_regime'] == 'low',
            low_vol_weight * momentum_component + (1 - low_vol_weight) * volume_component,
            normal_vol_weight * momentum_component + (1 - normal_vol_weight) * volume_component
        )
    )
    
    # Volatility Scaling
    # Use appropriate volatility measure based on regime
    data['scaling_vol'] = np.where(
        data['vol_regime'] == 'high', data['vol_5d'],
        np.where(data['vol_regime'] == 'low', data['vol_20d'], 
                (data['vol_5d'] + data['vol_20d']) / 2)
    )
    
    # Final factor with volatility scaling
    data['factor'] = data['momentum_volume_blend'] / data['scaling_vol']
    
    # Normalize by volatility regime
    regime_normalizers = data.groupby('vol_regime')['factor'].transform(
        lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0
    )
    data['factor_normalized'] = regime_normalizers
    
    return data['factor_normalized']
