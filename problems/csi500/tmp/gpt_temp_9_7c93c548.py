import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Generate a regime-adaptive composite alpha factor combining momentum, mean reversion,
    order flow, and breakout quality signals with dynamic threshold scaling.
    """
    # Copy data to avoid modifying original
    data = df.copy()
    
    # Calculate basic price features
    data['returns'] = data['close'].pct_change()
    data['high_low_range'] = (data['high'] - data['low']) / data['close']
    data['true_range'] = np.maximum(
        data['high'] - data['low'],
        np.maximum(
            abs(data['high'] - data['close'].shift(1)),
            abs(data['low'] - data['close'].shift(1))
        )
    )
    
    # Volatility regime calculation
    data['volatility_15d'] = data['returns'].rolling(window=15, min_periods=10).std()
    data['volatility_regime'] = np.where(data['volatility_15d'] > data['volatility_15d'].rolling(window=30).median(), 'high', 'low')
    
    # Volume regime calculation
    data['volume_10d_avg'] = data['volume'].rolling(window=10, min_periods=5).mean()
    data['volume_regime'] = np.where(data['volume'] > data['volume_10d_avg'], 'high', 'low')
    
    # 1. Regime-Adaptive Momentum Factor
    # Multi-timeframe momentum
    data['momentum_3d'] = (data['close'] - data['close'].shift(3)) / data['true_range'].rolling(window=5).mean()
    data['momentum_10d'] = (data['close'] - data['close'].shift(10)) / (data['high'].rolling(window=10).max() - data['low'].rolling(window=10).min())
    data['momentum_accel'] = data['momentum_3d'] - data['momentum_3d'].shift(1)
    
    # Momentum regime classification
    data['momentum_alignment'] = np.sign(data['momentum_3d']) * np.sign(data['momentum_10d'])
    data['momentum_strength'] = (abs(data['momentum_3d']) + abs(data['momentum_10d'])) / 2
    
    # Volume confirmation
    data['volume_trend'] = data['volume'].rolling(window=3).apply(lambda x: 1 if (x.diff().iloc[1:] > 0).all() else (-1 if (x.diff().iloc[1:] < 0).all() else 0))
    data['volume_momentum_alignment'] = np.sign(data['momentum_3d']) * data['volume_trend']
    
    # Dynamic momentum scoring
    momentum_factor = (
        data['momentum_3d'] * 0.4 +
        data['momentum_10d'] * 0.3 +
        data['momentum_alignment'] * data['momentum_strength'] * 0.2 +
        data['volume_momentum_alignment'] * 0.1
    )
    
    # Volatility scaling
    momentum_factor = momentum_factor / data['volatility_15d'].replace(0, 1)
    
    # 2. Dynamic Mean Reversion Factor
    # Price extremity
    data['ma_5d'] = data['close'].rolling(window=5).mean()
    data['range_mid_10d'] = (data['high'].rolling(window=10).max() + data['low'].rolling(window=10).min()) / 2
    data['deviation_5d'] = (data['close'] - data['ma_5d']) / data['true_range'].rolling(window=5).mean()
    data['deviation_10d'] = (data['close'] - data['range_mid_10d']) / (data['high'].rolling(window=10).max() - data['low'].rolling(window=10).min())
    
    # Combined extremity score
    data['extremity_score'] = (abs(data['deviation_5d']) + abs(data['deviation_10d'])) / 2
    
    # Liquidity integration
    data['liquidity_score'] = (
        (data['volume'] / data['volume_10d_avg']) * 0.6 +
        (1 - data['high_low_range'] / data['high_low_range'].rolling(window=10).mean()) * 0.4
    )
    
    # Mean reversion factor
    mean_reversion_factor = -data['deviation_5d'] * data['liquidity_score'] * data['extremity_score']
    
    # 3. Smart Order Flow Momentum Factor
    # Directional amount flow
    data['amount_flow'] = data['amount'] * np.sign(data['returns'])
    data['flow_3d_avg'] = data['amount_flow'].rolling(window=3).mean()
    data['flow_10d_avg'] = data['amount_flow'].rolling(window=10).mean()
    
    # Flow efficiency
    data['flow_efficiency'] = data['amount'] / data['volume'].replace(0, 1)
    data['flow_efficiency_trend'] = data['flow_efficiency'].rolling(window=5).apply(lambda x: 1 if x.iloc[-1] > x.mean() else -1)
    
    # Flow-price convergence
    data['flow_momentum_alignment'] = np.sign(data['amount_flow'] - data['flow_10d_avg']) * np.sign(data['momentum_3d'])
    
    # Smart flow factor
    flow_factor = (
        (data['amount_flow'] - data['flow_10d_avg']) / data['flow_10d_avg'].abs().replace(0, 1) * 0.5 +
        data['flow_efficiency_trend'] * 0.3 +
        data['flow_momentum_alignment'] * 0.2
    )
    
    # 4. Breakout Quality Confirmation Factor
    # Range expansion
    data['range_expansion'] = data['true_range'] / data['true_range'].rolling(window=8).mean() - 1
    data['range_persistence'] = (data['range_expansion'] > 0).rolling(window=3).sum() / 3
    
    # Volume breakout
    data['volume_spike'] = data['volume'] / data['volume_10d_avg'] - 1
    data['volume_breakout'] = (data['volume_spike'] > data['volume_spike'].rolling(window=20).quantile(0.7)).astype(int)
    
    # Flow breakout
    data['flow_breakout'] = (abs(data['amount_flow'] - data['flow_10d_avg']) / data['flow_10d_avg'].abs().replace(0, 1) > 0.5).astype(int)
    
    # Multi-dimensional breakout confirmation
    breakout_factor = (
        data['range_expansion'] * 0.4 +
        data['volume_breakout'] * data['volume_spike'] * 0.3 +
        data['flow_breakout'] * np.sign(data['amount_flow']) * 0.3
    ) * data['range_persistence']
    
    # Volatility adjustment for breakout factor
    breakout_factor = breakout_factor / (1 + data['volatility_15d'])
    
    # 5. Composite Factor with Regime Weighting
    # Regime-based weighting
    high_vol_weight = np.where(data['volatility_regime'] == 'high', 0.3, 0.5)
    low_vol_weight = np.where(data['volatility_regime'] == 'low', 0.5, 0.3)
    
    high_volume_weight = np.where(data['volume_regime'] == 'high', 0.6, 0.4)
    
    # Final composite factor
    composite_factor = (
        momentum_factor * high_vol_weight * 0.3 +
        mean_reversion_factor * low_vol_weight * 0.25 +
        flow_factor * high_volume_weight * 0.25 +
        breakout_factor * 0.2
    )
    
    # Normalize and clean
    composite_factor = (composite_factor - composite_factor.rolling(window=50, min_periods=30).mean()) / composite_factor.rolling(window=50, min_periods=30).std()
    composite_factor = composite_factor.replace([np.inf, -np.inf], np.nan).fillna(method='ffill').fillna(0)
    
    return pd.Series(composite_factor, index=data.index, name='composite_alpha_factor')
