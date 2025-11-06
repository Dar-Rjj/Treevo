import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Multi-Scale Momentum with Volume-Microstructure Enhancement factor
    """
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Market return proxy (equal-weighted market)
    data['market_return'] = data['close'].pct_change()
    
    # Residual Momentum Calculation
    # Short-term (5-day) residual momentum
    data['stock_return_5d'] = data['close'].pct_change(5)
    data['market_return_5d'] = data['market_return'].rolling(5).sum()
    data['residual_momentum_5d'] = data['stock_return_5d'] - data['market_return_5d']
    
    # Medium-term (20-day) residual momentum
    data['stock_return_20d'] = data['close'].pct_change(20)
    data['market_return_20d'] = data['market_return'].rolling(20).sum()
    data['residual_momentum_20d'] = data['stock_return_20d'] - data['market_return_20d']
    
    # Momentum Quality Assessment
    # Momentum Acceleration
    data['momentum_acceleration'] = data['residual_momentum_5d'] / (data['residual_momentum_20d'] + 1e-8)
    
    # Directional Consistency
    data['directional_consistency'] = np.sign(data['residual_momentum_5d']) * np.sign(data['residual_momentum_20d'])
    
    # Volume Confirmation
    data['volume_slope_5d'] = data['volume'].rolling(5).apply(lambda x: np.polyfit(range(5), x, 1)[0])
    data['volume_momentum_alignment'] = np.sign(data['residual_momentum_5d']) * np.sign(data['volume_slope_5d'])
    
    # Microstructure Analysis - Volatility Environment
    # True Range calculation
    data['tr1'] = data['high'] - data['low']
    data['tr2'] = abs(data['high'] - data['close'].shift(1))
    data['tr3'] = abs(data['low'] - data['close'].shift(1))
    data['true_range'] = data[['tr1', 'tr2', 'tr3']].max(axis=1)
    
    # Volatility measures
    data['volatility_5d'] = data['true_range'].rolling(5).mean()
    data['volatility_20d'] = data['true_range'].rolling(20).mean()
    data['volatility_ratio'] = data['volatility_5d'] / (data['volatility_20d'] + 1e-8)
    
    # Order Flow Dynamics
    data['price_change'] = data['close'] - data['open']
    data['signed_volume'] = np.sign(data['price_change']) * data['volume']
    data['order_flow'] = data['signed_volume'] * abs(data['price_change'])
    data['order_flow_persistence'] = data['order_flow'] / (data['order_flow'].rolling(10).mean() + 1e-8)
    
    # Regime-Adaptive Processing
    # Volatility regime classification
    data['high_vol_regime'] = (data['volatility_ratio'] > 1.2).astype(int)
    data['low_vol_regime'] = (data['volatility_ratio'] < 0.8).astype(int)
    
    # High Volatility: volatility-adjusted momentum with divergence detection
    volatility_adjustment = 1 / (data['volatility_5d'] + 1e-8)
    data['high_vol_momentum'] = data['residual_momentum_5d'] * volatility_adjustment
    data['momentum_divergence'] = data['residual_momentum_5d'] - data['residual_momentum_20d']
    
    # Low Volatility: breakout detection with trend fracture analysis
    data['price_range'] = (data['high'] - data['low']) / data['close']
    data['breakout_signal'] = (data['price_range'] > data['price_range'].rolling(10).mean() * 1.5).astype(int)
    data['trend_fracture'] = abs(data['residual_momentum_5d'] - data['residual_momentum_20d'].shift(5))
    
    # Composite Factor Construction
    # Quality-weighted momentum
    quality_weight = (0.4 * data['directional_consistency'] + 
                      0.3 * data['volume_momentum_alignment'] + 
                      0.3 * np.tanh(data['momentum_acceleration']))
    
    quality_weighted_momentum = data['residual_momentum_5d'] * quality_weight
    
    # Microstructure-enhanced signals with regime awareness
    microstructure_signal = (0.5 * np.tanh(data['order_flow_persistence']) + 
                             0.3 * np.tanh(data['volatility_ratio'] - 1) + 
                             0.2 * np.tanh(data['momentum_divergence']))
    
    # Regime-specific adjustments
    regime_adjustment = (data['high_vol_regime'] * data['high_vol_momentum'] + 
                         data['low_vol_regime'] * data['breakout_signal'] * data['trend_fracture'])
    
    # Final composite factor
    composite_factor = (0.6 * quality_weighted_momentum + 
                        0.3 * microstructure_signal + 
                        0.1 * regime_adjustment)
    
    # Ensure no future data leakage
    result = composite_factor.copy()
    
    # Clean up intermediate columns
    cols_to_drop = ['market_return', 'stock_return_5d', 'market_return_5d', 'stock_return_20d', 
                   'market_return_20d', 'tr1', 'tr2', 'tr3', 'true_range']
    for col in cols_to_drop:
        if col in data.columns:
            data.drop(col, axis=1, inplace=True)
    
    return result
