import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Multi-Timeframe Efficiency-Momentum Velocity with Adaptive Regime Integration
    """
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Calculate returns
    df['returns'] = df['close'].pct_change()
    df['abs_returns'] = df['returns'].abs()
    
    # Multi-Timeframe Efficiency Analysis
    # Directional efficiency calculation
    for window in [5, 20, 60]:
        df[f'return_{window}d'] = df['close'].pct_change(window)
        df[f'abs_return_sum_{window}d'] = df['abs_returns'].rolling(window).sum()
        df[f'efficiency_{window}d'] = df[f'return_{window}d'] / (df[f'abs_return_sum_{window}d'] + 1e-8)
    
    # Efficiency momentum (velocity)
    df['efficiency_momentum_5d'] = df['efficiency_5d'] - df['efficiency_5d'].shift(5)
    df['efficiency_acceleration_20d'] = (df['efficiency_20d'] - df['efficiency_20d'].shift(10)) - (df['efficiency_20d'].shift(10) - df['efficiency_20d'].shift(20))
    
    # Efficiency trend persistence
    df['efficiency_trend_5d'] = df['efficiency_5d'].rolling(5).apply(lambda x: np.corrcoef(range(len(x)), x)[0,1] if len(x) > 1 else 0, raw=True)
    df['efficiency_trend_20d'] = df['efficiency_20d'].rolling(20).apply(lambda x: np.corrcoef(range(len(x)), x)[0,1] if len(x) > 1 else 0, raw=True)
    
    # Cross-timeframe efficiency alignment
    df['efficiency_alignment_sm'] = df['efficiency_5d'] * df['efficiency_20d']
    df['efficiency_alignment_ml'] = df['efficiency_20d'] * df['efficiency_60d']
    df['efficiency_convergence'] = (df['efficiency_5d'] + df['efficiency_20d'] + df['efficiency_60d']) / 3
    
    # Volume-Amount Efficiency Dynamics
    # Volume efficiency metrics
    df['price_change_per_volume'] = df['returns'] / (df['volume'] + 1e-8)
    df['volume_momentum'] = df['volume'].pct_change(5)
    df['volume_acceleration'] = df['volume_momentum'] - df['volume_momentum'].shift(5)
    
    # Amount-based order flow context
    df['avg_trade_size'] = df['amount'] / (df['volume'] + 1e-8)
    df['large_order_concentration'] = df['avg_trade_size'].rolling(10).std()
    df['amount_volume_divergence'] = df['amount'].pct_change(5) - df['volume'].pct_change(5)
    df['order_flow_efficiency'] = df['returns'] / (df['amount'] + 1e-8)
    
    # Volume-amount confirmation
    df['volume_efficiency_confirmation'] = df['price_change_per_volume'] * df['efficiency_5d']
    df['amount_momentum_confirmation'] = df['amount'].pct_change(5) * df['efficiency_momentum_5d']
    df['multi_measurement_alignment'] = (df['volume_efficiency_confirmation'] + df['amount_momentum_confirmation']) / 2
    
    # Regime-Adaptive Momentum Velocity
    # Volatility regime classification
    volatility_20d = df['returns'].rolling(20).std()
    df['volatility_regime'] = pd.cut(volatility_20d, 
                                   bins=[0, volatility_20d.quantile(0.33), volatility_20d.quantile(0.67), np.inf],
                                   labels=['low', 'transition', 'high'])
    
    # Regime-specific momentum calculation
    df['high_vol_momentum'] = np.where(df['volatility_regime'] == 'high', df['efficiency_momentum_5d'], 0)
    df['low_vol_momentum'] = np.where(df['volatility_regime'] == 'low', df['efficiency_momentum_5d'], 0)
    df['transition_momentum'] = np.where(df['volatility_regime'] == 'transition', df['efficiency_momentum_5d'], 0)
    
    # Momentum velocity analysis
    df['cross_timeframe_momentum'] = (df['efficiency_momentum_5d'] + df['efficiency_acceleration_20d']) / 2
    df['momentum_acceleration'] = df['efficiency_momentum_5d'] - df['efficiency_momentum_5d'].shift(5)
    
    # Regime-adaptive weighting
    regime_weights = {'high': 0.4, 'transition': 0.3, 'low': 0.3}
    df['regime_weighted_momentum'] = (df['high_vol_momentum'] * regime_weights['high'] + 
                                    df['transition_momentum'] * regime_weights['transition'] + 
                                    df['low_vol_momentum'] * regime_weights['low'])
    
    # Range Dynamics and Breakout Timing
    # Range efficiency assessment
    df['daily_range'] = (df['high'] - df['low']) / df['close']
    df['range_utilization'] = (df['close'] - df['open']) / (df['high'] - df['low'] + 1e-8)
    df['range_efficiency_avg'] = df['range_utilization'].rolling(10).mean()
    df['range_consistency'] = df['daily_range'].rolling(10).std()
    
    # Breakout quality metrics
    df['range_expansion'] = df['daily_range'] / df['daily_range'].rolling(10).mean()
    df['close_position_strength'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-8)
    df['breakout_confirmation'] = df['range_expansion'] * df['close_position_strength']
    
    # Compression-release analysis
    df['range_compression'] = df['daily_range'] / df['daily_range'].rolling(20).mean()
    df['compression_duration'] = (df['range_compression'] < 0.8).rolling(10).sum()
    df['release_probability'] = df['compression_duration'] * df['range_expansion']
    
    # Adaptive Signal Integration
    # ATR scaling for volatility context
    df['atr'] = (df['high'] - df['low']).rolling(14).mean()
    
    # Final factor calculation
    efficiency_component = (df['efficiency_convergence'] * 0.3 + 
                          df['cross_timeframe_momentum'] * 0.25 + 
                          df['regime_weighted_momentum'] * 0.2)
    
    volume_component = (df['multi_measurement_alignment'] * 0.15 + 
                       df['order_flow_efficiency'] * 0.1)
    
    range_component = (df['breakout_confirmation'] * 0.15 + 
                      df['release_probability'] * 0.1)
    
    # Volatility scaling
    volatility_scale = 1 / (df['atr'] + 1e-8)
    
    # Final factor with volatility context enhancement
    factor = (efficiency_component + volume_component + range_component) * volatility_scale
    
    return factor
