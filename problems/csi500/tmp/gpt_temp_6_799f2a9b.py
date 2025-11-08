import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(data):
    """
    Regime-Adaptive Asymmetric Microstructure Momentum factor
    """
    df = data.copy()
    
    # Calculate True Range
    df['TR'] = np.maximum(df['high'] - df['low'], 
                         np.maximum(abs(df['high'] - df['close'].shift(1)), 
                                   abs(df['low'] - df['close'].shift(1))))
    
    # Asymmetric Momentum Dynamics
    # Directional Momentum Divergence
    close_3d = df['close'] / df['close'].shift(3) - 1
    close_8d = df['close'] / df['close'].shift(8) - 1
    price_direction = (df['close'] - df['close'].shift(3)) > 0
    df['directional_momentum_div'] = np.where(price_direction, 
                                             close_3d - close_8d, 
                                             -(close_3d - close_8d))
    
    # Volume momentum asymmetry
    vol_3d = df['volume'] / df['volume'].shift(3) - 1
    vol_8d = df['volume'] / df['volume'].shift(8) - 1
    df['volume_momentum_asym'] = np.where(price_direction, 
                                         vol_3d - vol_8d, 
                                         -(vol_3d - vol_8d))
    
    # Range momentum asymmetry
    tr_3d = df['TR'] / df['TR'].shift(3) - 1
    tr_8d = df['TR'] / df['TR'].shift(8) - 1
    df['range_momentum_asym'] = np.where(price_direction, 
                                        tr_3d - tr_8d, 
                                        -(tr_3d - tr_8d))
    
    # Asymmetric Microstructure Quality
    # Up-move quality vs down-move quality
    price_change = abs(df['close'] - df['close'].shift(1))
    daily_range = df['high'] - df['low']
    directional_weight = np.where(df['close'] > df['close'].shift(1), 1, -1)
    df['microstructure_quality'] = (price_change / daily_range) * directional_weight
    
    # Price rejection asymmetry
    upper_rejection = (df['high'] - np.maximum(df['open'], df['close'])) / daily_range
    lower_rejection = (np.minimum(df['open'], df['close']) - df['low']) / daily_range
    df['price_rejection_asym'] = upper_rejection - lower_rejection
    
    # Gap efficiency asymmetry
    gap_low = np.minimum(df['open'], df['close'].shift(1))
    gap_high = np.maximum(df['open'], df['close'].shift(1))
    gap_range = gap_high - gap_low
    gap_efficiency = (df['close'] - gap_low) / gap_range
    df['gap_efficiency_asym'] = np.where(df['close'] > df['open'], 
                                        gap_efficiency, 
                                        -(1 - gap_efficiency))
    
    # Asymmetric Volume-Volatility Alignment
    # Volume-volatility correlation asymmetry
    vol_5d = df['volume'] / df['volume'].shift(5) - 1
    vol_10d = df['volume'] / df['volume'].shift(10) - 1
    vol_trend_sign = np.sign(vol_5d - vol_10d)
    
    # Rolling volatility ratios
    df['volatility_5d'] = df['close'].rolling(window=5).std()
    df['volatility_10d'] = df['close'].rolling(window=10).std()
    vol_ratio = df['volatility_5d'] / df['volatility_10d'].shift(5) - 1
    
    df['volume_volatility_asym'] = vol_trend_sign * vol_ratio * np.where(price_direction, 1, -1)
    
    # Large trade momentum asymmetry
    amount_3d = df['amount'] / df['amount'].shift(3) - 1
    amount_8d = df['amount'] / df['amount'].shift(8) - 1
    df['large_trade_asym'] = np.where(price_direction, 
                                     amount_3d - amount_8d, 
                                     -(amount_3d - amount_8d))
    
    # Value-weighted momentum divergence
    price_change_3d = df['close'] - df['close'].shift(3)
    price_change_8d = df['close'] - df['close'].shift(8)
    df['value_weighted_momentum'] = (price_change_3d * df['amount'] / (abs(price_change_3d) + 1e-8) - 
                                    price_change_8d * df['amount'] / (abs(price_change_8d) + 1e-8))
    
    # Regime-Switching Asymmetry Patterns
    # Volatility Regime Classification
    df['ATR_5d'] = df['TR'].rolling(window=5).mean()
    df['ATR_15d'] = df['TR'].rolling(window=15).mean()
    df['range_ratio'] = df['ATR_5d'] / df['ATR_15d']
    
    # Regime states based on Recent Range Ratio
    regime_quantiles = df['range_ratio'].rolling(window=50, min_periods=20).apply(
        lambda x: pd.qcut(x, q=[0, 0.3, 0.7, 1.0], labels=False, duplicates='drop').iloc[-1] 
        if len(x.dropna()) >= 20 else np.nan, raw=False
    )
    
    # Multi-Timeframe Asymmetry Convergence
    # Short-term asymmetric momentum (1-3 days)
    df['short_term_momentum'] = (df['close'] / df['close'].shift(1) - 1) * directional_weight
    
    # Medium-term asymmetric momentum (5-10 days)
    close_5d = df['close'] / df['close'].shift(5) - 1
    close_10d = df['close'] / df['close'].shift(10) - 1
    df['medium_term_momentum'] = np.where(close_5d > close_10d, 
                                         close_5d - close_10d, 
                                         -(close_5d - close_10d))
    
    # Volume-Value Asymmetric Confirmation
    # Asymmetric Amount Efficiency
    daily_return = df['close'] - df['close'].shift(1)
    df['directional_amount_momentum'] = daily_return * df['amount'] / (abs(daily_return) + 1e-8)
    
    # Value-weighted range efficiency
    df['value_range_efficiency'] = df['amount'] / daily_range * directional_weight
    
    # Behavioral Asymmetry in Momentum
    # Reference Point Momentum Asymmetry
    df['prev_close_anchor'] = (df['close'] - df['close'].shift(1)) / df['close'].shift(1) * directional_weight
    df['open_momentum_memory'] = (df['close'] - df['open']) / df['open'] * np.where(df['close'] > df['open'], 1, -1)
    
    # Range Boundary Momentum Behavior
    mid_range = (df['high'] + df['low']) / 2
    df['range_boundary_momentum'] = (df['close'] - mid_range) / (df['high'] - df['low']) * directional_weight
    
    # Integrated Regime-Adaptive Asymmetry Factor
    # Combine all asymmetry components
    momentum_components = [
        'directional_momentum_div', 'volume_momentum_asym', 'range_momentum_asym',
        'microstructure_quality', 'price_rejection_asym', 'gap_efficiency_asym',
        'volume_volatility_asym', 'large_trade_asym', 'value_weighted_momentum',
        'short_term_momentum', 'medium_term_momentum', 'directional_amount_momentum',
        'value_range_efficiency', 'prev_close_anchor', 'open_momentum_memory',
        'range_boundary_momentum'
    ]
    
    # Calculate raw composite score
    raw_composite = df[momentum_components].mean(axis=1, skipna=True)
    
    # Apply regime-adaptive weighting
    high_vol_regime = (regime_quantiles == 2)  # Top 30%
    low_vol_regime = (regime_quantiles == 0)   # Bottom 30%
    
    # Emphasize different components based on volatility regime
    if high_vol_regime.any():
        vol_weighted = (df['directional_momentum_div'] * 0.3 + 
                       df['volume_volatility_asym'] * 0.3 + 
                       df['range_momentum_asym'] * 0.2 + 
                       raw_composite * 0.2)
        raw_composite = np.where(high_vol_regime, vol_weighted, raw_composite)
    
    if low_vol_regime.any():
        quality_weighted = (df['microstructure_quality'] * 0.4 + 
                           df['price_rejection_asym'] * 0.3 + 
                           df['gap_efficiency_asym'] * 0.3)
        raw_composite = np.where(low_vol_regime, quality_weighted, raw_composite)
    
    # Final factor output
    factor = raw_composite
    
    return factor
