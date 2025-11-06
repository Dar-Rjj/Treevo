import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Asymmetric Price-Volume Dynamics with Regime-Dependent Memory factor
    """
    # Make copy to avoid modifying original data
    data = df.copy()
    
    # Ensure data is sorted by date
    data = data.sort_index()
    
    # Pre-calculate basic components
    data['close_ret'] = data['close'] / data['close'].shift(1) - 1
    data['volume_ret'] = data['volume'] / data['volume'].shift(1) - 1
    data['range'] = data['high'] - data['low']
    data['close_to_low'] = data['close'] - data['low']
    data['high_to_close'] = data['high'] - data['close']
    data['body'] = data['close'] - data['open']
    
    # Directional Asymmetry Analysis
    # Up-Move Characteristics
    data['price_acceleration'] = (data['close_ret'] / data['close_ret'].shift(1)).replace([np.inf, -np.inf], np.nan)
    data['volume_pressure'] = data['volume_ret'] * (data['high_to_close'] / data['range'].replace(0, np.nan))
    data['gap_persistence'] = ((data['open'] - data['close'].shift(1)) / data['close'].shift(1)) * (data['body'] / data['close'])
    
    # Down-Move Characteristics  
    data['price_deceleration'] = data['price_acceleration']  # Same calculation, different interpretation
    data['volume_support'] = data['volume_ret'] * (data['close_to_low'] / data['range'].replace(0, np.nan))
    data['recovery_potential'] = (data['close_to_low'] / data['range'].replace(0, np.nan)) * (data['volume'] / data['volume'].shift(1))
    
    # Sideways Behavior
    data['compression_intensity'] = (data['range'] / data['close']) * (data['volume'] / data['volume'].rolling(3).mean())
    data['breakout_readiness'] = (abs(data['close'] - (data['high'] + data['low'])/2) / data['range'].replace(0, np.nan)) * (data['volume'] / data['volume'].shift(1))
    data['range_efficiency'] = (data['body'] / data['range'].replace(0, np.nan)) * (data['volume'] / data['volume'].shift(1))
    
    # Memory-Dependent Regime Classification
    # Short Memory (1-3 days)
    data['momentum_persistence'] = data['close_ret'] * data['close_ret'].shift(1)
    data['volume_memory'] = (data['volume'] / data['volume'].shift(1)) * (data['volume'].shift(1) / data['volume'].shift(2))
    data['range_expansion'] = (data['range'] / data['range'].shift(1).replace(0, np.nan)) * (data['volume'] / data['volume'].shift(1))
    
    # Medium Memory (5-10 days)
    data['trend_consistency'] = ((data['close'] / data['close'].shift(5) - 1) * 
                                (data['close'].shift(5) / data['close'].shift(10) - 1))
    
    vol_ma5 = data['volume'].rolling(5).mean()
    vol_ma10 = data['volume'].rolling(10).mean()
    data['volume_regime'] = (data['volume'] / vol_ma5) * (vol_ma5 / vol_ma10)
    
    data['volatility_memory'] = ((data['range'] / data['close']) * 
                                (data['range'].shift(5) / data['close'].shift(5)))
    
    # Long Memory (20-40 days)
    data['structural_momentum'] = ((data['close'] / data['close'].shift(20) - 1) * 
                                  (data['close'].shift(20) / data['close'].shift(40) - 1))
    
    vol_ma20 = data['volume'].rolling(20).mean()
    vol_ma40 = data['volume'].rolling(40).mean()
    data['volume_structure'] = (data['volume'] / vol_ma20) * (vol_ma20 / vol_ma40)
    
    data['range_evolution'] = ((data['range'] / data['close']) * 
                              (data['range'].shift(20) / data['close'].shift(20)))
    
    # Regime-Adaptive Signal Generation
    # Trending Regime Signals
    data['strong_uptrend'] = ((data['price_acceleration'] > 0) & (data['volume_pressure'] > 0)).astype(float)
    data['weak_uptrend'] = ((data['price_acceleration'] < 0) & (data['close_ret'] > 0)).astype(float)
    data['strong_downtrend'] = ((data['price_deceleration'] < 0) & (data['volume_support'] < 0)).astype(float)
    data['weak_downtrend'] = ((data['price_deceleration'] > 0) & (data['close_ret'] < 0)).astype(float)
    
    # Ranging Regime Signals
    compression_rank = data['compression_intensity'].rolling(10).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)
    breakout_rank = data['breakout_readiness'].rolling(10).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)
    efficiency_rank = data['range_efficiency'].rolling(10).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)
    
    data['breakout_potential'] = ((compression_rank > 0.7) & (breakout_rank.diff() > 0)).astype(float)
    data['range_continuation'] = ((efficiency_rank < 0.3) & (compression_rank.diff().abs() < 0.1)).astype(float)
    data['false_breakout'] = ((breakout_rank > 0.7) & (efficiency_rank < 0.3)).astype(float)
    
    # Memory-Weighted Signal Integration
    # Calculate memory weights
    short_momentum = data['momentum_persistence'].rolling(3).mean()
    medium_momentum = data['trend_consistency'].rolling(5).mean()
    long_momentum = data['structural_momentum'].rolling(10).mean()
    
    # Signal validation and enhancement
    # Asymmetry Confirmation
    data['up_down_differential'] = (data['volume_pressure'].rolling(5).mean() - 
                                   data['volume_support'].rolling(5).mean())
    
    # Regime Coherence
    uptrend_signals = data['strong_uptrend'] + data['weak_uptrend']
    downtrend_signals = data['strong_downtrend'] + data['weak_downtrend']
    range_signals = data['breakout_potential'] + data['range_continuation'] + data['false_breakout']
    
    data['regime_coherence'] = (uptrend_signals.rolling(3).std().fillna(0) + 
                               downtrend_signals.rolling(3).std().fillna(0) + 
                               range_signals.rolling(3).std().fillna(0))
    
    # Final Alpha Factor Construction
    # Core components
    trend_component = (data['strong_uptrend'] - data['strong_downtrend'] + 
                      0.5 * (data['weak_uptrend'] - data['weak_downtrend']))
    
    range_component = (data['breakout_potential'] - 0.5 * data['false_breakout'] - 
                      0.3 * data['range_continuation'])
    
    # Memory alignment
    memory_alignment = (np.sign(short_momentum) * np.sign(medium_momentum) * np.sign(long_momentum) *
                       (abs(short_momentum) + 0.7 * abs(medium_momentum) + 0.3 * abs(long_momentum)))
    
    # Asymmetric Memory-Regime Factor
    alpha_factor = (
        0.4 * trend_component +
        0.3 * range_component +
        0.2 * data['up_down_differential'] +
        0.1 * memory_alignment -
        0.05 * data['regime_coherence']
    )
    
    # Volume reliability adjustment
    volume_reliability = data['volume'] / data['volume'].rolling(20).mean()
    alpha_factor = alpha_factor * np.minimum(volume_reliability, 2.0)
    
    # Clean up and return
    alpha_series = alpha_factor.replace([np.inf, -np.inf], np.nan).fillna(0)
    return alpha_series
