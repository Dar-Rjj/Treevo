import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Generate novel alpha factor combining Price-Volume Fractal Divergence, 
    Range Compression Breakout System, and Order Flow Momentum Composite
    """
    
    # Create copy to avoid modifying original data
    data = df.copy()
    
    # Price-Volume Fractal Divergence Component
    # Multi-Scale Momentum Patterns
    data['mom_3d'] = data['close'] / data['close'].shift(3) - 1
    data['mom_5d'] = data['close'] / data['close'].shift(5) - 1
    data['mom_10d'] = data['close'] / data['close'].shift(10) - 1
    
    # Momentum curvature (acceleration)
    data['mom_accel_3vs10'] = data['mom_3d'] - data['mom_10d']
    
    # Timeframe coherence (rolling correlation)
    data['mom_coherence'] = data['mom_3d'].rolling(window=5).corr(data['mom_10d'])
    
    # Volume Fractal Dynamics
    data['volume_intensity'] = data['volume'] / data['volume'].shift(1)
    
    # Volume persistence (consecutive increases)
    volume_increase = data['volume'] > data['volume'].shift(1)
    data['volume_persistence'] = volume_increase.rolling(window=5).apply(
        lambda x: x.astype(int).sum(), raw=False
    )
    
    # Volume-momentum divergence
    data['vol_mom_divergence'] = np.sign(data['mom_3d']) * data['volume_intensity']
    
    # Fractal Confirmation System
    data['fractal_momentum'] = (data['mom_accel_3vs10'] * 
                               data['volume_persistence'] * 
                               data['mom_coherence'].fillna(0))
    
    # Range Compression Breakout System
    # Range Dynamics Analysis
    daily_range = data['high'] - data['low']
    prev_range = data['high'].shift(1) - data['low'].shift(1)
    data['range_compression'] = daily_range / prev_range
    
    # Range persistence (consecutive compression days)
    range_compressed = daily_range < daily_range.shift(1)
    data['range_persistence'] = range_compressed.rolling(window=5).apply(
        lambda x: x.astype(int).sum(), raw=False
    )
    
    # Breakout potential
    min_range_5d = daily_range.rolling(window=5).min()
    data['breakout_potential'] = daily_range / min_range_5d
    
    # Breakout Detection Mechanism
    avg_range_5d = daily_range.rolling(window=5).mean()
    data['range_expansion'] = (daily_range > 1.5 * avg_range_5d).astype(int)
    
    # Breakout direction (position within day's range)
    range_position = (data['close'] - data['low']) / daily_range
    data['breakout_direction'] = np.where(range_position > 0.66, 1, 
                                         np.where(range_position < 0.33, -1, 0))
    
    # Volume expansion confirmation
    avg_volume_5d = data['volume'].rolling(window=5).mean()
    data['volume_expansion'] = data['volume'] / avg_volume_5d
    
    # Adaptive Breakout Scoring
    data['breakout_score'] = (data['range_expansion'] * 
                             data['breakout_direction'] * 
                             data['volume_expansion'] * 
                             data['range_persistence'])
    
    # Order Flow Momentum Composite
    # Amount-Based Flow Analysis
    data['amount_intensity'] = data['amount'] / data['amount'].shift(1)
    
    # Flow persistence (consecutive amount increases)
    amount_increase = data['amount'] > data['amount'].shift(1)
    data['flow_persistence'] = amount_increase.rolling(window=5).apply(
        lambda x: x.astype(int).sum(), raw=False
    )
    
    # Amount-price divergence
    data['amount_price_divergence'] = np.sign(data['mom_3d']) * data['amount_intensity']
    
    # Volume-Price Efficiency Metrics
    daily_price_move = data['close'] - data['open']
    data['price_efficiency'] = daily_price_move / daily_range
    
    # Volume concentration during price movement
    significant_move = abs(daily_price_move) > (daily_range * 0.3)
    data['volume_concentration'] = np.where(significant_move, data['volume'], 0)
    
    # Efficiency persistence
    data['efficiency_persistence'] = data['price_efficiency'].rolling(window=5).mean()
    
    # Composite Factor Construction
    data['order_flow_composite'] = (data['amount_intensity'] * 
                                   data['price_efficiency'] * 
                                   data['flow_persistence'] * 
                                   data['efficiency_persistence'])
    
    # Dynamic Alpha Integration
    # Fractal-Regime Adaptive Weighting based on range compression
    range_compression_level = daily_range.rolling(window=10).std() / daily_range.rolling(window=10).mean()
    
    # High range compression regime
    high_compression = range_compression_level < range_compression_level.quantile(0.3)
    # Low range compression regime  
    low_compression = range_compression_level > range_compression_level.quantile(0.7)
    
    # Initialize factor weights
    fractal_weight = np.ones(len(data)) * 0.33
    breakout_weight = np.ones(len(data)) * 0.33
    order_flow_weight = np.ones(len(data)) * 0.33
    
    # Adjust weights based on regime
    fractal_weight[high_compression] = 0.3
    breakout_weight[high_compression] = 0.5
    order_flow_weight[high_compression] = 0.2
    
    fractal_weight[low_compression] = 0.4
    breakout_weight[low_compression] = 0.25
    order_flow_weight[low_compression] = 0.35
    
    # Cross-Factor Validation System
    # Fractal momentum confirmed by order flow direction
    fractal_confirmed = (np.sign(data['fractal_momentum']) == np.sign(data['order_flow_composite']))
    # Breakout signals validated by volume fractal patterns
    breakout_confirmed = (data['range_expansion'] & (data['volume_intensity'] > 1))
    # Order flow efficiency supporting momentum sustainability
    flow_efficient = data['efficiency_persistence'] > 0
    
    # Signal Persistence Framework
    # Multi-timeframe momentum alignment
    mom_alignment = ((np.sign(data['mom_3d']) == np.sign(data['mom_5d'])) & 
                    (np.sign(data['mom_5d']) == np.sign(data['mom_10d'])))
    
    # Volume-amount flow convergence
    flow_convergence = (np.sign(data['volume_intensity'] - 1) == np.sign(data['amount_intensity'] - 1))
    
    # Range breakout confirmation across factors
    range_confirmation = (data['breakout_score'] != 0) & (data['fractal_momentum'] != 0)
    
    # Final composite factor with validation and persistence
    validated_fractal = data['fractal_momentum'] * fractal_confirmed.astype(float)
    validated_breakout = data['breakout_score'] * breakout_confirmed.astype(float)
    validated_order_flow = data['order_flow_composite'] * flow_efficient.astype(float)
    
    # Apply persistence multipliers
    persistence_multiplier = (mom_alignment.astype(float) + 
                             flow_convergence.astype(float) + 
                             range_confirmation.astype(float)) / 3
    
    # Final weighted composite
    composite_factor = (validated_fractal * fractal_weight + 
                       validated_breakout * breakout_weight + 
                       validated_order_flow * order_flow_weight) * persistence_multiplier
    
    # Normalize and return
    factor_series = pd.Series(composite_factor, index=data.index)
    factor_series = (factor_series - factor_series.rolling(window=20).mean()) / factor_series.rolling(window=20).std()
    
    return factor_series.fillna(0)
