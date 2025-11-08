import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Multi-Scale Order Flow Imbalance & Price Efficiency Framework
    Generates a composite alpha factor combining order flow microstructure analysis
    with multi-timeframe price efficiency metrics
    """
    data = df.copy()
    
    # Order Flow Microstructure Analysis
    # Bidirectional Order Flow Imbalance
    data['upward_flow'] = np.where(data['close'] > data['open'], data['volume'], 0)
    data['downward_flow'] = np.where(data['close'] < data['open'], data['volume'], 0)
    data['net_flow_imbalance'] = (data['upward_flow'] - data['downward_flow']) / (data['volume'] + 1e-8)
    
    # Order flow persistence
    data['flow_direction'] = np.sign(data['net_flow_imbalance'])
    data['flow_persistence'] = data['flow_direction'].groupby(data.index).expanding().apply(
        lambda x: (x == x.iloc[-1]).sum() if len(x) > 0 else 0
    ).reset_index(level=0, drop=True)
    
    # Flow magnitude asymmetry
    data['flow_asymmetry'] = (data['upward_flow'] + 1e-8) / (data['downward_flow'] + 1e-8)
    
    # Order Flow Velocity Dynamics
    data['flow_acceleration'] = data['net_flow_imbalance'].diff()
    data['flow_momentum_3d'] = data['net_flow_imbalance'].rolling(window=3, min_periods=1).mean()
    data['flow_volatility'] = data['net_flow_imbalance'].rolling(window=5, min_periods=1).std()
    
    # Flow-return efficiency
    data['daily_return'] = (data['close'] - data['open']) / data['open']
    data['flow_return_efficiency'] = data['daily_return'] / (abs(data['net_flow_imbalance']) + 1e-8)
    
    # Multi-Timeframe Price Efficiency Metrics
    # Intraday Efficiency Spectrum
    data['opening_efficiency'] = abs(data['close'] - data['open']) / (data['high'] - data['low'] + 1e-8)
    data['high_low_utilization'] = (data['close'] - data['low']) / (data['high'] - data['low'] + 1e-8)
    
    # Range compression efficiency
    daily_range = data['high'] - data['low']
    movement = abs(data['close'] - data['open'])
    data['range_compression_efficiency'] = movement / (daily_range + 1e-8)
    
    # Efficiency momentum and trends
    data['efficiency_momentum'] = data['opening_efficiency'].diff()
    data['efficiency_trend_3d'] = data['opening_efficiency'].rolling(window=3, min_periods=1).mean()
    data['efficiency_volatility'] = data['opening_efficiency'].rolling(window=5, min_periods=1).std()
    
    # Efficiency persistence
    data['efficiency_level'] = np.where(data['opening_efficiency'] > data['opening_efficiency'].rolling(window=10, min_periods=1).mean(), 1, -1)
    data['efficiency_persistence'] = data['efficiency_level'].groupby(data.index).expanding().apply(
        lambda x: (x == x.iloc[-1]).sum() if len(x) > 0 else 0
    ).reset_index(level=0, drop=True)
    
    # Flow-Efficiency Interaction Framework
    # Order Flow Efficiency Confirmation
    high_efficiency_threshold = data['opening_efficiency'].rolling(window=20, min_periods=1).quantile(0.7)
    high_flow_threshold = data['net_flow_imbalance'].abs().rolling(window=20, min_periods=1).quantile(0.7)
    
    data['high_efficiency'] = data['opening_efficiency'] > high_efficiency_threshold
    data['high_flow'] = data['net_flow_imbalance'].abs() > high_flow_threshold
    
    # Flow-efficiency alignment scoring
    data['flow_efficiency_alignment'] = 0
    data.loc[data['high_flow'] & data['high_efficiency'], 'flow_efficiency_alignment'] = 1  # Strong directional
    data.loc[~data['high_flow'] & data['high_efficiency'], 'flow_efficiency_alignment'] = 2  # Efficient discovery
    data.loc[data['high_flow'] & ~data['high_efficiency'], 'flow_efficiency_alignment'] = -1  # Contested levels
    
    # Efficiency-Driven Flow Patterns
    data['efficiency_breakout_flow'] = data['flow_acceleration'] * data['efficiency_momentum']
    data['efficiency_flow_momentum'] = data['flow_momentum_3d'] * data['efficiency_trend_3d']
    
    # Multi-Scale Factor Integration
    # Timeframe-specific components
    data['intraday_signal'] = (
        data['net_flow_imbalance'] * data['opening_efficiency'] * 
        np.sign(data['flow_efficiency_alignment'])
    )
    
    data['short_term_signal'] = (
        data['flow_momentum_3d'] * data['efficiency_trend_3d'] * 
        data['flow_persistence'].clip(upper=5) / 5
    )
    
    # Regime detection
    efficiency_regime = data['opening_efficiency'].rolling(window=10, min_periods=1).mean()
    flow_regime = data['net_flow_imbalance'].abs().rolling(window=10, min_periods=1).mean()
    
    # Regime-adaptive weighting
    data['regime_weight'] = np.where(
        efficiency_regime > efficiency_regime.rolling(window=20, min_periods=1).quantile(0.6),
        0.6,  # High efficiency: emphasize flow confirmation
        np.where(
            flow_regime > flow_regime.rolling(window=20, min_periods=1).quantile(0.7),
            0.8,  # High flow: focus on breakout potential
            0.4   # Normal: balanced approach
        )
    )
    
    # Composite Alpha Generation
    # Flow-Efficiency Scoring System
    flow_strength = data['net_flow_imbalance'] * data['flow_persistence'].clip(upper=5) / 5
    efficiency_score = data['opening_efficiency'] * data['efficiency_persistence'].clip(upper=5) / 5
    alignment_consistency = data['flow_efficiency_alignment'].rolling(window=3, min_periods=1).mean()
    
    # Multi-timeframe integration
    data['composite_alpha'] = (
        data['regime_weight'] * flow_strength * efficiency_score *
        alignment_consistency * np.sign(data['flow_efficiency_alignment']) +
        (1 - data['regime_weight']) * data['intraday_signal'] *
        data['short_term_signal']
    )
    
    # Signal Quality Assessment and Refinement
    recent_performance = data['composite_alpha'].rolling(window=5, min_periods=1).std()
    signal_strength = data['composite_alpha'].abs()
    
    # Final alpha with dynamic refinement
    alpha_factor = (
        data['composite_alpha'] * 
        (1 / (recent_performance + 1e-8)) *  # Normalize by recent volatility
        np.tanh(signal_strength / signal_strength.rolling(window=10, min_periods=1).mean())  # Smooth extreme values
    )
    
    return alpha_factor
