import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Multi-Scale Gap Efficiency with Microstructure Alignment alpha factor
    """
    data = df.copy()
    
    # Gap Efficiency Analysis
    # Multi-Timeframe Gap Components
    data['overnight_gap'] = data['open'] / data['close'].shift(1) - 1
    data['intraday_gap'] = data['close'] / data['open'] - 1
    
    # Gap Persistence: 3-day direction streak
    data['gap_direction'] = np.sign(data['overnight_gap'])
    data['gap_streak'] = data['gap_direction'].groupby(data.index).transform(
        lambda x: x * (x == x.shift(1)).astype(int).groupby((x != x.shift(1)).cumsum()).cumsum() + 1
    )
    
    # Gap Efficiency Measurement
    data['intraday_gap_efficiency'] = (data['close'] - data['open']) / (data['high'] - data['low'])
    
    # Short-Term Gap Efficiency (3-day)
    data['high_3d'] = data['high'].rolling(window=3, min_periods=1).max()
    data['low_3d'] = data['low'].rolling(window=3, min_periods=1).min()
    data['short_term_gap_efficiency'] = (data['close'] - data['close'].shift(3)) / (data['high_3d'] - data['low_3d'])
    
    # Long-Term Gap Efficiency (10-day)
    data['high_10d'] = data['high'].rolling(window=10, min_periods=1).max()
    data['low_10d'] = data['low'].rolling(window=10, min_periods=1).min()
    data['long_term_gap_efficiency'] = (data['close'] - data['close'].shift(10)) / (data['high_10d'] - data['low_10d'])
    
    # Gap Efficiency Divergence
    data['gap_efficiency_divergence'] = data['short_term_gap_efficiency'] - data['long_term_gap_efficiency']
    
    # Volume Microstructure Integration
    # Gap Absorption Analysis
    data['opening_gap_absorption'] = (data['open'] - data['close'].shift(1)) / (data['high'] - data['low'])
    data['closing_pressure'] = (data['close'] - (data['high'] + data['low'])/2) / (data['high'] - data['low'])
    data['volume_weighted_gap_momentum'] = (data['close'] - data['close'].shift(3)) * data['volume']
    
    # Volume Distribution Efficiency
    data['volume_persistence'] = data['volume'] / data['volume'].shift(1)
    data['volume_concentration'] = data['volume'] / (data['volume'].rolling(window=5, min_periods=1).mean())
    
    # Volatility-Regime Adaptive Framework
    # True Range Calculation
    data['true_range'] = np.maximum(
        data['high'] - data['low'],
        np.maximum(
            abs(data['high'] - data['close'].shift(1)),
            abs(data['low'] - data['close'].shift(1))
        )
    )
    data['atr'] = data['true_range'].rolling(window=14, min_periods=1).mean()
    data['volatility_regime'] = data['atr'] / data['atr'].shift(10)
    
    # Price Impact Analysis
    data['price_impact'] = (data['high'] - data['low']) / data['volume']
    
    # Multi-Dimensional Gap Convergence
    # Gap Efficiency Quality Scoring
    data['gap_efficiency_quality'] = data['gap_efficiency_divergence'] * data['volume_persistence']
    
    # Order Flow Imbalance approximation
    data['order_flow_imbalance'] = (2 * data['close'] - data['low'] - data['high']) / (data['high'] - data['low'])
    
    # Integrate microstructure efficiency
    data['microstructure_efficiency'] = data['gap_efficiency_quality'] * data['order_flow_imbalance']
    
    # Volatility-regime adaptive weighting
    data['volatility_weight'] = np.where(
        data['volatility_regime'] > 1.2,  # High volatility
        data['gap_efficiency_divergence'] * data['gap_streak'],
        data['gap_efficiency_divergence'] * data['volatility_regime']  # Low volatility
    )
    
    # Microstructure convergence multiplier
    data['microstructure_convergence'] = (
        data['opening_gap_absorption'] * data['closing_pressure'] * 
        data['volume_concentration']
    )
    
    # Comprehensive gap efficiency score
    data['gap_alignment_score'] = (
        data['intraday_gap_efficiency'] * 
        data['short_term_gap_efficiency'] * 
        data['gap_efficiency_divergence']
    )
    
    # Final Alpha Signal Generation
    alpha_signal = (
        data['gap_efficiency_quality'] * 0.3 +
        data['microstructure_efficiency'] * 0.25 +
        data['volatility_weight'] * 0.2 +
        data['microstructure_convergence'] * 0.15 +
        data['gap_alignment_score'] * 0.1
    )
    
    # Clean up intermediate columns
    result = alpha_signal.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    return result
