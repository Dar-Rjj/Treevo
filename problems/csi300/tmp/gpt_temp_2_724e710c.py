import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Cross-Timeframe Momentum Efficiency with Volatility-Adjusted Gap Analysis
    """
    data = df.copy()
    
    # Multi-Timeframe Momentum Efficiency Analysis
    # Cross-Timeframe Momentum Divergence
    data['short_term_momentum'] = data['close'] / data['close'].shift(3)
    data['medium_term_momentum'] = data['close'] / data['close'].shift(10)
    data['momentum_divergence'] = data['short_term_momentum'] - data['medium_term_momentum']
    
    # Movement Efficiency
    data['price_change_per_volume'] = (data['close'] - data['close'].shift(1)) / data['volume']
    
    # 5-day baseline efficiency calculation
    baseline_efficiency = []
    for i in range(len(data)):
        if i >= 10:
            baseline = 0
            count = 0
            for j in range(1, 6):
                if i-j >= 0 and i-j-5 >= 0:
                    price_change = data['close'].iloc[i-j] - data['close'].iloc[i-j-5]
                    volume_val = data['volume'].iloc[i-j]
                    if volume_val > 0:
                        baseline += price_change / volume_val
                        count += 1
            baseline_efficiency.append(baseline / count if count > 0 else 1)
        else:
            baseline_efficiency.append(1)
    
    data['baseline_efficiency'] = baseline_efficiency
    data['efficiency_ratio'] = data['price_change_per_volume'] / data['baseline_efficiency']
    data['efficiency_ratio'] = data['efficiency_ratio'].replace([np.inf, -np.inf], 0).fillna(0)
    
    # Acceleration Patterns
    data['short_term_accel'] = (data['close']/data['close'].shift(3)) / (data['close'].shift(1)/data['close'].shift(4))
    data['medium_term_accel'] = (data['close']/data['close'].shift(10)) / (data['close'].shift(1)/data['close'].shift(11))
    
    # Gap Analysis with Volatility Context
    # Gap Identification and Measurement
    data['overnight_gap'] = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
    data['intraday_breakout_gap'] = (data['high'] - data['low']) / data['close'].shift(1)
    data['gap_persistence'] = (data['close'] - data['open']) / (data['high'] - data['low'])
    data['gap_persistence'] = data['gap_persistence'].replace([np.inf, -np.inf], 0).fillna(0)
    
    # Volatility-Adjusted Gap Analysis
    data['true_range'] = np.maximum(
        data['high'] - data['low'],
        np.maximum(
            abs(data['high'] - data['close'].shift(1)),
            abs(data['low'] - data['close'].shift(1))
        )
    )
    data['avg_true_range_5d'] = data['true_range'].rolling(window=5, min_periods=1).mean()
    data['volatility_adjusted_gap'] = data['overnight_gap'] / data['avg_true_range_5d']
    data['volatility_adjusted_gap'] = data['volatility_adjusted_gap'].replace([np.inf, -np.inf], 0).fillna(0)
    
    # Gap Confirmation Signals
    data['volume_support'] = data['volume'] / data['volume'].shift(1)
    data['intraday_momentum_validation'] = (data['close'] - data['open']) / (data['high'] - data['low'])
    data['intraday_momentum_validation'] = data['intraday_momentum_validation'].replace([np.inf, -np.inf], 0).fillna(0)
    
    # Volume-Price Efficiency Integration
    # Volume Intensity Assessment
    data['volume_intensity'] = data['volume'] / data['volume'].shift(1)
    data['volume_concentration'] = data['volume'] / (
        data['volume'].shift(4) + data['volume'].shift(3) + 
        data['volume'].shift(2) + data['volume'].shift(1) + data['volume']
    )
    
    # Efficiency-Adjusted Volume Analysis
    data['momentum_adjusted_volume'] = data['volume'] * (data['short_term_momentum'] + data['medium_term_momentum'])
    data['efficiency_weighted_volume'] = data['momentum_adjusted_volume'] * data['efficiency_ratio']
    
    # Volume-Price Divergence Detection
    data['price_roc_5d'] = (data['close'] - data['close'].shift(5)) / data['close'].shift(5)
    data['volume_roc_5d'] = (data['volume'] - data['volume'].shift(5)) / data['volume'].shift(5)
    
    # Cross-Timeframe Consistency Assessment
    # Directional Alignment Analysis
    data['momentum_direction_alignment'] = (
        (data['short_term_momentum'] > 1).astype(int) == (data['medium_term_momentum'] > 1).astype(int)
    ).astype(int)
    
    data['gap_momentum_alignment'] = (
        (data['overnight_gap'] > 0).astype(int) == (data['short_term_momentum'] > 1).astype(int)
    ).astype(int)
    
    data['efficiency_momentum_alignment'] = (
        (data['efficiency_ratio'] > 1).astype(int) == (data['short_term_momentum'] > 1).astype(int)
    ).astype(int)
    
    # Persistence Scoring
    data['momentum_div_persistence'] = (
        (data['momentum_divergence'] > data['momentum_divergence'].shift(1)).rolling(window=3, min_periods=1).mean()
    )
    
    # Signal Coherence Evaluation
    data['consistency_score'] = (
        data['momentum_direction_alignment'] + 
        data['gap_momentum_alignment'] + 
        data['efficiency_momentum_alignment']
    ) / 3
    
    # Composite Factor Generation
    # Core Signal Combination
    core_signal = (
        data['momentum_divergence'] * 
        data['efficiency_ratio'] * 
        data['volatility_adjusted_gap'] * 
        data['efficiency_weighted_volume']
    )
    
    # Confirmation Weighting Application
    # Volume-Price Divergence Weighting
    volume_price_weight = np.ones(len(data))
    for i in range(len(data)):
        if data['price_roc_5d'].iloc[i] > 0 and data['volume_roc_5d'].iloc[i] > 0:
            volume_price_weight[i] = 1.5  # Strong Confirmation
        elif data['price_roc_5d'].iloc[i] > 0 and data['volume_roc_5d'].iloc[i] <= 0:
            volume_price_weight[i] = 0.8  # Weak Confirmation
        elif data['price_roc_5d'].iloc[i] <= 0 and data['volume_roc_5d'].iloc[i] > 0:
            volume_price_weight[i] = 0.5  # No Confirmation
    
    # Cross-Timeframe Consistency Weighting
    consistency_weight = np.where(
        data['consistency_score'] > 0.8, 1.3,
        np.where(data['consistency_score'] > 0.5, 1.0, 0.7)
    )
    
    # Final Predictive Signal Generation
    final_factor = (
        core_signal * 
        volume_price_weight * 
        consistency_weight * 
        data['short_term_accel'] * 
        data['gap_persistence']
    )
    
    # Clean and return the factor
    final_factor = final_factor.replace([np.inf, -np.inf], 0).fillna(0)
    return pd.Series(final_factor, index=data.index)
