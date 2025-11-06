import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Price-Volume-Momentum Synthesis with Intraday Confirmation factor
    """
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Momentum Component Synthesis
    # Multi-Timeframe Momentum Alignment
    data['momentum_5d'] = data['close'] - data['close'].shift(5)
    data['momentum_10d'] = data['close'] - data['close'].shift(10)
    
    # Momentum consistency ratio (alignment between short and medium term momentum)
    data['momentum_consistency'] = np.where(
        (data['momentum_5d'] * data['momentum_10d']) > 0,
        np.abs(data['momentum_5d'] / (data['momentum_10d'] + 1e-8)),
        0
    )
    
    # High-Low Range Momentum
    data['high_low_range_5d'] = (data['high'] - data['low']).rolling(window=5).mean()
    data['high_low_range_momentum'] = (data['high_low_range_5d'] - data['high_low_range_5d'].shift(5)) / (data['high_low_range_5d'].shift(5) + 1e-8)
    
    # Volume-Amount Integration
    # Volume Divergence Analysis
    data['volume_5d_avg'] = data['volume'].rolling(window=5).mean()
    data['volume_20d_avg'] = data['volume'].rolling(window=20).mean()
    data['volume_divergence'] = (data['volume_5d_avg'] - data['volume_20d_avg']) / (data['volume_20d_avg'] + 1e-8)
    
    # Amount Flow Efficiency
    data['amount_volume_ratio'] = data['amount'] / (data['volume'] + 1e-8)
    data['amount_flow_trend'] = data['amount_volume_ratio'].rolling(window=5).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 5 else np.nan
    )
    
    # Smart money flow pattern detection
    data['amount_consistency'] = data['amount'].rolling(window=5).std() / (data['amount'].rolling(window=5).mean() + 1e-8)
    data['smart_money_flow'] = np.where(
        (data['amount_flow_trend'] > 0) & (data['amount_consistency'] < 0.5),
        data['amount_flow_trend'],
        0
    )
    
    # Intraday Confirmation Layer
    # Open-to-Close Pattern Analysis
    data['intraday_return'] = (data['close'] - data['open']) / (data['open'] + 1e-8)
    data['intraday_persistence'] = data['intraday_return'].rolling(window=3).apply(
        lambda x: 1 if all(x > 0) or all(x < 0) else 0 if len(x) == 3 else np.nan
    )
    
    # Overnight Gap Validation
    data['overnight_gap'] = (data['open'] - data['close'].shift(1)) / (data['close'].shift(1) + 1e-8)
    data['gap_momentum_alignment'] = np.where(
        data['overnight_gap'] * data['momentum_5d'] > 0,
        np.abs(data['overnight_gap']),
        0
    )
    
    # Signal Synthesis and Weighting
    # Momentum-Volume Alignment
    data['momentum_volume_alignment'] = data['momentum_consistency'] * data['volume_divergence']
    data['momentum_volume_alignment'] = data['momentum_volume_alignment'] * np.sign(data['momentum_5d'])
    
    # Amount-Based Confidence Weighting
    data['amount_weighted_signal'] = data['momentum_volume_alignment'] * data['smart_money_flow']
    data['amount_filtered_signal'] = np.where(
        data['amount_consistency'] < 1.0,  # Filter high volatility periods
        data['amount_weighted_signal'],
        0
    )
    
    # Intraday-Overnight Convergence
    data['intraday_overnight_convergence'] = data['amount_filtered_signal'] * data['intraday_persistence']
    data['final_factor'] = data['intraday_overnight_convergence'] * (1 + data['gap_momentum_alignment'])
    
    # Apply rolling normalization to make factor more stable
    data['factor_normalized'] = (data['final_factor'] - data['final_factor'].rolling(window=20).mean()) / (data['final_factor'].rolling(window=20).std() + 1e-8)
    
    return data['factor_normalized']
