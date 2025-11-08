import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Gap Momentum Analysis
    # Calculate overnight gaps
    data['prev_close'] = data['close'].shift(1)
    data['gap_size'] = (data['open'] / data['prev_close']) - 1
    data['gap_abs'] = np.abs(data['gap_size'])
    
    # Classify gap magnitude
    conditions = [
        data['gap_abs'] < 0.008,
        (data['gap_abs'] >= 0.008) & (data['gap_abs'] <= 0.02),
        data['gap_abs'] > 0.02
    ]
    choices = [1, 2, 3]  # Small, Medium, Large weights
    data['gap_magnitude_weight'] = np.select(conditions, choices, default=1)
    
    # Gap fill rate
    data['gap_fill_rate'] = (data['close'] - data['open']) / (data['gap_abs'] * data['prev_close'] + 1e-8)
    
    # Momentum persistence (count of same-direction closes)
    data['price_change'] = data['close'] - data['close'].shift(1)
    data['direction'] = np.sign(data['price_change'])
    data['momentum_persistence'] = (
        (data['direction'] == data['direction'].shift(1)) & 
        (data['direction'] != 0)
    ).rolling(window=3, min_periods=1).sum()
    
    # Momentum quality (std dev of intraday returns)
    data['intraday_return'] = (data['close'] - data['open']) / data['open']
    data['momentum_quality'] = data['intraday_return'].rolling(window=5, min_periods=3).std()
    
    # Volatility Adjustment
    # True Range calculation
    data['tr_high'] = np.maximum(data['high'], data['prev_close'])
    data['tr_low'] = np.minimum(data['low'], data['prev_close'])
    data['true_range'] = data['tr_high'] - data['tr_low']
    
    # Volatility-scaled momentum
    data['volatility_scaled_momentum'] = data['gap_fill_rate'] / (data['true_range'] / data['prev_close'] + 1e-8)
    
    # Liquidity Quality Assessment
    # Volume-to-Range Efficiency
    data['price_range'] = data['high'] - data['low']
    data['vre'] = data['volume'] / (data['price_range'] + 1e-8)
    data['vre_5d_avg'] = data['vre'].rolling(window=5, min_periods=3).mean()
    data['vre_ratio'] = data['vre'] / (data['vre_5d_avg'] + 1e-8)
    
    # Order Flow Analysis
    data['flow_imbalance'] = (data['close'] - data['open']) / (data['price_range'] + 1e-8)
    data['log_volume'] = np.log(data['volume'] + 1)
    data['flow_signal'] = data['flow_imbalance'] * data['log_volume']
    
    # Flow persistence (3-day trend)
    data['flow_trend'] = data['flow_signal'].rolling(window=3, min_periods=2).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) >= 2 else 0
    )
    
    # Signal Integration
    # Combine components
    data['liquidity_quality'] = data['vre_ratio'] * (1 + np.abs(data['flow_trend']))
    
    # Main factor calculation
    data['raw_factor'] = (
        data['volatility_scaled_momentum'] * 
        data['liquidity_quality'] * 
        data['gap_magnitude_weight'] * 
        (1 + 0.1 * data['momentum_persistence'])
    )
    
    # Apply direction based on gap fill completion
    data['gap_direction'] = np.sign(data['gap_size'])
    data['direction_adjusted'] = data['raw_factor'] * data['gap_direction']
    
    # Scale by volume acceleration
    data['volume_5d_avg'] = data['volume'].rolling(window=5, min_periods=3).mean()
    data['volume_acceleration'] = data['volume'] / (data['volume_5d_avg'] + 1e-8)
    data['volume_scaled'] = data['direction_adjusted'] * np.sqrt(data['volume_acceleration'])
    
    # Filter low liquidity signals
    liquidity_threshold = data['volume_5d_avg'].quantile(0.2)
    data['liquidity_filter'] = data['volume'] > liquidity_threshold
    
    # Final alpha factor
    data['alpha_factor'] = data['volume_scaled'] * data['liquidity_filter']
    
    # Clean up and return
    result = data['alpha_factor'].replace([np.inf, -np.inf], np.nan).fillna(0)
    return result
