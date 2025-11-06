import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Generate alpha factor combining price-volume divergence, range efficiency, 
    extreme reversal detection, amount flow analysis, and volatility-adaptive volume patterns.
    """
    # Price-Volume Divergence Momentum
    df['momentum_5d'] = df['close'] / df['close'].shift(5) - 1
    df['momentum_10d'] = df['close'] / df['close'].shift(10) - 1
    df['momentum_ratio'] = (df['close'] / df['close'].shift(5)) / (df['close'].shift(5) / df['close'].shift(10))
    
    df['volume_trend'] = df['volume'] / df['volume'].shift(5)
    df['volume_momentum'] = (df['volume'] / df['volume'].shift(5)) / (df['volume'].shift(5) / df['volume'].shift(10))
    
    # Volume persistence (count of days with volume > previous day's volume over last 5 days)
    volume_persistence = []
    for i in range(len(df)):
        if i < 5:
            volume_persistence.append(np.nan)
        else:
            window = df['volume'].iloc[i-5:i]
            count = (window > window.shift(1)).sum()
            volume_persistence.append(count)
    df['volume_persistence'] = volume_persistence
    
    # Divergence signal
    df['divergence_signal'] = df['momentum_5d'] * (1 / df['volume_momentum'])
    
    # High-Low Range Efficiency
    df['daily_efficiency'] = abs(df['close'] - df['close'].shift(1)) / (df['high'] - df['low'])
    
    # 3-day efficiency
    def efficiency_3d(row_idx):
        if row_idx < 3:
            return np.nan
        price_change = abs(df['close'].iloc[row_idx] - df['close'].iloc[row_idx-3])
        range_sum = sum(df['high'].iloc[row_idx-2:row_idx+1] - df['low'].iloc[row_idx-2:row_idx+1])
        return price_change / range_sum if range_sum > 0 else np.nan
    
    df['efficiency_3d'] = [efficiency_3d(i) for i in range(len(df))]
    df['efficiency_trend'] = df['daily_efficiency'] / df['daily_efficiency'].shift(1)
    
    df['range_expansion'] = (df['high'] - df['low']) / (df['high'].shift(1) - df['low'].shift(1))
    
    # Range consistency (coefficient of variation over 5 days)
    range_consistency = []
    for i in range(len(df)):
        if i < 5:
            range_consistency.append(np.nan)
        else:
            window = df['high'].iloc[i-4:i+1] - df['low'].iloc[i-4:i+1]
            range_consistency.append(window.std() / window.mean() if window.mean() > 0 else np.nan)
    df['range_consistency'] = range_consistency
    
    # Volume-Scaled Extreme Reversal
    df['price_extremity'] = (df['close'] - df['close'].shift(1)) / (df['high'] - df['low'])
    
    def deviation_3d(row_idx):
        if row_idx < 3:
            return np.nan
        window_close = df['close'].iloc[row_idx-2:row_idx+1]
        window_high = df['high'].iloc[row_idx-2:row_idx+1]
        window_low = df['low'].iloc[row_idx-2:row_idx+1]
        return (df['close'].iloc[row_idx] - window_close.mean()) / (window_high.max() - window_low.min())
    
    df['deviation_3d'] = [deviation_3d(i) for i in range(len(df))]
    
    # Volume extremity
    volume_extremity = []
    for i in range(len(df)):
        if i < 5:
            volume_extremity.append(np.nan)
        else:
            window = df['volume'].iloc[i-4:i+1]
            volume_extremity.append(df['volume'].iloc[i] / window.mean())
    df['volume_extremity'] = volume_extremity
    
    # Amount Flow Regime Detection
    df['net_flow'] = df['amount'] * np.sign(df['close'] - df['close'].shift(1))
    
    # Flow momentum (3-day)
    flow_momentum = []
    for i in range(len(df)):
        if i < 3:
            flow_momentum.append(np.nan)
        else:
            net_flow_sum = df['net_flow'].iloc[i-2:i+1].sum()
            amount_sum = df['amount'].iloc[i-2:i+1].sum()
            flow_momentum.append(net_flow_sum / amount_sum if amount_sum > 0 else np.nan)
    df['flow_momentum'] = flow_momentum
    
    # Flow consistency (count of same direction flow over 5 days)
    flow_consistency = []
    for i in range(len(df)):
        if i < 5:
            flow_consistency.append(np.nan)
        else:
            window = df['net_flow'].iloc[i-4:i+1]
            same_dir_count = (np.sign(window) == np.sign(window.iloc[-1])).sum()
            flow_consistency.append(same_dir_count)
    df['flow_consistency'] = flow_consistency
    
    # Volatility-Adaptive Volume Clustering
    # Relative volatility
    rel_volatility = []
    for i in range(len(df)):
        if i < 10:
            rel_volatility.append(np.nan)
        else:
            recent_vol = df['close'].iloc[i-4:i+1].std()
            previous_vol = df['close'].iloc[i-9:i-4].std()
            rel_volatility.append(recent_vol / previous_vol if previous_vol > 0 else np.nan)
    df['rel_volatility'] = rel_volatility
    
    # Range volatility
    range_volatility = []
    for i in range(len(df)):
        if i < 5:
            range_volatility.append(np.nan)
        else:
            window_high = df['high'].iloc[i-4:i+1]
            window_low = df['low'].iloc[i-4:i+1]
            range_volatility.append((window_high.max() - window_low.min()) / df['close'].iloc[i-5])
    df['range_volatility'] = range_volatility
    
    # Volume spike clustering
    volume_spike_clustering = []
    for i in range(len(df)):
        if i < 5:
            volume_spike_clustering.append(np.nan)
        else:
            window = df['volume'].iloc[i-4:i+1]
            median_vol = window.median()
            spike_count = (window > 2 * median_vol).sum()
            volume_spike_clustering.append(spike_count)
    df['volume_spike_clustering'] = volume_spike_clustering
    
    # Combine all signals into final alpha factor
    # Normalize each component and combine with weights
    components = [
        'divergence_signal', 'daily_efficiency', 'efficiency_3d', 
        'range_expansion', 'price_extremity', 'deviation_3d',
        'flow_momentum', 'flow_consistency', 'rel_volatility',
        'volume_spike_clustering'
    ]
    
    # Z-score normalization for each component
    normalized_components = {}
    for comp in components:
        if comp in df.columns:
            mean_val = df[comp].mean()
            std_val = df[comp].std()
            normalized_components[comp] = (df[comp] - mean_val) / std_val
    
    # Final alpha factor combining all signals
    alpha = (
        0.15 * normalized_components.get('divergence_signal', 0) +
        0.12 * normalized_components.get('daily_efficiency', 0) +
        0.10 * normalized_components.get('efficiency_3d', 0) +
        0.08 * normalized_components.get('range_expansion', 0) +
        0.12 * normalized_components.get('price_extremity', 0) +
        0.10 * normalized_components.get('deviation_3d', 0) +
        0.13 * normalized_components.get('flow_momentum', 0) +
        0.08 * normalized_components.get('flow_consistency', 0) +
        0.07 * normalized_components.get('rel_volatility', 0) +
        0.05 * normalized_components.get('volume_spike_clustering', 0)
    )
    
    return alpha
