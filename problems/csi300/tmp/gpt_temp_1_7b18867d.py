import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Momentum-Fractal Cluster Entropy Factor
    Combines momentum divergence with range-fractal cluster entropy patterns
    """
    data = df.copy()
    
    # Momentum-Fractal Divergence
    data['momentum_short'] = data['close'].pct_change(5)
    data['momentum_medium'] = data['close'].pct_change(20)
    data['momentum_divergence'] = data['momentum_short'] - data['momentum_medium']
    
    # Range-Fractal Cluster Entropy components
    data['daily_range'] = (data['high'] - data['low']) / data['close']
    data['range_efficiency'] = data['daily_range'] / (abs(data['close'].pct_change()) + 1e-8)
    
    # Volume-Close patterns for cluster detection
    data['volume_close_correlation'] = data['volume'].rolling(10).corr(data['close'])
    data['volume_flow'] = data['volume'] * data['close'].pct_change()
    
    # Cluster-Transition Entropy calculation
    data['price_cluster'] = pd.cut(data['close'], bins=5, labels=False)
    cluster_transitions = data['price_cluster'].diff().abs()
    data['cluster_entropy'] = cluster_transitions.rolling(15).apply(
        lambda x: -np.sum(np.unique(x, return_counts=True)[1] / len(x) * 
                         np.log(np.unique(x, return_counts=True)[1] / len(x) + 1e-8))
    )
    
    # Duration-Efficiency Patterns
    data['momentum_efficiency'] = data['momentum_short'] / (data['daily_range'].rolling(5).std() + 1e-8)
    
    # Momentum runs detection
    momentum_sign = np.sign(data['momentum_short'])
    momentum_changes = (momentum_sign != momentum_sign.shift(1)).astype(int)
    data['momentum_run_length'] = momentum_changes.groupby(
        momentum_changes.cumsum()
    ).cumcount() + 1
    
    # Asymmetric Duration Signals
    up_moves = data['close'] > data['close'].shift(1)
    down_moves = data['close'] < data['close'].shift(1)
    
    data['up_duration'] = up_moves.groupby((~up_moves).cumsum()).cumcount()
    data['down_duration'] = down_moves.groupby((~down_moves).cumsum()).cumcount()
    data['duration_asymmetry'] = data['up_duration'] - data['down_duration']
    
    # Signal Generation
    # Momentum-Cluster Alignment
    data['momentum_cluster_alignment'] = (
        data['momentum_divergence'] * data['cluster_entropy'].shift(1)
    )
    
    # Efficiency-Entropy Convergence
    data['efficiency_entropy'] = (
        data['momentum_efficiency'] * data['cluster_entropy'].shift(1)
    )
    
    # Final Alpha Signals - weighted combination
    alpha_signal = (
        0.4 * data['momentum_cluster_alignment'] +
        0.3 * data['efficiency_entropy'] +
        0.2 * data['duration_asymmetry'] +
        0.1 * data['volume_close_correlation']
    )
    
    # Normalize the final signal
    alpha_signal = (alpha_signal - alpha_signal.rolling(50).mean()) / alpha_signal.rolling(50).std()
    
    return alpha_signal
