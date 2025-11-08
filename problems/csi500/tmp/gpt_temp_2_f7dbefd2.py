import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Price-Volume Divergence Component
    # True Range calculation
    data['prev_close'] = data['close'].shift(1)
    data['tr1'] = data['high'] - data['low']
    data['tr2'] = np.abs(data['high'] - data['prev_close'])
    data['tr3'] = np.abs(data['low'] - data['prev_close'])
    data['true_range'] = data[['tr1', 'tr2', 'tr3']].max(axis=1)
    
    # Directional Movement Ratio
    data['price_change'] = data['close'] - data['prev_close']
    data['direction_strength'] = data['price_change'] / (data['true_range'] + 1e-8)
    
    # Volume-Price Alignment
    data['volume_weighted_change'] = data['price_change'] * data['volume']
    
    # Volume-Price Correlation (5-day rolling, excluding current day)
    volume_changes = data['volume'].pct_change()
    price_changes = data['close'].pct_change()
    
    # Create shifted series for correlation calculation
    vol_shifted = volume_changes.shift(1)
    price_shifted = price_changes.shift(1)
    
    # Calculate rolling correlation using only past data
    data['volume_price_corr'] = vol_shifted.rolling(window=5, min_periods=3).corr(price_shifted)
    
    # Compute Divergence Signal
    data['divergence_raw'] = data['direction_strength'] * data['volume_price_corr']
    data['divergence_signal'] = np.tanh(data['divergence_raw'])
    
    # Liquidity-Adjusted Momentum Component
    # Amihud Illiquidity Ratio
    data['amihud_ratio'] = np.abs(data['price_change']) / (data['volume'] + 1e-8)
    
    # Liquidity-Adjusted Return
    data['liquidity_adjusted_return'] = data['price_change'] / (data['amihud_ratio'] + 1e-8)
    
    # Momentum Persistence
    # Calculate consecutive direction days
    returns_sign = np.sign(data['price_change'])
    persistence_count = np.zeros(len(data))
    
    for i in range(1, len(data)):
        if returns_sign.iloc[i] == returns_sign.iloc[i-1]:
            persistence_count[i] = persistence_count[i-1] + 1
        else:
            persistence_count[i] = 1
    
    data['persistence'] = persistence_count
    
    # Momentum Decay Factor (inverse relationship with persistence)
    data['decay_factor'] = 1.0 / (1.0 + data['persistence'].clip(upper=10))
    
    # Compute Adjusted Momentum
    data['effective_momentum'] = data['liquidity_adjusted_return'] * data['decay_factor']
    data['adjusted_momentum'] = np.sign(data['effective_momentum']) * np.abs(data['effective_momentum']) ** (1/3)
    
    # Market Microstructure Integration
    # High-Low Efficiency Ratio
    data['hl_range'] = data['high'] - data['low']
    data['efficiency_ratio'] = np.abs(data['price_change']) / (data['hl_range'] + 1e-8)
    
    # Spread Impact Adjustment
    data['spread_impact'] = np.log(1.0 / (data['efficiency_ratio'] + 1e-8))
    
    # Volume Cluster Detection
    # Volume spikes relative to 20-day median
    volume_median = data['volume'].rolling(window=20, min_periods=10).median()
    data['volume_ratio'] = data['volume'] / (volume_median + 1e-8)
    
    # Geometric mean threshold
    threshold = np.exp(data['volume_ratio'].rolling(window=20, min_periods=10).apply(lambda x: np.mean(np.log(x + 1e-8))))
    data['volume_spike'] = (data['volume_ratio'] > threshold * 1.5).astype(int)
    
    # Cluster Persistence with Fibonacci weighting
    cluster_persistence = np.zeros(len(data))
    fib_weights = [1, 1, 2, 3, 5, 8]  # Fibonacci sequence
    
    for i in range(len(data)):
        if i < 5:
            cluster_persistence[i] = data['volume_spike'].iloc[max(0, i-5):i+1].sum()
        else:
            window = data['volume_spike'].iloc[i-5:i+1].values
            cluster_persistence[i] = np.dot(window, fib_weights[:len(window)])
    
    data['cluster_persistence'] = cluster_persistence
    
    # Dynamic Factor Synthesis
    # Divergence-Momentum Integration
    data['divergence_momentum'] = data['divergence_signal'] * data['adjusted_momentum']
    
    # Preserve sign while scaling
    data['integrated_factor'] = np.sign(data['divergence_momentum']) * np.abs(data['divergence_momentum']) ** 0.5
    
    # Microstructure Adjustment
    data['micro_adjusted'] = data['integrated_factor'] * data['spread_impact']
    
    # Volume Cluster as regime filter
    cluster_weight = 1.0 / (1.0 + data['cluster_persistence'] * 0.1)
    data['cluster_filtered'] = data['micro_adjusted'] * cluster_weight
    
    # Signal Refinement with asymmetric smoothing
    positive_signal = data['cluster_filtered'].where(data['cluster_filtered'] > 0, 0)
    negative_signal = data['cluster_filtered'].where(data['cluster_filtered'] < 0, 0)
    
    # Different smoothing parameters for positive and negative signals
    positive_smoothed = positive_signal.rolling(window=5, min_periods=3).mean()
    negative_smoothed = negative_signal.rolling(window=3, min_periods=2).mean()
    
    # Combine with persistence weighting
    persistence_weight = 1.0 / (1.0 + data['persistence'] * 0.2)
    data['final_signal'] = (positive_smoothed + negative_smoothed) * persistence_weight
    
    # Final bounded output
    alpha_factor = np.tanh(data['final_signal'])
    
    # Clean up and return
    result = pd.Series(alpha_factor, index=data.index, name='alpha_factor')
    return result.dropna()
