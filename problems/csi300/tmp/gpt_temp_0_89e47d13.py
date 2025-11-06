import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Fractal Momentum-Pressure Entropy Factor
    Combines momentum divergence, pressure dynamics, range-volume entropy, 
    and volatility-adaptive confirmation for alpha generation.
    """
    data = df.copy()
    
    # Initialize result series
    alpha = pd.Series(index=data.index, dtype=float)
    
    # Ensure sufficient data length
    if len(data) < 20:
        return alpha
    
    # 1. Calculate Fractal Momentum Divergence with Pressure
    # Multi-Timeframe Momentum Divergence
    data['momentum_short'] = data['close'] / data['close'].shift(5) - 1
    data['momentum_medium'] = data['close'] / data['close'].shift(20) - 1
    data['momentum_divergence'] = data['momentum_short'] - data['momentum_medium']
    
    # Intraday Pressure Components
    data['opening_pressure'] = (data['open'] - data['close'].shift(1)) * data['volume']
    data['closing_pressure'] = (data['close'] - data['open']) * data['volume']
    data['net_pressure'] = data['closing_pressure'] - data['opening_pressure']
    
    # Pressure Consistency Count
    pressure_signs = data['net_pressure'].rolling(window=5).apply(
        lambda x: len(set(np.sign(x.dropna()))) if len(x.dropna()) == 5 else np.nan
    )
    data['pressure_consistency'] = 5 - (pressure_signs - 1)  # Higher for more consistent signs
    
    # Momentum-Pressure Interaction
    data['momentum_pressure'] = data['momentum_divergence'] * data['net_pressure'] * data['pressure_consistency']
    
    # 2. Calculate Range-Volume Cluster Entropy
    # Range Efficiency with Fractal Characteristics
    data['current_range'] = data['high'] - data['low']
    data['historical_range'] = data['current_range'].rolling(window=10).mean()
    data['range_efficiency'] = abs(data['close'] - data['open']) / (data['current_range'] + 1e-8)
    
    # Volume-Price Entropy Clusters
    data['volume_concentration'] = data['volume'] / data['volume'].rolling(window=5).sum()
    
    # Price-Volume Oscillation
    data['price_volume_osc'] = abs(data['close'] - data['close'].shift(1)) * np.sqrt(data['volume'])
    
    # Cluster Entropy Measurement
    def calculate_cluster_entropy(series):
        if len(series) < 5:
            return np.nan
        # Simple entropy proxy based on price-volume pattern changes
        changes = series.diff().dropna()
        if len(changes) < 4:
            return np.nan
        # Count significant pattern changes
        threshold = changes.abs().mean()
        pattern_changes = (changes.abs() > threshold).sum()
        return pattern_changes / len(changes)
    
    data['cluster_entropy'] = data['price_volume_osc'].rolling(window=5).apply(
        calculate_cluster_entropy, raw=False
    )
    
    # Range-Volume Entropy Signal
    data['efficiency_entropy'] = data['range_efficiency'] * data['cluster_entropy'] * data['volume_concentration']
    
    # Volume breakout detection
    volume_avg = data['volume'].rolling(window=10).mean()
    volume_breakout = data['volume'] > (1.2 * volume_avg)
    data['range_volume_entropy'] = data['efficiency_entropy'] * np.where(volume_breakout, 1.5, 1.0)
    
    # 3. Calculate Volatility-Adaptive Confirmation
    # Volatility Environment
    data['recent_volatility'] = data['close'].rolling(window=5).std()
    data['baseline_volatility'] = data['close'].rolling(window=20).std()
    data['volatility_ratio'] = data['recent_volatility'] / (data['baseline_volatility'] + 1e-8)
    
    # Volatility-Adjusted Returns and Momentum Persistence
    data['vol_adj_returns'] = (data['close'] - data['close'].shift(1)) / (data['recent_volatility'] + 1e-8)
    
    def count_positive_returns(series):
        return (series > 0).sum() if len(series) == 5 else np.nan
    
    data['momentum_persistence'] = data['vol_adj_returns'].rolling(window=5).apply(
        count_positive_returns, raw=False
    )
    
    # Volatility-Weighted Momentum
    data['vol_weighted_momentum'] = data['momentum_persistence'] * data['volatility_ratio']
    
    # Range Expansion Confirmation
    data['range_expansion'] = data['current_range'] / (data['historical_range'] + 1e-8)
    data['momentum_acceleration'] = (data['close'] / data['close'].shift(1)) / (data['close'].shift(1) / data['close'].shift(2) + 1e-8)
    data['expansion_momentum'] = data['range_expansion'] * data['momentum_acceleration']
    
    # Adaptive Confirmation
    data['volatility_confirmation'] = data['vol_weighted_momentum'] * data['expansion_momentum']
    
    # 4. Detect Asymmetric Efficiency Patterns
    # Combined efficiency score
    data['momentum_pressure_aligned'] = np.sign(data['momentum_pressure']) * data['range_volume_entropy']
    data['efficiency_alignment'] = data['momentum_pressure_aligned'] * data['volatility_confirmation']
    
    # Duration-efficiency relationships
    data['short_momentum_run'] = data['momentum_short'].rolling(window=3).mean()
    data['pressure_accumulation'] = data['net_pressure'].rolling(window=5).sum()
    
    # Convergence signals
    def detect_convergence(row):
        if pd.isna(row['momentum_pressure']) or pd.isna(row['range_volume_entropy']) or pd.isna(row['volatility_confirmation']):
            return np.nan
        # Early alignment: all components positive and increasing
        components = [row['momentum_pressure'], row['range_volume_entropy'], row['volatility_confirmation']]
        positive_count = sum(1 for x in components if x > 0)
        return positive_count / 3.0
    
    # Apply convergence detection
    convergence_scores = []
    for i in range(len(data)):
        if i < 5:
            convergence_scores.append(np.nan)
            continue
        row = data.iloc[i]
        convergence_scores.append(detect_convergence(row))
    
    data['convergence_signal'] = convergence_scores
    
    # 5. Generate Final Alpha Factor
    # Combine core components
    data['core_alpha'] = (data['momentum_pressure'] * data['range_volume_entropy'] * 
                         data['volatility_confirmation'] * data['convergence_signal'])
    
    # Dynamic smoothing based on volatility
    def dynamic_smoothing(series, volatility_series):
        smoothed = pd.Series(index=series.index, dtype=float)
        for i in range(len(series)):
            if i < 10:
                smoothed.iloc[i] = series.iloc[i]
                continue
            current_vol = volatility_series.iloc[i]
            # Shorter window for high volatility, longer for low volatility
            window_size = max(3, min(10, int(10 / (current_vol + 0.1))))
            start_idx = max(0, i - window_size + 1)
            window_data = series.iloc[start_idx:i+1]
            # Pressure-consistency weighted average
            pressure_weights = data['pressure_consistency'].iloc[start_idx:i+1]
            if len(window_data) > 0 and not window_data.isna().all():
                weighted_avg = np.average(window_data.fillna(0), weights=pressure_weights.fillna(1))
                smoothed.iloc[i] = weighted_avg
            else:
                smoothed.iloc[i] = series.iloc[i]
        return smoothed
    
    # Apply dynamic smoothing
    alpha = dynamic_smoothing(data['core_alpha'], data['recent_volatility'])
    
    return alpha
