import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Fractal Entropy Dynamics
    # Multi-Scale Price Entropy
    def calculate_entropy(series, window):
        entropy_values = []
        for i in range(len(series)):
            if i < window:
                entropy_values.append(np.nan)
                continue
                
            window_data = series.iloc[i-window:i+1]
            price_changes = window_data.diff().abs().dropna()
            
            if price_changes.sum() == 0:
                entropy_values.append(0)
                continue
                
            probabilities = price_changes / price_changes.sum()
            entropy = -np.sum(probabilities * np.log(probabilities + 1e-10))
            entropy_values.append(entropy)
        
        return pd.Series(entropy_values, index=series.index)
    
    price_entropy_5 = calculate_entropy(data['close'], 5)
    price_entropy_10 = calculate_entropy(data['close'], 10)
    
    # Volume Entropy Components
    def calculate_volume_entropy(volume_series, window):
        entropy_values = []
        for i in range(len(volume_series)):
            if i < window:
                entropy_values.append(np.nan)
                continue
                
            window_data = volume_series.iloc[i-window:i+1]
            probabilities = window_data / window_data.sum()
            entropy = -np.sum(probabilities * np.log(probabilities + 1e-10))
            entropy_values.append(entropy)
        
        return pd.Series(entropy_values, index=volume_series.index)
    
    volume_entropy_5 = calculate_volume_entropy(data['volume'], 5)
    volume_entropy_10 = calculate_volume_entropy(data['volume'], 10)
    
    # Entropy Divergence Quality
    entropy_divergence = (price_entropy_10 - price_entropy_5) * (volume_entropy_10 - volume_entropy_5)
    volume_entropy_diff = volume_entropy_10 - volume_entropy_5
    entropy_quality = np.tanh(entropy_divergence * volume_entropy_diff)
    
    # Anchored Momentum Patterns
    # Multi-Scale Momentum
    momentum_3d = (data['close'] / data['close'].shift(2) - 1) * (data['close'].shift(1) / data['close'].shift(3) - 1)
    momentum_5d = (data['close'] / data['close'].shift(4) - 1) * (data['close'].shift(2) / data['close'].shift(6) - 1)
    momentum_8d = (data['close'] / data['close'].shift(7) - 1) * (data['close'].shift(4) / data['close'].shift(11) - 1)
    
    # VWAP Anchoring
    # Calculate cumulative price * volume and cumulative volume
    cumulative_price_volume = (data['close'] * data['volume']).expanding().sum()
    cumulative_volume = data['volume'].expanding().sum()
    vwap_intraday = cumulative_price_volume / cumulative_volume
    
    vwap_proximity = (data['close'] - vwap_intraday) / (data['high'] - data['low'])
    vwap_proximity = vwap_proximity.replace([np.inf, -np.inf], np.nan)
    
    momentum_fractals = pd.concat([momentum_3d, momentum_5d, momentum_8d], axis=1).mean(axis=1)
    anchored_momentum = momentum_fractals * (1 + vwap_proximity)
    
    # Momentum Consistency
    momentum_signs = pd.concat([momentum_3d, momentum_5d, momentum_8d], axis=1).apply(np.sign, axis=1)
    fractal_consistency = momentum_signs.sum(axis=1).abs()
    
    # Volume-Price Fractal Alignment
    # Velocity Components
    price_velocity = (data['close'] - data['close'].shift(2)) / (data['close'].shift(2) - data['close'].shift(4))
    price_velocity = price_velocity.replace([np.inf, -np.inf], np.nan)
    
    volume_rolling_mean = data['volume'].rolling(window=4, min_periods=1).mean().shift(1)
    turnover_velocity = data['volume'] / volume_rolling_mean
    turnover_velocity = turnover_velocity.replace([np.inf, -np.inf], np.nan)
    
    intraday_pressure = (data['high'] - data['open']) - (data['open'] - data['low'])
    
    # Multi-Scale Alignment
    short_term_alignment = np.sign(data['close'] - data['close'].shift(1)) * np.sign(data['volume'] - data['volume'].shift(1))
    medium_term_volume = data['volume'].rolling(window=5, min_periods=1).mean().shift(1)
    medium_term_alignment = np.sign(data['close'] - data['close'].shift(5)) * np.sign(data['volume'] - medium_term_volume)
    
    alignment_score = (short_term_alignment + medium_term_alignment) * np.minimum(np.abs(price_velocity), np.abs(turnover_velocity))
    
    # Velocity Confluence
    velocity_confluence = alignment_score * intraday_pressure
    
    # Microstructure Quality Filters
    # Liquidity Components
    liquidity_efficiency = np.abs(data['close'] - data['close'].shift(1)) / data['volume']
    liquidity_efficiency = liquidity_efficiency.replace([np.inf, -np.inf], np.nan)
    
    numerator = np.abs((data['close'] - data['open']) * data['volume'])
    denominator = np.abs((data['open'] - data['close'].shift(1)) * data['volume']) + numerator
    flow_asymmetry = numerator / denominator
    flow_asymmetry = flow_asymmetry.replace([np.inf, -np.inf], np.nan)
    
    # Noise Adjustment
    upper_shadow = data['high'] - np.maximum(data['open'], data['close'])
    lower_shadow = np.minimum(data['open'], data['close']) - data['low']
    net_shadow_pressure = (upper_shadow - lower_shadow) / (data['high'] - data['low'])
    net_shadow_pressure = net_shadow_pressure.replace([np.inf, -np.inf], np.nan)
    
    gap_noise = (np.abs(data['open'] - data['close'].shift(1)) / data['close'].shift(1)) * (1 - np.abs(data['close'] - data['open']) / (data['high'] - data['low']))
    gap_noise = gap_noise.replace([np.inf, -np.inf], np.nan)
    
    # Quality Score
    quality_factor = liquidity_efficiency * flow_asymmetry / (1 + np.abs(net_shadow_pressure) + gap_noise)
    
    # Adaptive Signal Synthesis
    # Component Integration
    integrated_score = entropy_quality * anchored_momentum * velocity_confluence * quality_factor
    
    # Signal Validation
    fractal_consistency_filter = fractal_consistency >= 2
    velocity_alignment_filter = (np.sign(price_velocity) == np.sign(turnover_velocity)) & (~price_velocity.isna()) & (~turnover_velocity.isna())
    quality_threshold = quality_factor > 0
    
    # Apply filters
    final_signal = integrated_score.copy()
    mask = fractal_consistency_filter & velocity_alignment_filter & quality_threshold
    final_signal[~mask] = 0
    
    # Final Signal with cubic root transformation
    final_signal = np.sign(final_signal) * np.abs(final_signal) ** (1/3)
    
    return final_signal
