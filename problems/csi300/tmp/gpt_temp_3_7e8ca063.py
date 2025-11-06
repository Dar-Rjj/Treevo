import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Hierarchical Momentum-Volume Entropy with Structural Break Detection
    """
    data = df.copy()
    
    # Calculate returns for different timeframes
    data['ret_1'] = data['close'].pct_change()
    data['ret_3'] = data['close'].pct_change(3)
    data['ret_5'] = data['close'].pct_change(5)
    
    # Calculate momentum entropy components
    def calculate_entropy(series, window=5):
        """Calculate entropy of a series over rolling window"""
        entropy_values = []
        for i in range(len(series)):
            if i < window:
                entropy_values.append(np.nan)
                continue
            window_data = series.iloc[i-window:i]
            # Normalize to probabilities
            normalized = (window_data - window_data.min()) / (window_data.max() - window_data.min() + 1e-8)
            normalized = normalized[normalized > 0]  # Remove zeros for log
            if len(normalized) > 1:
                entropy = -np.sum(normalized * np.log(normalized))
                entropy_values.append(entropy)
            else:
                entropy_values.append(0)
        return pd.Series(entropy_values, index=series.index)
    
    # Momentum entropy components
    data['momentum_entropy_3'] = calculate_entropy(data['ret_1'], 3)
    data['momentum_entropy_5'] = calculate_entropy(data['ret_1'], 5)
    
    # Volume entropy components
    data['volume_entropy_3'] = calculate_entropy(data['volume'], 3)
    data['volume_entropy_5'] = calculate_entropy(data['volume'], 5)
    
    # Amount entropy components
    data['amount_entropy_3'] = calculate_entropy(data['amount'], 3)
    data['amount_entropy_5'] = calculate_entropy(data['amount'], 5)
    
    # Structural break detection
    def detect_structural_breaks(series, window=5, threshold=2.0):
        """Detect structural breaks using rolling statistics"""
        breaks = []
        for i in range(window, len(series)):
            if pd.isna(series.iloc[i]):
                breaks.append(0)
                continue
            prev_window = series.iloc[i-window:i]
            current_val = series.iloc[i]
            
            if len(prev_window.dropna()) < 3:
                breaks.append(0)
                continue
                
            mean_prev = prev_window.mean()
            std_prev = prev_window.std()
            
            if std_prev == 0:
                breaks.append(0)
            else:
                z_score = abs(current_val - mean_prev) / (std_prev + 1e-8)
                breaks.append(1 if z_score > threshold else 0)
        return pd.Series([0]*window + breaks, index=series.index)
    
    # Detect breaks in momentum and volume
    data['momentum_break_3'] = detect_structural_breaks(data['ret_1'], 3, 2.0)
    data['momentum_break_5'] = detect_structural_breaks(data['ret_1'], 5, 2.0)
    data['volume_break_3'] = detect_structural_breaks(data['volume'], 3, 2.0)
    data['volume_break_5'] = detect_structural_breaks(data['volume'], 5, 2.0)
    
    # Combined break indicator
    data['combined_break'] = (
        data['momentum_break_3'] + data['momentum_break_5'] + 
        data['volume_break_3'] + data['volume_break_5']
    ) / 4.0
    
    # Momentum-Volume Entropy Divergence
    data['momentum_volume_divergence_3'] = (
        data['momentum_entropy_3'] - data['volume_entropy_3']
    )
    data['momentum_volume_divergence_5'] = (
        data['momentum_entropy_5'] - data['volume_entropy_5']
    )
    
    # Amount-enhanced entropy patterns
    data['amount_momentum_coupling_3'] = (
        data['momentum_entropy_3'] * data['amount_entropy_3']
    )
    data['amount_momentum_coupling_5'] = (
        data['momentum_entropy_5'] * data['amount_entropy_5']
    )
    
    # Multi-scale entropy integration
    data['multi_scale_entropy'] = (
        data['momentum_entropy_3'] + data['momentum_entropy_5'] +
        data['volume_entropy_3'] + data['volume_entropy_5'] +
        data['amount_entropy_3'] + data['amount_entropy_5']
    ) / 6.0
    
    # Break-adaptive weighting
    def break_adaptive_weighting(row):
        """Apply break-adaptive weighting based on break conditions"""
        if pd.isna(row['combined_break']):
            return 1.0
        
        break_strength = row['combined_break']
        
        # Pre-break emphasis (low break strength)
        if break_strength < 0.25:
            # Emphasize entropy compression and volume concentration
            weight = 1.5 + (0.25 - break_strength) * 2.0
        # Post-break emphasis (high break strength)
        elif break_strength > 0.75:
            # Emphasize entropy expansion and volume dispersion
            weight = 1.0 + (break_strength - 0.75) * 3.0
        # Transition phase
        else:
            weight = 1.0
            
        return min(weight, 3.0)  # Cap at 3.0
    
    data['break_weight'] = data.apply(break_adaptive_weighting, axis=1)
    
    # Final alpha factor construction
    data['core_entropy_divergence'] = (
        data['momentum_volume_divergence_3'] * 0.4 +
        data['momentum_volume_divergence_5'] * 0.6
    )
    
    data['amount_enhanced_patterns'] = (
        data['amount_momentum_coupling_3'] * 0.3 +
        data['amount_momentum_coupling_5'] * 0.7
    )
    
    # Final composite factor
    data['alpha_factor'] = (
        data['core_entropy_divergence'] * 0.5 +
        data['amount_enhanced_patterns'] * 0.3 +
        data['multi_scale_entropy'] * 0.2
    ) * data['break_weight']
    
    # Normalize the final factor
    rolling_mean = data['alpha_factor'].rolling(window=20, min_periods=10).mean()
    rolling_std = data['alpha_factor'].rolling(window=20, min_periods=10).std()
    data['final_alpha'] = (data['alpha_factor'] - rolling_mean) / (rolling_std + 1e-8)
    
    return data['final_alpha']
