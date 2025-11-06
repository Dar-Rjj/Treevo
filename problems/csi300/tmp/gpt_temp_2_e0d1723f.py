import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import linregress

def heuristics_v2(df):
    """
    Multi-Scale Price-Volume Fractal Dynamics with Regime-Sensitive Momentum Persistence
    """
    data = df.copy()
    
    # Initialize result series
    alpha = pd.Series(index=data.index, dtype=float)
    
    # Fractal Price-Volume Structure Analysis
    # Daily Price Efficiency Ratio
    data['price_efficiency'] = np.abs(data['close'] - data['open']) / (data['high'] - data['low']).replace(0, np.nan)
    
    # Multi-scale fractal dimensions
    for window in [5, 10, 20]:
        data[f'price_fractal_{window}'] = calculate_fractal_dimension(data['close'], window)
        data[f'volume_fractal_{window}'] = calculate_fractal_dimension(data['volume'], window)
    
    # Volume Clustering Index
    data['volume_median_20'] = data['volume'].rolling(window=20, min_periods=10).median()
    data['volume_deviation'] = data['volume'] / data['volume_median_20']
    
    # Volume clustering consecutive days
    data['above_median_volume'] = (data['volume'] > data['volume_median_20']).astype(int)
    data['volume_cluster_count'] = data['above_median_volume'].groupby(
        (data['above_median_volume'] != data['above_median_volume'].shift()).cumsum()
    ).cumcount() + 1
    
    # Volume-Price Fractal Divergence
    for window in [5, 10, 20]:
        data[f'fractal_divergence_{window}'] = (
            data[f'price_fractal_{window}'] - data[f'volume_fractal_{window}']
        )
    
    # Multi-Timeframe Fractal Coherence
    for window in [5, 10, 20]:
        data[f'fractal_coherence_{window}'] = 1 - np.abs(
            data[f'price_fractal_{window}'] - data[f'volume_fractal_{window}']
        )
    
    # Regime-Sensitive Momentum Persistence Framework
    # Market regime identification
    data['fractal_regime'] = identify_fractal_regime(data)
    
    # Regime-adaptive momentum
    data['regime_momentum'] = calculate_regime_momentum(data)
    
    # Momentum autocorrelation
    for window in [5, 10, 20]:
        data[f'momentum_acf_{window}'] = calculate_momentum_autocorrelation(data, window)
    
    # Regime-weighted persistence
    data['weighted_persistence'] = calculate_weighted_persistence(data)
    
    # Price-Volume Multi-Scale Entropy Analysis
    for window in [5, 10, 20]:
        data[f'price_entropy_{window}'] = calculate_sample_entropy(data['close'], window)
    
    data['volume_entropy'] = calculate_sample_entropy(data['volume'], 10)
    data['volume_change_entropy'] = calculate_sample_entropy(data['volume'].diff(), 10)
    
    # Intraday Microstructure Integration
    data['opening_pressure'] = calculate_opening_pressure(data)
    data['vwap_deviation'] = calculate_vwap_deviation(data)
    data['volume_skew'] = calculate_volume_skew(data)
    
    # Multi-Fractal Regime Transition Detection
    data['regime_transition_prob'] = calculate_regime_transition_probability(data)
    
    # Composite Alpha Factor Construction
    # Fractal-Coherent Momentum Score
    fractal_coherent_momentum = (
        data['regime_momentum'] * 
        (0.4 * data['fractal_coherence_5'] + 
         0.35 * data['fractal_coherence_10'] + 
         0.25 * data['fractal_coherence_20'])
    )
    
    # Apply entropy-based quality adjustment
    entropy_adjustment = 1 - (0.3 * data['price_entropy_5'] + 
                             0.4 * data['price_entropy_10'] + 
                             0.3 * data['price_entropy_20']) / 3
    
    fractal_coherent_momentum *= entropy_adjustment
    
    # Microstructure-Enhanced Signal
    microstructure_signal = (
        data['opening_pressure'].fillna(0) * 0.3 +
        data['vwap_deviation'].fillna(0) * 0.4 +
        data['volume_skew'].fillna(0) * 0.3
    )
    
    # Regime Transition Multiplier
    regime_multiplier = 1 + data['regime_transition_prob'] * 0.5
    
    # Final Alpha Integration
    alpha = (
        fractal_coherent_momentum * 
        (1 + microstructure_signal) * 
        regime_multiplier
    )
    
    # Apply bounds and normalization
    alpha = alpha.clip(lower=-3, upper=3)
    alpha = (alpha - alpha.rolling(window=63, min_periods=21).mean()) / alpha.rolling(window=63, min_periods=21).std()
    
    return alpha

# Helper functions
def calculate_fractal_dimension(series, window):
    """Calculate fractal dimension using R/S analysis"""
    result = pd.Series(index=series.index, dtype=float)
    
    for i in range(window, len(series)):
        window_data = series.iloc[i-window:i]
        if len(window_data) < window:
            result.iloc[i] = np.nan
            continue
            
        # R/S analysis
        mean_val = window_data.mean()
        deviations = window_data - mean_val
        cumulative_deviations = deviations.cumsum()
        R = cumulative_deviations.max() - cumulative_deviations.min()
        S = window_data.std()
        
        if S > 0:
            # Simple fractal dimension approximation
            result.iloc[i] = 2 - np.log(R/S) / np.log(window)
        else:
            result.iloc[i] = np.nan
    
    return result

def identify_fractal_regime(data):
    """Identify market regime based on fractal characteristics"""
    regime = pd.Series(index=data.index, dtype=str)
    
    for i in range(20, len(data)):
        current_fractal = data['price_fractal_20'].iloc[i]
        
        if pd.isna(current_fractal):
            regime.iloc[i] = 'unknown'
        elif current_fractal > 1.6:
            regime.iloc[i] = 'trending'
        elif current_fractal < 1.4:
            regime.iloc[i] = 'mean_reverting'
        else:
            regime.iloc[i] = 'random_walk'
    
    return regime

def calculate_regime_momentum(data):
    """Calculate regime-adaptive momentum"""
    momentum = pd.Series(index=data.index, dtype=float)
    
    for i in range(20, len(data)):
        regime = data['fractal_regime'].iloc[i]
        
        if regime == 'trending':
            # Directional momentum with acceleration
            mom_5 = data['close'].iloc[i] / data['close'].iloc[i-5] - 1
            mom_10 = data['close'].iloc[i] / data['close'].iloc[i-10] - 1
            acceleration = mom_5 - mom_10
            momentum.iloc[i] = mom_5 + 0.5 * acceleration
            
        elif regime == 'mean_reverting':
            # Mean reversion probability
            current_price = data['close'].iloc[i]
            ma_20 = data['close'].iloc[i-20:i].mean()
            std_20 = data['close'].iloc[i-20:i].std()
            
            if std_20 > 0:
                z_score = (current_price - ma_20) / std_20
                momentum.iloc[i] = -z_score * 0.1  # Mean reversion signal
            else:
                momentum.iloc[i] = 0
                
        else:  # random_walk
            # Short-term momentum with decay
            mom_3 = data['close'].iloc[i] / data['close'].iloc[i-3] - 1
            volatility = data['close'].iloc[i-20:i].std()
            if volatility > 0:
                momentum.iloc[i] = mom_3 / volatility * 0.5
            else:
                momentum.iloc[i] = mom_3 * 0.5
    
    return momentum

def calculate_momentum_autocorrelation(data, window):
    """Calculate momentum autocorrelation"""
    acf = pd.Series(index=data.index, dtype=float)
    
    for i in range(window + 10, len(data)):
        momentum_series = (data['close'].iloc[i-window-10:i] / 
                          data['close'].iloc[i-window-11:i-1] - 1)
        
        if len(momentum_series) >= window and momentum_series.std() > 0:
            try:
                lag1_corr = momentum_series.autocorr(lag=1)
                acf.iloc[i] = lag1_corr if not pd.isna(lag1_corr) else 0
            except:
                acf.iloc[i] = 0
        else:
            acf.iloc[i] = 0
    
    return acf

def calculate_weighted_persistence(data):
    """Calculate regime-weighted persistence"""
    persistence = pd.Series(index=data.index, dtype=float)
    
    for i in range(20, len(data)):
        regime = data['fractal_regime'].iloc[i]
        
        if regime == 'trending':
            weights = [0.5, 0.3, 0.2]  # Emphasize short-term
        elif regime == 'mean_reverting':
            weights = [0.2, 0.3, 0.5]  # Emphasize long-term
        else:
            weights = [0.4, 0.4, 0.2]  # Balanced
            
        weighted_pers = (
            weights[0] * data['momentum_acf_5'].iloc[i] +
            weights[1] * data['momentum_acf_10'].iloc[i] +
            weights[2] * data['momentum_acf_20'].iloc[i]
        )
        persistence.iloc[i] = weighted_pers
    
    return persistence

def calculate_sample_entropy(series, window):
    """Calculate sample entropy for complexity analysis"""
    entropy = pd.Series(index=series.index, dtype=float)
    
    for i in range(window + 5, len(series)):
        window_data = series.iloc[i-window:i].values
        
        if len(window_data) < window or np.std(window_data) == 0:
            entropy.iloc[i] = np.nan
            continue
            
        # Simplified sample entropy calculation
        m = 2
        r = 0.2 * np.std(window_data)
        
        # Count similar patterns
        patterns_m = []
        patterns_m1 = []
        
        for j in range(len(window_data) - m):
            pattern_m = window_data[j:j+m]
            pattern_m1 = window_data[j:j+m+1]
            
            for k in range(j+1, len(window_data) - m):
                pattern_m_comp = window_data[k:k+m]
                pattern_m1_comp = window_data[k:k+m+1]
                
                if np.max(np.abs(pattern_m - pattern_m_comp)) <= r:
                    patterns_m.append(1)
                    if np.max(np.abs(pattern_m1 - pattern_m1_comp)) <= r:
                        patterns_m1.append(1)
        
        if len(patterns_m) > 0 and len(patterns_m1) > 0:
            A = len(patterns_m1)
            B = len(patterns_m)
            entropy.iloc[i] = -np.log(A / B) if B > 0 and A > 0 else 0
        else:
            entropy.iloc[i] = 0
    
    return entropy

def calculate_opening_pressure(data):
    """Calculate opening auction pressure"""
    pressure = pd.Series(index=data.index, dtype=float)
    
    for i in range(1, len(data)):
        prev_high = data['high'].iloc[i-1]
        prev_low = data['low'].iloc[i-1]
        current_open = data['open'].iloc[i]
        
        if prev_high != prev_low:
            pressure.iloc[i] = (current_open - (prev_high + prev_low)/2) / ((prev_high - prev_low)/2)
        else:
            pressure.iloc[i] = 0
    
    return pressure

def calculate_vwap_deviation(data):
    """Calculate VWAP deviation from close"""
    vwap_dev = pd.Series(index=data.index, dtype=float)
    
    for i in range(len(data)):
        # Simplified VWAP approximation using typical price
        typical_price = (data['high'].iloc[i] + data['low'].iloc[i] + data['close'].iloc[i]) / 3
        vwap_approx = typical_price  # Simplified for daily data
        
        if data['close'].iloc[i] > 0:
            vwap_dev.iloc[i] = (data['close'].iloc[i] - vwap_approx) / data['close'].iloc[i]
        else:
            vwap_dev.iloc[i] = 0
    
    return vwap_dev

def calculate_volume_skew(data):
    """Calculate intraday volume skew (simplified)"""
    # Using daily volume patterns as proxy
    volume_skew = pd.Series(index=data.index, dtype=float)
    
    for i in range(10, len(data)):
        recent_volume = data['volume'].iloc[i-10:i]
        if len(recent_volume) >= 5 and recent_volume.std() > 0:
            # Skewness of recent volume distribution
            mean_vol = recent_volume.mean()
            std_vol = recent_volume.std()
            if std_vol > 0:
                skew = ((recent_volume - mean_vol) ** 3).mean() / (std_vol ** 3)
                volume_skew.iloc[i] = skew
            else:
                volume_skew.iloc[i] = 0
        else:
            volume_skew.iloc[i] = 0
    
    return volume_skew

def calculate_regime_transition_probability(data):
    """Calculate probability of regime transition"""
    transition_prob = pd.Series(index=data.index, dtype=float)
    
    for i in range(20, len(data)):
        # Monitor changes in fractal dimensions
        fractal_change_5 = np.abs(data['price_fractal_5'].iloc[i] - data['price_fractal_5'].iloc[i-5])
        fractal_change_10 = np.abs(data['price_fractal_10'].iloc[i] - data['price_fractal_10'].iloc[i-10])
        fractal_change_20 = np.abs(data['price_fractal_20'].iloc[i] - data['price_fractal_20'].iloc[i-20])
        
        avg_change = (fractal_change_5 + fractal_change_10 + fractal_change_20) / 3
        transition_prob.iloc[i] = min(avg_change * 5, 1)  # Scale to probability
    
    return transition_prob
