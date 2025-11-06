import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis

def heuristics_v2(df):
    """
    Nonlinear Regime Shift Detection with Fractal Market Structure
    Combines price fractality, volume asymmetry, and microstructure efficiency
    to detect regime transitions and generate alpha signals.
    """
    result = pd.Series(index=df.index, dtype=float)
    
    for i in range(max(20, len(df))):
        if i < 20:
            result.iloc[i] = 0
            continue
            
        current_data = df.iloc[:i+1]
        
        # 1. Multi-Scale Price Fractality
        hurst_signal = _calculate_hurst_exponent(current_data['close'].iloc[-20:])
        fractal_dim_signal = _calculate_fractal_dimension(
            current_data['high'].iloc[-10:],
            current_data['low'].iloc[-10:],
            current_data['close'].iloc[-10:]
        )
        price_regime_signal = _detect_price_regime_transition(hurst_signal, fractal_dim_signal)
        
        # 2. Volume Distribution Asymmetry
        volume_skew = _calculate_volume_skewness(current_data['volume'].iloc[-15:])
        volume_fractal = _calculate_volume_fractal_pattern(current_data['volume'].iloc[-15:])
        volume_price_coupling = _detect_volume_price_coupling(
            current_data['close'].iloc[-15:],
            current_data['volume'].iloc[-15:]
        )
        volume_regime_signal = _detect_volume_regime_transition(volume_skew, volume_fractal, volume_price_coupling)
        
        # 3. Market Microstructure Efficiency
        efficiency_signal = _calculate_microstructure_efficiency(
            current_data['close'].iloc[-10:],
            current_data['volume'].iloc[-10:]
        )
        
        # 4. Synthesize Alpha Factor
        alpha_signal = _synthesize_alpha_factor(
            price_regime_signal,
            volume_regime_signal,
            efficiency_signal
        )
        
        result.iloc[i] = alpha_signal
    
    return result

def _calculate_hurst_exponent(close_prices):
    """Calculate Hurst exponent approximation for persistence detection"""
    if len(close_prices) < 20:
        return 0
    
    returns = close_prices.pct_change().dropna()
    if len(returns) < 10:
        return 0
    
    # Simplified Hurst calculation using variance ratio
    lags = [2, 5, 10]
    variances = []
    
    for lag in lags:
        if len(returns) >= lag:
            lag_returns = returns.rolling(lag).sum().dropna()
            if len(lag_returns) > 0:
                var_ratio = np.var(lag_returns) / (lag * np.var(returns))
                variances.append(var_ratio)
    
    if len(variances) > 0:
        hurst_approx = 0.5 + 0.5 * np.log2(np.mean(variances))
        return np.clip(hurst_approx, 0, 1)
    return 0.5

def _calculate_fractal_dimension(high_prices, low_prices, close_prices):
    """Calculate box-counting dimension approximation"""
    if len(high_prices) < 10:
        return 1.5
    
    price_range = high_prices - low_prices
    close_changes = close_prices.diff().abs()
    
    if price_range.std() == 0 or close_changes.std() == 0:
        return 1.5
    
    # Simplified fractal dimension using range vs movement ratio
    range_volatility = price_range.mean() / close_prices.mean()
    movement_volatility = close_changes.mean() / close_prices.mean()
    
    if movement_volatility > 0:
        fractal_dim = 1 + (range_volatility / movement_volatility)
        return np.clip(fractal_dim, 1.0, 2.0)
    
    return 1.5

def _detect_price_regime_transition(hurst, fractal_dim):
    """Detect price regime transitions based on fractality"""
    # Persistence to Anti-Persistence (trending to mean-reverting)
    if hurst > 0.6 and fractal_dim < 1.4:
        # Strong trending regime
        return -1  # Potential reversal signal
    elif hurst < 0.4 and fractal_dim > 1.6:
        # Strong mean-reverting regime  
        return 1   # Potential momentum initiation
    elif 0.4 <= hurst <= 0.6 and 1.4 <= fractal_dim <= 1.6:
        # Random walk regime
        return 0
    else:
        # Mixed signals
        return (hurst - 0.5) * (fractal_dim - 1.5)

def _calculate_volume_skewness(volume):
    """Calculate volume distribution asymmetry"""
    if len(volume) < 15:
        return 0
    
    volume_skew = skew(volume)
    volume_kurt = kurtosis(volume)
    
    # Combine skewness and kurtosis for asymmetry measure
    asymmetry = volume_skew * (1 + abs(volume_kurt) / 10)
    return np.clip(asymmetry, -2, 2)

def _calculate_volume_fractal_pattern(volume):
    """Calculate volume persistence across time scales"""
    if len(volume) < 15:
        return 0
    
    # Multi-scale autocorrelation
    lags = [1, 3, 5]
    autocorrs = []
    
    for lag in lags:
        if len(volume) > lag:
            autocorr = volume.autocorr(lag=lag)
            if not np.isnan(autocorr):
                autocorrs.append(autocorr)
    
    if len(autocorrs) > 0:
        return np.mean(autocorrs)
    return 0

def _detect_volume_price_coupling(close_prices, volume):
    """Detect coupling between price and volume fractality"""
    if len(close_prices) < 15 or len(volume) < 15:
        return 0
    
    price_volatility = close_prices.pct_change().std()
    volume_volatility = volume.pct_change().std()
    
    if volume_volatility == 0:
        return 0
    
    # Correlation between absolute price changes and volume changes
    price_changes = close_prices.pct_change().abs().dropna()
    volume_changes = volume.pct_change().abs().dropna()
    
    if len(price_changes) > 5 and len(volume_changes) > 5:
        min_len = min(len(price_changes), len(volume_changes))
        correlation = np.corrcoef(price_changes.iloc[-min_len:], 
                                volume_changes.iloc[-min_len:])[0, 1]
        if np.isnan(correlation):
            return 0
        return correlation
    return 0

def _detect_volume_regime_transition(skewness, fractal, coupling):
    """Detect volume regime transitions"""
    # Strong asymmetric volume with persistence
    if abs(skewness) > 1.0 and fractal > 0.3:
        return skewness * fractal
    
    # Volume-price decoupling
    if abs(coupling) < 0.2:
        return -0.5  # Reduced predictability
    
    # Volume-price coupling with symmetric distribution
    if abs(skewness) < 0.5 and coupling > 0.5:
        return 0.5   # Efficient market regime
    
    return (skewness + fractal + coupling) / 3

def _calculate_microstructure_efficiency(close_prices, volume):
    """Calculate market microstructure efficiency"""
    if len(close_prices) < 10 or len(volume) < 10:
        return 0
    
    price_changes = close_prices.pct_change().dropna()
    volume_changes = volume.pct_change().dropna()
    
    if len(price_changes) < 5 or len(volume_changes) < 5:
        return 0
    
    # Simplified mutual information approximation using variance ratio
    price_variance = price_changes.var()
    volume_variance = volume_changes.var()
    
    if price_variance == 0 or volume_variance == 0:
        return 0
    
    # Efficiency measure: lower correlation suggests more efficient market
    min_len = min(len(price_changes), len(volume_changes))
    correlation = np.corrcoef(price_changes.iloc[-min_len:], 
                            volume_changes.iloc[-min_len:])[0, 1]
    
    if np.isnan(correlation):
        return 0
    
    # High efficiency = low predictability, Low efficiency = high predictability
    efficiency = 1 - abs(correlation)
    return efficiency

def _synthesize_alpha_factor(price_signal, volume_signal, efficiency_signal):
    """Synthesize final alpha factor from regime components"""
    
    # Combine signals with efficiency weighting
    raw_signal = price_signal * volume_signal
    
    # Efficiency acts as signal strength modulator
    # Low efficiency (high predictability) strengthens signals
    # High efficiency (low predictability) weakens signals
    efficiency_weight = 1 - efficiency_signal
    
    final_signal = raw_signal * efficiency_weight
    
    # Apply non-linear transformation for regime emphasis
    if abs(final_signal) > 0.3:
        # Strong regime shift
        final_signal = np.sign(final_signal) * (abs(final_signal) ** 0.7)
    elif abs(final_signal) < 0.1:
        # Weak or conflicting signals
        final_signal *= 0.5
    
    return np.clip(final_signal, -1, 1)
