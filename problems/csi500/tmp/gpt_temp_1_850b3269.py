import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate returns
    data['return_1d'] = data['close'].pct_change()
    data['return_5d'] = data['close'] / data['close'].shift(5) - 1
    
    # Price acceleration calculations
    data['price_acceleration'] = (data['return_5d'] - data['return_5d'].shift(5)) / 5
    data['price_jerk'] = (data['price_acceleration'] - data['price_acceleration'].shift(5)) / 5
    
    # Volume calculations
    data['volume_5d_change'] = data['volume'] / data['volume'].shift(5) - 1
    data['volume_acceleration'] = (data['volume_5d_change'] - data['volume_5d_change'].shift(5)) / 5
    
    # Autocorrelation decay calculations
    decay_ratios = []
    for i in range(len(data)):
        if i < 20:
            decay_ratios.append(np.nan)
            continue
        
        returns_window = data['return_1d'].iloc[i-19:i+1]
        if len(returns_window.dropna()) < 10:
            decay_ratios.append(np.nan)
            continue
        
        autocorr_1 = returns_window.autocorr(lag=1)
        autocorr_5 = returns_window.autocorr(lag=5) if len(returns_window) >= 6 else np.nan
        
        if autocorr_1 is not None and autocorr_5 is not None and abs(autocorr_1) > 1e-6:
            decay_ratio = autocorr_5 / autocorr_1
        else:
            decay_ratio = np.nan
        decay_ratios.append(decay_ratio)
    
    data['decay_ratio'] = decay_ratios
    
    # Hurst exponent approximation using rescaled range
    hurst_approx = []
    for i in range(len(data)):
        if i < 20:
            hurst_approx.append(np.nan)
            continue
        
        prices_window = data['close'].iloc[i-19:i+1]
        if len(prices_window.dropna()) < 20:
            hurst_approx.append(np.nan)
            continue
        
        log_prices = np.log(prices_window)
        mean_log = log_prices.mean()
        deviations = log_prices - mean_log
        cumulative_deviations = deviations.cumsum()
        
        R = cumulative_deviations.max() - cumulative_deviations.min()
        S = deviations.std()
        
        if S > 0:
            hurst = np.log(R/S) / np.log(20)
        else:
            hurst = np.nan
        hurst_approx.append(hurst)
    
    data['hurst_approx'] = hurst_approx
    
    # Variance ratio test
    variance_ratios = []
    for i in range(len(data)):
        if i < 20:
            variance_ratios.append(np.nan)
            continue
        
        returns_window = data['return_1d'].iloc[i-19:i+1]
        if len(returns_window.dropna()) < 15:
            variance_ratios.append(np.nan)
            continue
        
        var_1d = returns_window.var()
        returns_5d = (data['close'].iloc[i-19:i+1] / data['close'].iloc[i-24:i-4] - 1).dropna()
        var_5d = returns_5d.var() if len(returns_5d) > 0 else np.nan
        
        if var_1d > 0 and var_5d is not None:
            variance_ratio = var_5d / (5 * var_1d)
        else:
            variance_ratio = np.nan
        variance_ratios.append(variance_ratio)
    
    data['variance_ratio'] = variance_ratios
    
    # Zero-crossing count for oscillation frequency
    zero_crossings = []
    for i in range(len(data)):
        if i < 20:
            zero_crossings.append(np.nan)
            continue
        
        prices_window = data['close'].iloc[i-19:i+1]
        if len(prices_window.dropna()) < 20:
            zero_crossings.append(np.nan)
            continue
        
        # Detrend using linear regression
        x = np.arange(len(prices_window))
        y = prices_window.values
        coef = np.polyfit(x, y, 1)
        detrended = y - (coef[0] * x + coef[1])
        
        # Count zero crossings
        crossings = np.where(np.diff(np.sign(detrended)))[0]
        zero_crossings.append(len(crossings))
    
    data['zero_crossings'] = zero_crossings
    
    # Structural break detection (variance ratio)
    structural_breaks = []
    for i in range(len(data)):
        if i < 60:
            structural_breaks.append(np.nan)
            continue
        
        recent_returns = data['return_1d'].iloc[i-19:i+1].dropna()
        historical_returns = data['return_1d'].iloc[i-59:i-19].dropna()
        
        if len(recent_returns) >= 10 and len(historical_returns) >= 20:
            recent_var = recent_returns.var()
            historical_var = historical_returns.var()
            
            if historical_var > 0:
                structural_break = recent_var / historical_var
            else:
                structural_break = np.nan
        else:
            structural_break = np.nan
        structural_breaks.append(structural_break)
    
    data['structural_break'] = structural_breaks
    
    # Regime persistence (sign consistency)
    regime_persistence = []
    for i in range(len(data)):
        if i < 60:
            regime_persistence.append(np.nan)
            continue
        
        returns_5d_window = data['return_5d'].iloc[i-59:i+1].dropna()
        if len(returns_5d_window) >= 10:
            signs = np.sign(returns_5d_window)
            persistence = (signs == signs.iloc[-1]).mean()
        else:
            persistence = np.nan
        regime_persistence.append(persistence)
    
    data['regime_persistence'] = regime_persistence
    
    # Volume regime shift
    volume_regime = []
    for i in range(len(data)):
        if i < 60:
            volume_regime.append(np.nan)
            continue
        
        volume_window = data['volume'].iloc[i-59:i+1].dropna()
        if len(volume_window) >= 30:
            current_volume = data['volume'].iloc[i]
            volume_percentile = np.percentile(volume_window, 60)
            
            if volume_percentile > 0:
                volume_shift = current_volume / volume_percentile
            else:
                volume_shift = np.nan
        else:
            volume_shift = np.nan
        volume_regime.append(volume_shift)
    
    data['volume_regime'] = volume_regime
    
    # Short-term signals
    data['acceleration_reversal'] = -data['price_jerk'] * data['price_acceleration']
    data['volume_confirmation'] = np.sign(data['return_5d']) * data['volume_5d_change']
    data['high_freq_mean_reversion'] = -data['return_1d'] * (1 - data['decay_ratio'])
    
    # Medium-term signals
    data['divergence_momentum'] = (data['price_acceleration'] - data['volume_acceleration']) * data['return_5d']
    data['decay_adjusted_trend'] = data['return_5d'] * (1 + data['hurst_approx'])
    data['oscillation_breakout'] = data['return_5d'] * data['zero_crossings']
    
    # Long-term signals
    data['structural_break_signal'] = data['structural_break']
    data['regime_persistence_signal'] = data['regime_persistence']
    data['volume_regime_signal'] = data['volume_regime']
    
    # Timeframe weighting
    data['short_term_weight'] = 1 / (1 + abs(data['decay_ratio']))
    data['medium_term_weight'] = data['variance_ratio']
    data['long_term_weight'] = data['structural_break']
    
    # Signal combination
    short_term_signals = ['acceleration_reversal', 'volume_confirmation', 'high_freq_mean_reversion']
    medium_term_signals = ['divergence_momentum', 'decay_adjusted_trend', 'oscillation_breakout']
    long_term_signals = ['structural_break_signal', 'regime_persistence_signal', 'volume_regime_signal']
    
    data['short_term_component'] = data[short_term_signals].mean(axis=1) * data['short_term_weight']
    data['medium_term_component'] = data[medium_term_signals].mean(axis=1) * data['medium_term_weight']
    data['long_term_component'] = data[long_term_signals].mean(axis=1) * data['long_term_weight']
    
    # Final factor
    data['factor'] = data['short_term_component'] + data['medium_term_component'] + data['long_term_component']
    
    return data['factor']
