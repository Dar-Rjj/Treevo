import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy import stats

def heuristics_v2(df):
    """
    Regime-Adaptive Price-Volume Convergence factor
    Combines price-volume cointegration, cross-asset momentum alignment, and regime detection
    """
    
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # 1. Price-Volume Cointegration
    # Calculate price-volume regression residuals
    def calc_pv_residuals(window_data):
        if len(window_data) < 5:
            return np.nan
        try:
            # Log transform for better stationarity
            log_volume = np.log(window_data['volume'].replace(0, np.nan))
            log_price = np.log(window_data['close'])
            
            # Remove NaN values
            valid_mask = ~(np.isnan(log_volume) | np.isnan(log_price))
            if valid_mask.sum() < 5:
                return np.nan
                
            log_volume_clean = log_volume[valid_mask]
            log_price_clean = log_price[valid_mask]
            
            # Price-volume regression
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                log_volume_clean, log_price_clean
            )
            
            # Calculate residuals (price deviation from volume relationship)
            predicted_price = intercept + slope * log_volume_clean
            residuals = log_price_clean - predicted_price
            
            return residuals.iloc[-1] if len(residuals) > 0 else np.nan
        except:
            return np.nan
    
    # Rolling price-volume residuals (21-day window)
    pv_residuals = []
    for i in range(len(data)):
        if i < 20:
            pv_residuals.append(np.nan)
            continue
            
        window_data = data.iloc[i-20:i+1]
        residual = calc_pv_residuals(window_data)
        pv_residuals.append(residual)
    
    data['pv_residual'] = pv_residuals
    
    # Extreme deviation signals (mean reversion)
    data['pv_zscore'] = (data['pv_residual'] - data['pv_residual'].rolling(window=63, min_periods=21).mean()) / \
                        data['pv_residual'].rolling(window=63, min_periods=21).std()
    
    # 2. Cross-Asset Momentum Alignment (using rolling correlation with market)
    def calc_momentum_alignment(window_data):
        if len(window_data) < 10:
            return np.nan
        
        # Calculate returns at different timeframes
        ret_5d = window_data['close'].pct_change(5)
        ret_10d = window_data['close'].pct_change(10)
        ret_21d = window_data['close'].pct_change(21)
        
        # Momentum convergence score
        mom_conv = np.sign(ret_5d.iloc[-1]) * np.sign(ret_10d.iloc[-1]) * np.sign(ret_21d.iloc[-1])
        mom_strength = (abs(ret_5d.iloc[-1]) + abs(ret_10d.iloc[-1]) + abs(ret_21d.iloc[-1])) / 3
        
        return mom_conv * mom_strength if not np.isnan(mom_conv) else np.nan
    
    momentum_scores = []
    for i in range(len(data)):
        if i < 20:
            momentum_scores.append(np.nan)
            continue
            
        window_data = data.iloc[i-20:i+1]
        mom_score = calc_momentum_alignment(window_data)
        momentum_scores.append(mom_score)
    
    data['momentum_alignment'] = momentum_scores
    
    # 3. Market Regime Detection
    # Volatility regime
    data['volatility_21d'] = data['close'].pct_change().rolling(window=21, min_periods=10).std()
    vol_regime = (data['volatility_21d'] > data['volatility_21d'].rolling(window=126, min_periods=63).quantile(0.7)).astype(int)
    
    # Efficiency regime (low autocorrelation = efficient market)
    def calc_efficiency(window_data):
        if len(window_data) < 21:
            return np.nan
        returns = window_data['close'].pct_change().dropna()
        if len(returns) < 20:
            return np.nan
        # First-order autocorrelation as inefficiency measure
        autocorr = returns.autocorr(lag=1)
        return 1 - abs(autocorr) if not np.isnan(autocorr) else 0.5
    
    efficiency_scores = []
    for i in range(len(data)):
        if i < 20:
            efficiency_scores.append(np.nan)
            continue
            
        window_data = data.iloc[i-20:i+1]
        eff_score = calc_efficiency(window_data)
        efficiency_scores.append(eff_score)
    
    data['efficiency'] = efficiency_scores
    eff_regime = (data['efficiency'] > data['efficiency'].rolling(window=63, min_periods=21).median()).astype(int)
    
    # 4. Signal Integration
    # Regime-weighted factor construction
    data['regime_weight'] = 0.6 * vol_regime + 0.4 * eff_regime
    
    # Volume-confirmed convergence validation
    volume_trend = data['volume'].rolling(window=5, min_periods=3).apply(
        lambda x: 1 if (x.iloc[-1] > x.iloc[-3:].mean()) else -1, raw=False
    )
    
    # Final factor calculation
    # Mean reversion component (weighted by regime)
    mean_rev_component = -data['pv_zscore'] * (1 + data['regime_weight'])
    
    # Momentum alignment component
    mom_component = data['momentum_alignment'] * (1 - data['regime_weight'])
    
    # Volume confirmation
    volume_confirmation = volume_trend * np.sign(data['momentum_alignment'])
    
    # Combined factor
    factor = (0.5 * mean_rev_component + 0.3 * mom_component + 0.2 * volume_confirmation)
    
    # Normalize the factor
    factor_normalized = (factor - factor.rolling(window=126, min_periods=63).mean()) / \
                       factor.rolling(window=126, min_periods=63).std()
    
    return factor_normalized
