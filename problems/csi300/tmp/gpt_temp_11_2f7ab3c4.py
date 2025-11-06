import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Price-Volume Fractal Dynamics factor combining multi-scale fractal analysis,
    fractal momentum structure, volume fractal dynamics, and intraday fractal patterns.
    """
    
    # Helper function for Hurst exponent estimation
    def hurst_exponent(series, window):
        """Calculate Hurst exponent using rescaled range analysis"""
        if len(series) < window:
            return np.nan
        
        # Create lagged differences
        lags = range(2, min(window, len(series)//2))
        tau = []
        for lag in lags:
            # Rescaled range calculation
            series_lag = series.diff(lag).dropna()
            if len(series_lag) < 2:
                tau.append(np.nan)
                continue
            
            mean_lag = series_lag.mean()
            deviations = series_lag - mean_lag
            Z = deviations.cumsum()
            R = Z.max() - Z.min()
            S = series_lag.std()
            
            if S > 0:
                tau.append(R / S)
            else:
                tau.append(np.nan)
        
        tau = [t for t in tau if not np.isnan(t)]
        if len(tau) < 2:
            return 0.5
        
        # Linear regression to get Hurst exponent
        lags_valid = range(2, 2 + len(tau))
        try:
            hurst = np.polyfit(np.log(lags_valid), np.log(tau), 1)[0]
            return hurst
        except:
            return 0.5
    
    # Multi-Scale Fractal Analysis
    def calculate_fractal_dimensions(price_series):
        """Calculate fractal dimensions at different time scales"""
        hurst_5 = price_series.rolling(window=5, min_periods=5).apply(
            lambda x: hurst_exponent(pd.Series(x), 5), raw=False
        )
        hurst_10 = price_series.rolling(window=10, min_periods=10).apply(
            lambda x: hurst_exponent(pd.Series(x), 10), raw=False
        )
        hurst_20 = price_series.rolling(window=20, min_periods=20).apply(
            lambda x: hurst_exponent(pd.Series(x), 20), raw=False
        )
        return hurst_5, hurst_10, hurst_20
    
    # Calculate price fractal dimensions
    price_hurst_5, price_hurst_10, price_hurst_20 = calculate_fractal_dimensions(df['close'])
    
    # Calculate volume fractal dimensions
    volume_hurst_5, volume_hurst_10, volume_hurst_20 = calculate_fractal_dimensions(df['volume'])
    
    # Multi-Scale Volatility Fractal
    def volatility_scaling(returns, windows):
        """Calculate volatility scaling across different timeframes"""
        vol_scaling = []
        for window in windows:
            vol = returns.rolling(window=window).std()
            vol_scaling.append(vol)
        return vol_scaling
    
    returns = df['close'].pct_change()
    vol_3, vol_5, vol_10 = volatility_scaling(returns, [3, 5, 10])
    
    # Price-Volume Fractal Correlation
    def fractal_correlation(hurst_price, hurst_volume, window=10):
        """Calculate correlation between price and volume fractal dimensions"""
        return hurst_price.rolling(window=window).corr(hurst_volume)
    
    pv_fractal_corr = fractal_correlation(price_hurst_10, volume_hurst_10)
    
    # Fractal Momentum Structure
    def fractal_momentum_gradient(price, windows):
        """Calculate momentum persistence across different timeframes"""
        momentum_grad = []
        for window in windows:
            mom = price.pct_change(window)
            momentum_grad.append(mom)
        return momentum_grad
    
    mom_3, mom_5, mom_10 = fractal_momentum_gradient(df['close'], [3, 5, 10])
    
    # Fractal Acceleration
    def fractal_acceleration(momentum_short, momentum_long):
        """Calculate momentum change scaling"""
        return momentum_short - momentum_long
    
    mom_accel = fractal_acceleration(mom_3, mom_10)
    
    # Volatility-Fractal Interaction
    vol_fractal_interaction = vol_5 * price_hurst_5
    
    # Volume Fractal Dynamics
    volume_fractal_momentum = df['volume'].pct_change(5)
    
    # Price-Volume Fractal Divergence
    pv_fractal_divergence = price_hurst_5 - volume_hurst_5
    
    # Intraday Fractal Patterns
    def intraday_range_fractal(high, low, window=5):
        """Calculate daily range scaling pattern"""
        daily_range = (high - low) / high
        range_hurst = daily_range.rolling(window=window, min_periods=window).apply(
            lambda x: hurst_exponent(pd.Series(x), window), raw=False
        )
        return range_hurst
    
    range_fractal = intraday_range_fractal(df['high'], df['low'])
    
    # Gap Fractal Analysis
    def gap_fractal_analysis(open_price, close_prev):
        """Analyze gap size distribution pattern"""
        gap_size = (open_price - close_prev.shift(1)) / close_prev.shift(1)
        gap_abs = gap_size.abs()
        gap_hurst = gap_abs.rolling(window=5, min_periods=5).apply(
            lambda x: hurst_exponent(pd.Series(x), 5), raw=False
        )
        return gap_hurst
    
    gap_fractal = gap_fractal_analysis(df['open'], df['close'])
    
    # Efficiency Fractal
    def efficiency_fractal(open_price, close_price, window=5):
        """Calculate open-close efficiency scaling"""
        efficiency = (close_price - open_price) / open_price
        eff_hurst = efficiency.rolling(window=window, min_periods=window).apply(
            lambda x: hurst_exponent(pd.Series(x), window), raw=False
        )
        return eff_hurst
    
    efficiency_fractal = efficiency_fractal(df['open'], df['close'])
    
    # Signal Synthesis
    # Fractal Alignment (consistency across fractal measures)
    fractal_alignment = (price_hurst_5.rolling(5).std() + 
                        price_hurst_10.rolling(5).std() + 
                        price_hurst_20.rolling(5).std()) / 3
    
    # Multi-Scale Convergence (agreement across time horizons)
    multi_scale_convergence = (price_hurst_5 + price_hurst_10 + price_hurst_20) / 3
    
    # Fractal-Momentum Integration
    fractal_momentum_integration = price_hurst_5 * mom_5
    
    # Composite Factor Construction
    # Core Factor = Fractal Dimension × Fractal Momentum
    core_factor = price_hurst_5 * mom_5
    
    # Enhanced Factor = Core × Volume Fractal Divergence × Multi-Scale Convergence
    enhanced_factor = (core_factor * pv_fractal_divergence * multi_scale_convergence)
    
    # Final Factor = Enhanced × Intraday Fractal Patterns × Fractal Alignment
    intraday_patterns = (range_fractal + gap_fractal + efficiency_fractal) / 3
    
    final_factor = enhanced_factor * intraday_patterns * (1 / (fractal_alignment + 1e-6))
    
    # Normalize and clean the factor
    final_factor = final_factor.replace([np.inf, -np.inf], np.nan)
    final_factor = (final_factor - final_factor.rolling(20).mean()) / final_factor.rolling(20).std()
    
    return final_factor
