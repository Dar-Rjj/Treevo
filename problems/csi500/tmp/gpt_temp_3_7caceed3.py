import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Fractal Efficiency Momentum with Liquidity Acceleration factor
    """
    # Make copy to avoid modifying original data
    data = df.copy()
    
    # 1. Multi-Timeframe Efficiency-Fractal Analysis
    # Calculate daily price amplitude efficiency
    data['amplitude'] = (data['high'] - data['low']) / data['close']
    data['amplitude_roc_5'] = data['amplitude'].pct_change(5)
    data['amplitude_roc_10'] = data['amplitude'].pct_change(10)
    
    # Fractal efficiency momentum calculations
    data['fractal_eff_5'] = data['amplitude_roc_5'].rolling(window=5).mean()
    data['fractal_eff_10'] = data['amplitude_roc_10'].rolling(window=10).mean()
    data['fractal_eff_accel'] = data['fractal_eff_5'].diff(3)
    
    # Fractal efficiency convergence
    data['fractal_divergence'] = data['fractal_eff_5'] - data['fractal_eff_10']
    data['convergence_strength'] = data['fractal_divergence'].abs()
    
    # 2. Volume-Amplitude Efficiency Dynamics
    # Volume slope and correlation
    data['volume_slope_5'] = data['volume'].rolling(window=5).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 5 else np.nan
    )
    
    # Volume-amplitude efficiency correlation (3-day rolling)
    data['vol_amp_corr'] = data['volume'].rolling(window=3).corr(data['amplitude'])
    
    # Amount-based efficiency
    data['amount_per_trade'] = data['amount'] / data['volume'].replace(0, np.nan)
    data['amount_roc_5'] = data['amount_per_trade'].pct_change(5)
    
    # 3. Liquidity Acceleration Integration
    # Volume acceleration (2nd derivative)
    data['volume_velocity'] = data['volume'].diff(3)
    data['volume_accel'] = data['volume_velocity'].diff(3)
    
    # Dollar volume velocity
    data['dollar_volume'] = data['close'] * data['volume']
    data['dollar_vol_velocity'] = data['dollar_volume'].diff(3)
    
    # Efficiency flow acceleration
    data['efficiency_flow'] = data['fractal_eff_5'].diff(3)
    
    # 4. Fractal Breakout Efficiency Detection
    # Range efficiency ratio
    data['range_efficiency'] = (data['close'] - data['close'].shift(1)).abs() / (data['high'] - data['low'])
    data['range_efficiency'] = data['range_efficiency'].replace([np.inf, -np.inf], np.nan)
    
    # Breakout detection using fractal efficiency
    data['fractal_20d_high'] = data['fractal_eff_5'].rolling(window=20).max()
    data['fractal_20d_low'] = data['fractal_eff_5'].rolling(window=20).min()
    data['breakout_strength'] = np.where(
        data['fractal_eff_5'] > data['fractal_20d_high'].shift(1),
        (data['fractal_eff_5'] - data['fractal_20d_high'].shift(1)) / data['fractal_20d_high'].shift(1),
        np.where(
            data['fractal_eff_5'] < data['fractal_20d_low'].shift(1),
            (data['fractal_eff_5'] - data['fractal_20d_low'].shift(1)) / data['fractal_20d_low'].shift(1),
            0
        )
    )
    
    # 5. Multi-Dimensional Efficiency Correlation Enhancement
    # Fractal-volume efficiency correlation
    data['fractal_vol_corr_5'] = data['fractal_eff_5'].rolling(window=5).corr(data['volume'])
    data['fractal_vol_corr_20'] = data['fractal_eff_5'].rolling(window=20).corr(data['volume'])
    data['corr_momentum'] = data['fractal_vol_corr_5'].diff(3)
    
    # 6. Regime-Adaptive Efficiency Signal Generation
    # Efficiency path curvature for regime detection
    data['efficiency_curvature'] = data['fractal_eff_5'].diff(3).diff(3)
    
    # Efficiency volatility for regime detection
    data['efficiency_volatility'] = data['fractal_eff_5'].rolling(window=10).std()
    
    # 7. Final Composite Factor Construction
    # Base fractal efficiency convergence component
    fractal_component = (
        data['fractal_divergence'] * 
        data['fractal_eff_accel'] * 
        data['convergence_strength']
    )
    
    # Volume-amplitude confirmation component
    volume_component = (
        data['volume_slope_5'] * 
        data['vol_amp_corr'] * 
        data['amount_roc_5']
    )
    
    # Liquidity acceleration component
    liquidity_component = (
        data['volume_accel'] * 
        data['dollar_vol_velocity'] * 
        data['efficiency_flow']
    )
    
    # Breakout enhancement component
    breakout_component = (
        data['breakout_strength'] * 
        data['range_efficiency'] * 
        data['corr_momentum']
    )
    
    # Combine all components with adaptive scaling
    factor = (
        fractal_component * 0.35 +
        volume_component * 0.25 +
        liquidity_component * 0.20 +
        breakout_component * 0.20
    )
    
    # Apply regime-based scaling
    trending_regime = data['efficiency_curvature'].abs() > data['efficiency_curvature'].rolling(window=20).std()
    high_vol_regime = data['efficiency_volatility'] > data['efficiency_volatility'].rolling(window=20).quantile(0.7)
    
    # Scale down in high volatility regimes
    factor = np.where(high_vol_regime, factor * 0.7, factor)
    
    # Scale up in trending regimes with strong convergence
    strong_convergence = data['convergence_strength'] > data['convergence_strength'].rolling(window=20).quantile(0.6)
    factor = np.where(trending_regime & strong_convergence, factor * 1.2, factor)
    
    # Final normalization
    factor_series = pd.Series(factor, index=data.index)
    factor_series = (factor_series - factor_series.rolling(window=20).mean()) / factor_series.rolling(window=20).std()
    
    return factor_series
