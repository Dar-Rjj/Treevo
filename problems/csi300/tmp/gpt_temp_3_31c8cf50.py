import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Momentum-Decay Asymmetry with Volume-Price Fractal Coherence alpha factor
    """
    # Copy data to avoid modifying original
    data = df.copy()
    
    # Multi-Horizon Momentum Decay Patterns
    # Price momentum decay acceleration (5,10,20-day horizons)
    for window in [5, 10, 20]:
        data[f'momentum_{window}'] = data['close'].pct_change(window)
        data[f'momentum_decay_{window}'] = data[f'momentum_{window}'] - data[f'momentum_{window}'].shift(1)
    
    # 2nd derivative of momentum autocorrelation decay
    momentum_5 = data['momentum_5']
    autocorr_1 = momentum_5.rolling(10).corr(momentum_5.shift(1))
    autocorr_2 = momentum_5.rolling(10).corr(momentum_5.shift(2))
    data['momentum_decay_acceleration'] = (autocorr_1 - autocorr_2) - (autocorr_1.shift(1) - autocorr_2.shift(1))
    
    # Volume momentum persistence asymmetry
    volume_momentum = data['volume'].pct_change(5)
    volume_halflife = volume_momentum.rolling(10).apply(
        lambda x: np.log(0.5) / np.log(np.abs(x[-1] / x[0])) if x[0] != 0 and len(x) > 1 else 1
    )
    data['volume_persistence_asymmetry'] = volume_halflife - volume_halflife.shift(5)
    
    # Return reversal timing divergence
    returns = data['close'].pct_change()
    reversal_timing_5 = returns.rolling(5).apply(lambda x: len([i for i in range(1, len(x)) if x[i] * x[i-1] < 0]))
    reversal_timing_10 = returns.rolling(10).apply(lambda x: len([i for i in range(1, len(x)) if x[i] * x[i-1] < 0]))
    data['reversal_timing_divergence'] = (reversal_timing_10 / 10) - (reversal_timing_5 / 5)
    
    # Decay Asymmetry Measurement
    momentum_aligned = (
        (data['momentum_5'] > 0) & (data['momentum_decay_5'] < 0) |
        (data['momentum_5'] < 0) & (data['momentum_decay_5'] > 0)
    )
    decay_aligned = (
        (data['momentum_5'] > 0) & (data['momentum_decay_5'] > 0) |
        (data['momentum_5'] < 0) & (data['momentum_decay_5'] < 0)
    )
    data['momentum_decay_alignment_ratio'] = (
        momentum_aligned.rolling(10).sum() / decay_aligned.rolling(10).sum()
    ).replace([np.inf, -np.inf], 0).fillna(0)
    
    # Volume-Price Fractal Coherence
    # Multi-scale volume-price correlation persistence
    windows = [3, 5, 8]
    vp_correlations = []
    for window in windows:
        corr = data['volume'].rolling(window).corr(data['close'])
        vp_correlations.append(corr.rolling(5).std())  # Persistence measure
    
    data['fractal_coherence_persistence'] = pd.concat(vp_correlations, axis=1).mean(axis=1)
    
    # Hurst-like estimation for price and volume series
    def hurst_approximation(series, window=20):
        lags = [2, 5, 10]
        tau = []
        for lag in lags:
            if len(series) >= lag:
                ts = series[-window:]
                if len(ts) >= lag:
                    rs = ts.rolling(lag).std() / ts.rolling(lag).mean()
                    tau.append(np.log(rs.mean()) if not np.isnan(rs.mean()) else 0)
        if len(tau) > 1:
            return np.polyfit(np.log(lags[:len(tau)]), tau, 1)[0]
        return 0
    
    data['price_hurst'] = data['close'].rolling(20).apply(
        lambda x: hurst_approximation(x), raw=False
    )
    data['volume_hurst'] = data['volume'].rolling(20).apply(
        lambda x: hurst_approximation(x), raw=False
    )
    data['hurst_divergence'] = data['price_hurst'] - data['volume_hurst']
    
    # Fractal dimension approximation via volatility scaling
    def fractal_dimension_approx(series, window=15):
        scales = [1, 2, 4]
        fluctuations = []
        for scale in scales:
            if len(series) >= scale:
                scaled_vol = series.rolling(scale).std().dropna()
                if len(scaled_vol) > 0:
                    fluctuations.append(scaled_vol.iloc[-1])
        if len(fluctuations) > 1:
            return np.polyfit(np.log(scales[:len(fluctuations)]), np.log(fluctuations), 1)[0]
        return 1.5
    
    data['fractal_dimension'] = data['close'].rolling(15).apply(
        lambda x: fractal_dimension_approx(x), raw=False
    )
    
    # Coherence Regime Identification
    vp_corr_rolling = data['volume'].rolling(10).corr(data['close'])
    coherence_threshold = vp_corr_rolling.rolling(20).quantile(0.7)
    data['high_coherence_regime'] = (vp_corr_rolling > coherence_threshold).astype(int)
    
    # Regime transition detection
    data['coherence_regime_change'] = data['high_coherence_regime'].diff()
    
    # Scale-Adaptive Integration Framework
    # Volatility regime detection
    volatility = data['close'].pct_change().rolling(20).std()
    high_vol_regime = (volatility > volatility.rolling(50).quantile(0.7)).astype(int)
    
    # Cross-regime coherence comparison
    high_vol_coherence = data['fractal_coherence_persistence'][high_vol_regime == 1]
    low_vol_coherence = data['fractal_coherence_persistence'][high_vol_regime == 0]
    
    if len(high_vol_coherence) > 0 and len(low_vol_coherence) > 0:
        coherence_regime_ratio = high_vol_coherence.rolling(10).mean() / low_vol_coherence.rolling(10).mean()
    else:
        coherence_regime_ratio = pd.Series(1, index=data.index)
    
    # Momentum persistence acceleration patterns
    momentum_accel_5 = data['momentum_decay_5'].diff()
    momentum_accel_10 = data['momentum_decay_10'].diff()
    data['momentum_persistence_acceleration'] = (momentum_accel_5.rolling(5).mean() + 
                                               momentum_accel_10.rolling(5).mean()) / 2
    
    # Scale-Adaptive Signal Amplification
    regime_weight = np.where(data['high_coherence_regime'] == 1, 1.5, 1.0)
    volatility_weight = np.where(high_vol_regime == 1, 0.8, 1.2)
    
    # Composite Alpha Generation
    # Core components
    decay_asymmetry = data['momentum_decay_alignment_ratio'] * data['momentum_decay_acceleration']
    fractal_coherence = data['fractal_coherence_persistence'] * data['hurst_divergence']
    
    # Cross-scale integration
    momentum_coherence_interaction = decay_asymmetry * fractal_coherence
    
    # Multi-scale signal synthesis
    lead_lag_pattern = (data['momentum_5'].rolling(3).corr(data['momentum_10'].shift(2)) + 
                       data['momentum_10'].rolling(3).corr(data['momentum_20'].shift(3))) / 2
    
    # Final composite factor
    composite_factor = (
        momentum_coherence_interaction * 
        lead_lag_pattern * 
        regime_weight * 
        volatility_weight * 
        coherence_regime_ratio.reindex(data.index, fill_value=1)
    )
    
    # Clean and normalize
    composite_factor = composite_factor.replace([np.inf, -np.inf], np.nan)
    composite_factor = (composite_factor - composite_factor.rolling(50).mean()) / composite_factor.rolling(50).std()
    
    return composite_factor
