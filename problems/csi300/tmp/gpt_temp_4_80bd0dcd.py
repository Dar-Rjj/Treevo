import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Momentum-Decay Volume Asymmetry Factor
    Predicts future returns based on asymmetric relationships between price momentum,
    volume patterns, and momentum decay characteristics.
    """
    data = df.copy()
    
    # Calculate Directional Price Momentum Components
    data['prev_close'] = data['close'].shift(1)
    
    # Upward momentum components
    data['up_mom_open'] = (data['high'] - data['open']) / data['open']
    data['up_mom_prev_close'] = (data['high'] - data['prev_close']) / data['prev_close']
    
    # Downward momentum components  
    data['down_mom_open'] = (data['open'] - data['low']) / data['open']
    data['down_mom_prev_close'] = (data['prev_close'] - data['low']) / data['prev_close']
    
    # Net Intraday Momentum Bias
    avg_price = (data['open'] + data['high'] + data['low'] + data['close']) / 4
    data['net_momentum'] = (
        (data['up_mom_open'] + data['up_mom_prev_close']) - 
        (data['down_mom_open'] + data['down_mom_prev_close'])
    ) / avg_price
    
    # Analyze Volume Distribution Patterns
    data['vol_5d_median'] = data['volume'].rolling(window=5, min_periods=3).median()
    data['vol_20d_median'] = data['volume'].rolling(window=20, min_periods=10).median()
    
    data['vol_concentration_5d'] = data['volume'] / data['vol_5d_median']
    data['vol_concentration_20d'] = data['volume'] / data['vol_20d_median']
    data['vol_ratio_5d_20d'] = data['vol_5d_median'] / data['vol_20d_median']
    
    # Volume-Momentum Asymmetry
    data['volume_efficiency'] = abs(data['close'] - data['open']) / data['volume'].replace(0, 1)
    
    # Model Momentum Decay Characteristics
    # 3-day autocorrelation of intraday returns
    intraday_returns = (data['close'] - data['open']) / data['open']
    data['momentum_autocorr_3d'] = intraday_returns.rolling(window=3, min_periods=2).apply(
        lambda x: x.autocorr() if len(x) > 1 and not np.isnan(x.autocorr()) else 0, raw=False
    )
    
    # 5-day momentum decay rate (exponential decay fit)
    def calculate_decay_rate(window):
        if len(window) < 3:
            return 0
        try:
            log_vals = np.log(np.abs(window) + 1e-8)
            x = np.arange(len(window))
            slope = np.polyfit(x, log_vals, 1)[0]
            return -slope
        except:
            return 0
    
    data['momentum_decay_5d'] = intraday_returns.rolling(window=5, min_periods=3).apply(
        calculate_decay_rate, raw=False
    )
    
    # Momentum half-life estimation
    data['momentum_half_life'] = np.log(2) / (data['momentum_decay_5d'].replace(0, 1e-8) + 1e-8)
    data['momentum_half_life'] = data['momentum_half_life'].clip(upper=10)
    
    # Compute Momentum Quality Score
    momentum_magnitude = abs(intraday_returns)
    data['momentum_quality'] = (
        momentum_magnitude * 
        (1 - data['momentum_decay_5d'].clip(lower=0)) * 
        data['momentum_autocorr_3d'].clip(lower=0)
    )
    
    # Volume-adjusted persistence metric
    data['volume_adjusted_persistence'] = (
        data['momentum_quality'] / 
        (data['vol_concentration_20d'].replace(0, 1) + 1e-8)
    )
    
    # Construct Asymmetry Detection Framework
    # Price-Volume Asymmetry Components
    data['bullish_asymmetry'] = (
        data['net_momentum'].clip(lower=0) * 
        (1 / (data['vol_concentration_20d'] + 1e-8))
    )
    
    data['bearish_asymmetry'] = (
        data['net_momentum'].clip(upper=0).abs() * 
        data['vol_concentration_20d']
    )
    
    data['neutral_asymmetry'] = 1 - abs(data['bullish_asymmetry'] - data['bearish_asymmetry'])
    
    # Multi-Timeframe Asymmetry Convergence
    data['short_term_asymmetry'] = (
        data['bullish_asymmetry'].rolling(window=3, min_periods=2).mean() - 
        data['bearish_asymmetry'].rolling(window=3, min_periods=2).mean()
    )
    
    data['medium_term_asymmetry'] = (
        data['bullish_asymmetry'].rolling(window=10, min_periods=5).mean() - 
        data['bearish_asymmetry'].rolling(window=10, min_periods=5).mean()
    )
    
    data['asymmetry_convergence'] = (
        data['short_term_asymmetry'] * data['medium_term_asymmetry']
    )
    
    # Generate Predictive Alpha Signals
    # Core Asymmetry Factor
    data['core_asymmetry'] = (
        data['momentum_quality'] * 
        data['vol_concentration_5d'] * 
        (1 - data['momentum_decay_5d'].clip(lower=0)) * 
        data['net_momentum']
    )
    
    # Apply Regime-Based Enhancements
    # High Persistence Regime
    high_persistence_mask = data['momentum_half_life'] > 2
    data['high_persistence_signal'] = np.where(
        high_persistence_mask,
        data['core_asymmetry'] * (1 / (data['vol_concentration_20d'] + 1e-8)),
        0
    )
    
    # Decay Acceleration Regime
    decay_accel_mask = data['momentum_decay_5d'] > data['momentum_decay_5d'].rolling(window=10, min_periods=5).quantile(0.7)
    data['decay_accel_signal'] = np.where(
        decay_accel_mask,
        -data['core_asymmetry'] * data['vol_concentration_20d'],
        0
    )
    
    # Neutral Transition Regime
    neutral_mask = (
        (data['neutral_asymmetry'] > data['neutral_asymmetry'].rolling(window=20, min_periods=10).quantile(0.6)) &
        ~high_persistence_mask & ~decay_accel_mask
    )
    data['neutral_signal'] = np.where(
        neutral_mask,
        data['asymmetry_convergence'] * data['momentum_quality'],
        0
    )
    
    # Final Alpha Factor Construction
    data['raw_asymmetry_score'] = (
        data['core_asymmetry'] + 
        data['high_persistence_signal'] + 
        data['decay_accel_signal'] + 
        data['neutral_signal']
    )
    
    # Timeframe convergence weighting
    timeframe_weight = (
        data['short_term_asymmetry'].abs() * 0.3 + 
        data['medium_term_asymmetry'].abs() * 0.7
    )
    
    # Regime-adaptive signal scaling
    regime_weights = pd.Series(1.0, index=data.index)
    regime_weights[high_persistence_mask] = 1.2
    regime_weights[decay_accel_mask] = 0.8
    regime_weights[neutral_mask] = 1.0
    
    # Final factor
    alpha_factor = (
        data['raw_asymmetry_score'] * 
        timeframe_weight * 
        regime_weights
    )
    
    # Normalize and clean
    alpha_factor = alpha_factor.replace([np.inf, -np.inf], np.nan)
    alpha_factor = alpha_factor.fillna(method='ffill').fillna(0)
    
    return alpha_factor
