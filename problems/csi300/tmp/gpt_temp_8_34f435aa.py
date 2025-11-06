import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Regime-Adaptive Momentum Divergence factor
    Combines momentum, volume divergence, and volatility regime detection
    """
    data = df.copy()
    
    # Market Regime Identification
    # Calculate True Range for volatility estimation
    data['prev_close'] = data['close'].shift(1)
    data['tr1'] = data['high'] - data['low']
    data['tr2'] = abs(data['high'] - data['prev_close'])
    data['tr3'] = abs(data['low'] - data['prev_close'])
    data['true_range'] = data[['tr1', 'tr2', 'tr3']].max(axis=1)
    
    # Rolling volatility using True Range
    data['volatility_20'] = data['true_range'].rolling(window=20).mean()
    data['volatility_ratio'] = data['volatility_20'] / data['volatility_20'].rolling(window=60).mean()
    
    # Price fractal dimension approximation using Hurst exponent-like calculation
    data['log_ret'] = np.log(data['close'] / data['close'].shift(1))
    data['hurst_like'] = np.log(data['log_ret'].rolling(window=20).std()) / np.log(20)
    data['trend_strength'] = 1 - abs(data['hurst_like'] - 0.5) * 2
    
    # Momentum Component
    # Multiple time-window returns with exponential decay
    for window in [5, 10, 20]:
        data[f'momentum_{window}'] = data['close'] / data['close'].shift(window) - 1
    
    # Exponential decay weighted momentum (recent periods have higher weight)
    weights_5 = np.exp(-np.arange(5)/2.5)
    weights_5 = weights_5 / weights_5.sum()
    data['momentum_exp_5'] = data['close'].rolling(window=5).apply(
        lambda x: np.sum(weights_5 * (x / x[0] - 1)) if len(x) == 5 else np.nan
    )
    
    # Momentum acceleration
    data['momentum_accel'] = data['momentum_exp_5'] - data['momentum_exp_5'].shift(5)
    
    # Divergence Detection
    # Price trend vs volume trend correlation
    data['price_trend'] = data['close'].rolling(window=10).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 10 else np.nan
    )
    data['volume_trend'] = data['volume'].rolling(window=10).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 10 else np.nan
    )
    
    # Rolling correlation between price and volume trends
    data['price_volume_corr'] = data['price_trend'].rolling(window=20).corr(data['volume_trend'])
    
    # Divergence strength
    data['divergence_strength'] = -data['price_volume_corr'] * data['momentum_10']
    
    # Volume Analysis
    # Volume clusters using rolling percentiles
    data['volume_percentile_20'] = data['volume'].rolling(window=20).apply(
        lambda x: (x[-1] > np.percentile(x, 80)) if len(x) == 20 else np.nan
    )
    
    # Volume-to-amount ratio for liquidity assessment
    data['volume_amount_ratio'] = data['volume'] / data['amount']
    data['liquidity_score'] = data['volume_amount_ratio'].rolling(window=10).apply(
        lambda x: x[-1] / np.mean(x) if len(x) == 10 else np.nan
    )
    
    # Volatility Adjustment
    # Upside vs downside volatility
    data['upside_vol'] = data['log_ret'].rolling(window=20).apply(
        lambda x: np.std(x[x > 0]) if len(x[x > 0]) > 5 else 0
    )
    data['downside_vol'] = data['log_ret'].rolling(window=20).apply(
        lambda x: np.std(x[x < 0]) if len(x[x < 0]) > 5 else 0
    )
    data['vol_asymmetry'] = data['upside_vol'] / (data['downside_vol'] + 1e-8)
    
    # Volatility regime adjustment factor
    data['vol_regime'] = np.where(data['volatility_ratio'] > 1.2, 1, 
                                 np.where(data['volatility_ratio'] < 0.8, -1, 0))
    
    # Adaptive Signal Generation
    # High volatility regime: mean reversion with volume confirmation
    high_vol_signal = (
        -data['momentum_5'] * 0.6 +  # Mean reversion component
        data['divergence_strength'] * 0.3 +  # Divergence component
        data['volume_percentile_20'] * 0.1   # Volume confirmation
    )
    
    # Low volatility regime: momentum persistence with trend strength
    low_vol_signal = (
        data['momentum_exp_5'] * 0.5 +  # Momentum component
        data['momentum_accel'] * 0.3 +   # Acceleration component
        data['trend_strength'] * 0.2     # Trend confirmation
    )
    
    # Regime-specific signal combination with smooth transitions
    vol_weight = 1 / (1 + np.exp(-3 * (data['volatility_ratio'] - 1)))
    
    # Final factor calculation
    data['factor'] = (
        vol_weight * high_vol_signal + 
        (1 - vol_weight) * low_vol_signal
    ) * (1 + 0.2 * data['liquidity_score'])
    
    # Clean up intermediate columns
    result = data['factor'].copy()
    
    return result
