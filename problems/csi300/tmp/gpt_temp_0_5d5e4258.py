import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Hierarchical Entropy-Momentum Synthesis Alpha Factor
    """
    data = df.copy()
    
    # 1. Multi-Scale Entropy-Momentum Integration
    # Price change entropy over multiple horizons
    for window in [5, 10, 20]:
        data[f'return_{window}d'] = data['close'].pct_change(window)
    
    # Calculate entropy of price changes
    def calculate_entropy(series, window=10):
        returns = series.pct_change().dropna()
        hist, _ = np.histogram(returns.rolling(window).mean().dropna(), bins=20, density=True)
        hist = hist[hist > 0]
        return -np.sum(hist * np.log(hist))
    
    data['entropy_5d'] = data['close'].rolling(5).apply(lambda x: calculate_entropy(x, 5), raw=False)
    data['entropy_10d'] = data['close'].rolling(10).apply(lambda x: calculate_entropy(x, 10), raw=False)
    data['entropy_20d'] = data['close'].rolling(20).apply(lambda x: calculate_entropy(x, 20), raw=False)
    
    # Momentum acceleration
    data['mom_accel_5_10'] = data['return_5d'] - data['return_10d']
    data['mom_accel_10_20'] = data['return_10d'] - data['return_20d']
    
    # Entropy-weighted acceleration
    data['entropy_weighted_accel'] = (
        data['mom_accel_5_10'] * data['entropy_5d'] + 
        data['mom_accel_10_20'] * data['entropy_10d']
    ) / (data['entropy_5d'] + data['entropy_10d'] + 1e-8)
    
    # Volume entropy and momentum
    data['volume_ratio_5d'] = data['volume'] / data['volume'].rolling(5).mean()
    data['volume_momentum'] = data['volume_ratio_5d'].pct_change(5)
    
    # 2. Fractal Efficiency Reversal System
    # True range calculation
    data['prev_close'] = data['close'].shift(1)
    data['true_range'] = np.maximum(
        data['high'] - data['low'],
        np.maximum(
            abs(data['high'] - data['prev_close']),
            abs(data['low'] - data['prev_close'])
        )
    )
    
    # Efficiency ratio
    data['efficiency_ratio'] = abs(data['close'] - data['prev_close']) / data['true_range']
    data['efficiency_trend'] = data['efficiency_ratio'].rolling(3).mean() - data['efficiency_ratio'].rolling(10).mean()
    
    # Return reversal signals
    data['return_1d'] = data['close'].pct_change(1)
    data['return_3d'] = data['close'].pct_change(3)
    data['reversal_signal'] = -data['return_1d'] * data['return_3d']
    
    # 3. Microstructure Pressure Momentum
    # Spread proxy and intraday pressure
    data['spread_proxy'] = (data['high'] - data['low']) / data['close']
    data['intraday_pressure'] = ((data['close'] - data['low']) / (data['high'] - data['low'] + 1e-8)) * data['volume']
    data['pressure_momentum'] = data['intraday_pressure'].rolling(5).sum().pct_change(5)
    
    # Momentum acceleration
    data['momentum_1d'] = data['close'].pct_change(1)
    data['momentum_accel'] = data['momentum_1d'] - data['momentum_1d'].shift(1)
    
    # 4. Volatility-Entropy Skewness Factor
    # Return skewness
    data['return_skewness'] = data['close'].pct_change(1).rolling(20).skew()
    
    # Range efficiency
    data['range_utilization'] = abs(data['close'] - data['open']) / (data['high'] - data['low'] + 1e-8)
    
    # 5. Dynamic Volume-Surge Entropy Divergence
    data['volume_surge_1d'] = data['volume'] / data['volume'].shift(1)
    data['volume_surge_5d'] = data['volume'] / data['volume'].rolling(5).mean()
    
    # Momentum divergence
    data['momentum_divergence'] = (
        data['return_5d'] - data['return_10d'] + 
        data['return_10d'] - data['return_20d']
    )
    
    # Final alpha factor synthesis
    # Regime classification based on entropy
    high_entropy_regime = data['entropy_10d'] > data['entropy_10d'].rolling(20).quantile(0.7)
    low_entropy_regime = data['entropy_10d'] < data['entropy_10d'].rolling(20).quantile(0.3)
    
    # Core factor components
    entropy_momentum = data['entropy_weighted_accel'] * data['volume_momentum']
    fractal_efficiency = data['reversal_signal'] * (1 - data['efficiency_ratio'])
    microstructure = data['pressure_momentum'] * data['momentum_accel']
    volatility_skew = data['return_skewness'] * (1 - data['range_utilization'])
    volume_divergence = data['volume_surge_5d'] * data['momentum_divergence'] * data['entropy_10d']
    
    # Regime-adaptive weighting
    alpha_factor = pd.Series(index=data.index, dtype=float)
    
    # High entropy: emphasize volume confirmation and divergence
    alpha_factor[high_entropy_regime] = (
        0.3 * entropy_momentum +
        0.2 * fractal_efficiency +
        0.25 * microstructure +
        0.15 * volatility_skew +
        0.1 * volume_divergence
    )[high_entropy_regime]
    
    # Low entropy: pure momentum acceleration focus
    alpha_factor[low_entropy_regime] = (
        0.4 * entropy_momentum +
        0.3 * fractal_efficiency +
        0.2 * microstructure +
        0.1 * volatility_skew
    )[low_entropy_regime]
    
    # Transition regimes (medium entropy)
    transition_regime = ~high_entropy_regime & ~low_entropy_regime
    alpha_factor[transition_regime] = (
        0.25 * entropy_momentum +
        0.25 * fractal_efficiency +
        0.2 * microstructure +
        0.15 * volatility_skew +
        0.15 * volume_divergence
    )[transition_regime]
    
    # Risk adjustment based on entropy volatility
    entropy_vol = data['entropy_10d'].rolling(10).std()
    risk_adjustment = 1 / (1 + entropy_vol)
    alpha_factor = alpha_factor * risk_adjustment
    
    # Normalize and clean
    alpha_factor = alpha_factor.replace([np.inf, -np.inf], np.nan)
    alpha_factor = (alpha_factor - alpha_factor.rolling(20).mean()) / (alpha_factor.rolling(20).std() + 1e-8)
    
    return alpha_factor
