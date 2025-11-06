import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Price-Volume Fractal Efficiency
    df['hlc_avg'] = (df['high'] + df['low'] + df['close']) / 3
    df['price_path_length'] = (df['high'] - df['low']).abs()
    df['net_price_move'] = (df['hlc_avg'] - df['hlc_avg'].shift(1)).abs()
    df['fractal_efficiency'] = df['net_price_move'] / (df['price_path_length'] + 1e-8)
    df['volume_efficiency'] = df['fractal_efficiency'] / (df['volume'] + 1e-8)
    
    # Amplitude-Frequency Price Oscillation
    df['price_amplitude'] = (df['high'] - df['low']) / df['close']
    df['price_oscillation'] = (df['close'] - df['open']) / (df['high'] - df['low'] + 1e-8)
    
    # Calculate rolling amplitude and frequency characteristics
    df['amplitude_ma'] = df['price_amplitude'].rolling(window=10, min_periods=5).mean()
    df['amplitude_std'] = df['price_amplitude'].rolling(window=10, min_periods=5).std()
    df['oscillation_freq'] = (df['price_oscillation'].abs() > 0.5).rolling(window=10, min_periods=5).sum()
    
    # Microstructure Noise Ratio
    df['intraday_noise'] = ((df['high'] - df['low']) - (df['close'] - df['open']).abs()) / (df['high'] - df['low'] + 1e-8)
    df['signal_strength'] = (df['close'] - df['open']).abs() / (df['high'] - df['low'] + 1e-8)
    df['noise_ratio'] = df['intraday_noise'] / (df['signal_strength'] + 1e-8)
    
    # Volume Asymmetry Momentum
    df['up_volume'] = np.where(df['close'] > df['open'], df['volume'], 0)
    df['down_volume'] = np.where(df['close'] < df['open'], df['volume'], 0)
    df['volume_ratio'] = (df['up_volume'].rolling(window=5, min_periods=3).sum() + 1e-8) / \
                        (df['down_volume'].rolling(window=5, min_periods=3).sum() + 1e-8)
    df['volume_momentum'] = df['volume_ratio'] - df['volume_ratio'].shift(5)
    
    # Price Level Memory Effect
    df['price_level'] = (df['close'] // (df['close'].rolling(window=20, min_periods=10).std() * 0.1)) * (df['close'].rolling(window=20, min_periods=10).std() * 0.1)
    df['price_level_count'] = df.groupby('price_level')['price_level'].transform('count')
    df['memory_effect'] = df['price_level_count'] / df['price_level_count'].rolling(window=20, min_periods=10).mean()
    
    # Volatility Compression Expansion Cycle
    df['volatility'] = df['close'].pct_change().rolling(window=10, min_periods=5).std()
    df['volatility_ratio'] = df['volatility'] / df['volatility'].rolling(window=20, min_periods=10).mean()
    df['compression_signal'] = np.where(df['volatility_ratio'] < 0.7, 1, 0)
    df['position_vs_range'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-8)
    
    # Liquidity Gradient Factor
    df['price_tick'] = (df['high'] - df['low']) / (df['volume'] + 1e-8)
    df['liquidity_depth'] = df['amount'] / (df['high'] - df['low'] + 1e-8)
    df['liquidity_gradient'] = df['liquidity_depth'].pct_change()
    
    # Temporal Price Correlation Decay
    df['returns'] = df['close'].pct_change()
    autocorr_1 = df['returns'].rolling(window=20, min_periods=10).apply(lambda x: x.autocorr(lag=1), raw=False)
    autocorr_5 = df['returns'].rolling(window=20, min_periods=10).apply(lambda x: x.autocorr(lag=5), raw=False)
    df['correlation_decay'] = autocorr_1 - autocorr_5
    
    # Volume Spike Return Asymmetry
    df['volume_zscore'] = (df['volume'] - df['volume'].rolling(window=20, min_periods=10).mean()) / \
                         (df['volume'].rolling(window=20, min_periods=10).std() + 1e-8)
    df['volume_spike'] = np.where(df['volume_zscore'] > 2, 1, 0)
    df['spike_return'] = df['volume_spike'] * df['returns']
    df['spike_asymmetry'] = df['spike_return'].rolling(window=10, min_periods=5).mean()
    
    # Price Path Curvature Factor
    df['price_change_1'] = df['close'].pct_change(1)
    df['price_change_2'] = df['close'].pct_change(2)
    df['price_change_3'] = df['close'].pct_change(3)
    df['curvature'] = (df['price_change_1'] - 2*df['price_change_2'] + df['price_change_3']) / \
                     (1 + df['price_change_1'].abs() + df['price_change_2'].abs() + df['price_change_3'].abs())
    
    # Combine factors with appropriate weights
    factor = (
        0.15 * df['volume_efficiency'] +
        0.12 * df['oscillation_freq'] +
        0.10 * (1 - df['noise_ratio']) +
        0.13 * df['volume_momentum'] +
        0.11 * df['memory_effect'] +
        0.12 * df['compression_signal'] * df['position_vs_range'] +
        0.09 * df['liquidity_gradient'] +
        0.08 * df['correlation_decay'] +
        0.06 * df['spike_asymmetry'] +
        0.04 * df['curvature']
    )
    
    return factor
