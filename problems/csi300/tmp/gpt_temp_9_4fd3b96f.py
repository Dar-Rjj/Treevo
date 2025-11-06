import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Copy dataframe to avoid modifying original
    data = df.copy()
    
    # Calculate ATR for compression dynamics
    high_low = data['high'] - data['low']
    high_close_prev = np.abs(data['high'] - data['close'].shift(1))
    low_close_prev = np.abs(data['low'] - data['close'].shift(1))
    true_range = np.maximum(np.maximum(high_low, high_close_prev), low_close_prev)
    atr_5 = true_range.rolling(window=5).mean()
    
    # Multi-scale Compression Dynamics
    range_compression = high_low / (atr_5 + 1e-8)
    
    # Compression intensity across different periods
    compression_3 = range_compression.rolling(window=3).mean()
    compression_5 = range_compression.rolling(window=5).mean()
    compression_8 = range_compression.rolling(window=8).mean()
    
    # Fractal compression patterns using local minima/maxima
    compression_min_5 = range_compression.rolling(window=5, center=True).min()
    compression_max_5 = range_compression.rolling(window=5, center=True).max()
    compression_fractal = (range_compression - compression_min_5) / (compression_max_5 - compression_min_5 + 1e-8)
    
    # Asymmetric Breakout Efficiency
    close_prev = data['close'].shift(1)
    upside_breakout = (data['high'] - data['high'].shift(1)) / (data['high'].shift(1) + 1e-8)
    downside_breakout = (data['low'].shift(1) - data['low']) / (data['low'].shift(1) + 1e-8)
    
    # Breakout momentum persistence
    upside_persistence = upside_breakout.rolling(window=3).apply(lambda x: np.sum(x > 0) / len(x))
    downside_persistence = downside_breakout.rolling(window=3).apply(lambda x: np.sum(x < 0) / len(x))
    
    # Volatility-adaptive breakout strength
    vol_breakout = data['close'].pct_change().rolling(window=5).std()
    upside_strength = upside_breakout / (vol_breakout + 1e-8)
    downside_strength = downside_breakout / (vol_breakout + 1e-8)
    
    # Efficiency differentials
    breakout_efficiency_diff = upside_persistence - downside_persistence
    
    # Volume-Price Divergence Confirmation
    volume_compression = data['volume'].rolling(window=5).std() / (data['volume'].rolling(window=5).mean() + 1e-8)
    price_compression = range_compression.rolling(window=5).std() / (range_compression.rolling(window=5).mean() + 1e-8)
    
    # Volume compression vs price compression divergence
    compression_divergence = volume_compression - price_compression
    
    # Asymmetric volume confirmation during breakouts
    volume_breakout_ratio = data['volume'] / data['volume'].rolling(window=5).mean()
    upside_volume_conf = (upside_breakout > 0) * volume_breakout_ratio
    downside_volume_conf = (downside_breakout < 0) * volume_breakout_ratio
    volume_conf_asymmetry = upside_volume_conf - downside_volume_conf
    
    # Bidirectional Pressure Dynamics
    hl_range = data['high'] - data['low']
    hl_range = np.where(hl_range == 0, 1e-8, hl_range)
    
    buying_pressure = data['volume'] * (data['close'] - data['low']) / hl_range
    selling_pressure = data['volume'] * (data['high'] - data['close']) / hl_range
    
    # Net pressure asymmetry
    net_pressure = (buying_pressure - selling_pressure) / (buying_pressure + selling_pressure + 1e-8)
    
    # Regime Transition Detection
    cumulative_pressure = net_pressure.rolling(window=5).sum()
    pressure_regime_break = cumulative_pressure.diff().abs()
    
    # Pressure autocorrelation for regime persistence
    pressure_autocorr = net_pressure.rolling(window=5).apply(lambda x: x.autocorr(lag=1) if len(x) > 1 else 0)
    
    # Final alpha factor combining all components
    alpha = (
        compression_fractal * 0.15 +
        breakout_efficiency_diff * 0.20 +
        compression_divergence * 0.15 +
        volume_conf_asymmetry * 0.15 +
        net_pressure * 0.20 +
        pressure_regime_break * 0.10 +
        pressure_autocorr * 0.05
    )
    
    return alpha
