import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Generate novel alpha factors combining multiple market microstructure concepts
    """
    # Make copy to avoid modifying original data
    data = df.copy()
    
    # 1. Price-Volume Fractal Efficiency
    # Compute fractal dimension using high-low range
    high_low_range = data['high'] - data['low']
    fractal_window = 20
    
    # Calculate Hurst exponent approximation using rescaled range
    def hurst_approximation(series, window):
        hurst_values = []
        for i in range(len(series)):
            if i < window:
                hurst_values.append(np.nan)
                continue
                
            window_data = series.iloc[i-window:i]
            mean_val = window_data.mean()
            deviations = window_data - mean_val
            cumulative_dev = deviations.cumsum()
            range_val = cumulative_dev.max() - cumulative_dev.min()
            std_val = window_data.std()
            
            if std_val > 0:
                hurst = np.log(range_val / std_val) / np.log(window)
            else:
                hurst = np.nan
            hurst_values.append(hurst)
        
        return pd.Series(hurst_values, index=series.index)
    
    hurst_factor = hurst_approximation(data['close'], fractal_window)
    
    # Volume flow efficiency
    volume_ma = data['volume'].rolling(window=10).mean()
    price_change = data['close'].pct_change()
    volume_efficiency = (price_change.abs() * data['volume']) / (volume_ma + 1e-8)
    volume_efficiency_ma = volume_efficiency.rolling(window=10).mean()
    
    # 2. Order Imbalance Momentum
    # Calculate bid-ask pressure from OHLC
    typical_price = (data['high'] + data['low'] + data['close']) / 3
    close_position = (data['close'] - data['low']) / (data['high'] - data['low'] + 1e-8)
    order_imbalance = (2 * data['close'] - data['high'] - data['low']) / (data['high'] - data['low'] + 1e-8)
    
    # Cumulative order flow
    signed_volume = order_imbalance * data['volume']
    cumulative_order_flow = signed_volume.rolling(window=15).sum()
    
    # 3. Regime-Switching Mean Reversion
    # Detect mean-reversion periods using price clustering
    price_ma_short = data['close'].rolling(window=5).mean()
    price_ma_long = data['close'].rolling(window=20).mean()
    price_deviation = (data['close'] - price_ma_long) / price_ma_long
    
    # Volume confirmation for regime detection
    volume_zscore = (data['volume'] - data['volume'].rolling(window=20).mean()) / data['volume'].rolling(window=20).std()
    regime_strength = np.abs(price_deviation) * np.abs(volume_zscore)
    
    # Adaptive mean-reversion strength
    mean_reversion_strength = -price_deviation / (regime_strength.rolling(window=10).std() + 1e-8)
    
    # 4. Microstructure Noise Ratio
    # Estimate noise component from intraday volatility
    intraday_range = (data['high'] - data['low']) / data['close']
    overnight_gap = np.abs(data['open'] / data['close'].shift(1) - 1)
    noise_component = intraday_range / (overnight_gap + 1e-8)
    
    # Signal-to-noise ratio using price patterns
    trend_strength = data['close'].rolling(window=10).std() / data['close'].rolling(window=50).std()
    signal_noise_ratio = trend_strength / (noise_component + 1e-8)
    
    # 5. Volume-Weighted Price Acceleration
    # Compute second derivative of VWAP
    vwap = (data['close'] * data['volume']).rolling(window=10).sum() / data['volume'].rolling(window=10).sum()
    vwap_velocity = vwap.diff()
    vwap_acceleration = vwap_velocity.diff()
    
    # Acceleration persistence
    acceleration_persistence = vwap_acceleration.rolling(window=5).apply(
        lambda x: np.sum(x > 0) / len(x) if len(x) == 5 else np.nan
    )
    
    # 6. Liquidity Gap Convergence
    # Identify liquidity gaps from volume patterns
    volume_quantile = data['volume'].rolling(window=20).apply(
        lambda x: (x.iloc[-1] - x.quantile(0.2)) / (x.quantile(0.8) - x.quantile(0.2) + 1e-8)
    )
    
    # Price convergence toward gaps
    gap_detection = (data['close'] - data['open']) / (data['high'] - data['low'] + 1e-8)
    convergence_speed = gap_detection.diff().rolling(window=5).mean()
    
    # 7. Multi-Timeframe Pressure Alignment
    # Calculate buying pressure across multiple periods
    def buying_pressure(high, low, close, volume, window):
        pressure = ((close - low) - (high - close)) / (high - low + 1e-8) * volume
        return pressure.rolling(window=window).mean()
    
    pressure_short = buying_pressure(data['high'], data['low'], data['close'], data['volume'], 5)
    pressure_medium = buying_pressure(data['high'], data['low'], data['close'], data['volume'], 10)
    pressure_long = buying_pressure(data['high'], data['low'], data['close'], data['volume'], 20)
    
    # Alignment strength
    pressure_alignment = (pressure_short * pressure_medium * pressure_long) ** (1/3)
    
    # Combine all factors with appropriate weights
    factor = (
        0.15 * hurst_factor.rank(pct=True) +
        0.12 * cumulative_order_flow.rank(pct=True) +
        0.18 * mean_reversion_strength.rank(pct=True) +
        0.14 * signal_noise_ratio.rank(pct=True) +
        0.16 * acceleration_persistence.rank(pct=True) +
        0.13 * convergence_speed.rank(pct=True) +
        0.12 * pressure_alignment.rank(pct=True)
    )
    
    return factor
