import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import linregress

def heuristics_v2(df):
    # Copy data to avoid modifying original
    data = df.copy()
    
    # 1. Multi-timeframe Order Flow Pressure
    # Intraday Order Imbalance
    intraday_imbalance = (2 * data['close'] - data['high'] - data['low']) / (data['high'] - data['low'] + 1e-8)
    volume_weighted_imbalance = intraday_imbalance * np.sqrt(data['volume'])
    
    # Cumulative Order Flow Signals with Inverse Hyperbolic Sine
    short_term_pressure = volume_weighted_imbalance.rolling(window=3, min_periods=1).sum()
    medium_term_pressure = volume_weighted_imbalance.rolling(window=8, min_periods=1).sum()
    
    def inverse_hyperbolic_sine(x):
        return np.log(x + np.sqrt(x**2 + 1))
    
    order_flow_signal = inverse_hyperbolic_sine(short_term_pressure) + inverse_hyperbolic_sine(medium_term_pressure)
    
    # 2. Volatility Regime Context
    # ATR calculation
    def calculate_atr(data, period):
        high_low = data['high'] - data['low']
        high_close = np.abs(data['high'] - data['close'].shift(1))
        low_close = np.abs(data['low'] - data['close'].shift(1))
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return true_range.rolling(window=period, min_periods=1).mean()
    
    atr_5 = calculate_atr(data, 5)
    atr_20 = calculate_atr(data, 20)
    
    # Volatility Compression Ratio with Logistic Scaling
    volatility_ratio = atr_5 / (atr_20 + 1e-8)
    def logistic_scaling(x):
        return 1 / (1 + np.exp(-10 * (x - 0.5)))
    volatility_compression = logistic_scaling(volatility_ratio)
    
    # Regime-Adaptive Volatility Measure
    morning_range = (data['high'] - data['low']) / (data['open'] + 1e-8)
    liquidity_proxy = data['volume'] * (data['high'] - data['low']) / (data['close'] + 1e-8)
    regime_volatility = morning_range * liquidity_proxy
    regime_volatility_smooth = regime_volatility.rolling(window=3, min_periods=1).median()
    
    # 3. VWAP Momentum Components
    def calculate_vwap(data):
        return (data['close'] * data['volume']).cumsum() / data['volume'].cumsum()
    
    vwap = calculate_vwap(data)
    vwap_momentum_5 = vwap / vwap.shift(5) - 1
    vwap_momentum_10 = vwap / vwap.shift(10) - 1
    
    # Volume slope using linear regression
    def volume_slope(series, window):
        slopes = []
        for i in range(len(series)):
            if i < window - 1:
                slopes.append(0)
            else:
                y = series.iloc[i-window+1:i+1].values
                x = np.arange(window)
                slope, _, _, _, _ = linregress(x, y)
                slopes.append(slope)
        return pd.Series(slopes, index=series.index)
    
    vol_slope = volume_slope(data['volume'], 5)
    price_vwap_deviation = data['close'] / vwap - 1
    
    # 4. Core Reversion-Momentum Signal
    # Afternoon Return Component
    midday_level = (data['open'] + data['high']) / 2
    afternoon_return = (data['close'] - midday_level) / (midday_level + 1e-8)
    dollar_movement_adjusted = afternoon_return * np.abs(afternoon_return) * data['amount']
    
    # VWAP Momentum Alignment Score
    vwap_momentum_combined = vwap_momentum_5 + 0.5 * vwap_momentum_10
    volume_alignment = np.sign(vol_slope) * np.sign(vwap_momentum_combined)
    vwap_alignment_score = vwap_momentum_combined * volume_alignment
    
    # Combine components with volatility regime adjustment
    core_signal = dollar_movement_adjusted * order_flow_signal * vwap_alignment_score
    volatility_adjusted_signal = core_signal / (1 + volatility_compression) * regime_volatility_smooth
    
    # 5. Dynamic Signal Aggregation and Filtering
    # Volume-Confirmed Signal Enhancement
    volume_change_ratio = data['volume'] / (data['volume'].shift(1) + 1e-8)
    volume_enhanced_signal = volatility_adjusted_signal * volume_change_ratio
    
    # Adaptive Timeframe Selection
    atr_10 = calculate_atr(data, 10)
    atr_percentile = atr_10.rolling(window=10, min_periods=1).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )
    
    def adaptive_smoothing(signal, volatility_percentile, high_vol_threshold=0.7):
        ema_3 = signal.ewm(span=3, adjust=False).mean()
        sma_8 = signal.rolling(window=8, min_periods=1).mean()
        
        # Use EMA during high volatility, SMA during low volatility
        return np.where(volatility_percentile > high_vol_threshold, ema_3, sma_8)
    
    final_signal = adaptive_smoothing(volume_enhanced_signal, atr_percentile)
    
    return final_signal
