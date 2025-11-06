import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Generate alpha factor combining multi-timeframe momentum acceleration, 
    range-efficiency reversal, volatility-regime skewness, and volume-surge momentum divergence.
    """
    # Copy dataframe to avoid modifying original
    data = df.copy()
    
    # Multi-Timeframe Momentum Acceleration
    # Calculate momentum differences
    ret_5d = data['close'].pct_change(5)
    ret_10d = data['close'].pct_change(10)
    ret_20d = data['close'].pct_change(20)
    
    mom_diff_5_10 = ret_5d - ret_10d
    mom_diff_10_20 = ret_10d - ret_20d
    
    # Volume trend direction (5-day vs 20-day volume MA)
    vol_ma_5 = data['volume'].rolling(window=5).mean()
    vol_ma_20 = data['volume'].rolling(window=20).mean()
    vol_trend = (vol_ma_5 - vol_ma_20) / vol_ma_20
    
    # Combine momentum acceleration with volume trend
    momentum_accel = (mom_diff_5_10 * 0.6 + mom_diff_10_20 * 0.4) * (1 + vol_trend)
    
    # Range-Efficiency Reversal
    # Compute movement efficiency
    price_change = abs(data['close'] - data['close'].shift(1))
    true_range = np.maximum(
        data['high'] - data['low'],
        np.maximum(
            abs(data['high'] - data['close'].shift(1)),
            abs(data['low'] - data['close'].shift(1))
        )
    )
    movement_efficiency = price_change / true_range.replace(0, np.nan)
    
    # Daily amplitude (normalized range)
    daily_amplitude = (data['high'] - data['low']) / data['close']
    
    # Reversal signal (negative autocorrelation of returns)
    returns = data['close'].pct_change()
    reversal_signal = -returns.rolling(window=5).apply(lambda x: x.corr(x.shift(1)) if len(x.dropna()) > 3 else 0)
    
    # Weight reversal by efficiency and amplitude
    range_reversal = reversal_signal * movement_efficiency * daily_amplitude
    
    # Volatility-Regime Skewness
    # Calculate 20-day skewness
    returns_20d = data['close'].pct_change().rolling(window=20)
    skewness_20d = returns_20d.apply(lambda x: x.skew() if len(x.dropna()) > 15 else 0)
    
    # Volatility ratio (short-term vs long-term)
    vol_short = returns.rolling(window=5).std()
    vol_long = returns.rolling(window=20).std()
    vol_ratio = vol_short / vol_long.replace(0, np.nan)
    
    # Intraday pressure accumulation (close relative to range)
    intraday_pressure = (data['close'] - data['low']) / (data['high'] - data['low']).replace(0, np.nan)
    pressure_trend = intraday_pressure.rolling(window=5).mean()
    
    # Combine skewness with volatility regime and pressure
    vol_skewness = skewness_20d * vol_ratio * pressure_trend
    
    # Volume-Surge Momentum Divergence
    # Detect volume surges
    vol_1d = data['volume']
    vol_5d_ma = data['volume'].rolling(window=5).mean()
    volume_surge = vol_1d / vol_5d_ma.replace(0, np.nan)
    
    # Momentum divergence (current momentum vs smoothed momentum)
    mom_1d = returns
    mom_5d_smooth = returns.rolling(window=5).mean()
    momentum_divergence = mom_1d - mom_5d_smooth
    
    # Combine volume surge with momentum divergence
    volume_momentum_div = volume_surge * momentum_divergence
    
    # Final alpha factor combination
    # Normalize components and combine with weights
    components = pd.DataFrame({
        'momentum_accel': momentum_accel,
        'range_reversal': range_reversal,
        'vol_skewness': vol_skewness,
        'volume_momentum_div': volume_momentum_div
    })
    
    # Z-score normalization for each component
    normalized_components = components.apply(lambda x: (x - x.rolling(window=60).mean()) / x.rolling(window=60).std())
    
    # Final alpha factor with weights
    alpha_factor = (
        normalized_components['momentum_accel'] * 0.3 +
        normalized_components['range_reversal'] * 0.25 +
        normalized_components['vol_skewness'] * 0.25 +
        normalized_components['volume_momentum_div'] * 0.2
    )
    
    return alpha_factor
