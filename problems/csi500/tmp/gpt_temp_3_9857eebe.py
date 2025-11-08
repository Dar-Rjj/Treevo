import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Momentum Divergence
    price_5d_return = data['close'].pct_change(5)
    price_10d_return = data['close'].pct_change(10)
    price_momentum_ratio = price_5d_return / price_10d_return.replace(0, np.nan)
    
    volume_5d_avg = data['volume'].rolling(5).mean()
    volume_10d_avg = data['volume'].rolling(10).mean()
    volume_momentum_ratio = volume_5d_avg / volume_10d_avg.replace(0, np.nan)
    
    momentum_divergence = price_momentum_ratio - volume_momentum_ratio
    
    # Range Efficiency
    prev_close = data['close'].shift(1)
    range1 = data['high'] - data['low']
    range2 = abs(data['high'] - prev_close)
    range3 = abs(data['low'] - prev_close)
    max_range = pd.concat([range1, range2, range3], axis=1).max(axis=1)
    range_efficiency = abs(data['close'] - data['open']) / max_range.replace(0, np.nan)
    
    # Volume-Confirmed Breakout
    high_20d = data['high'].rolling(20).max()
    low_20d = data['low'].rolling(20).min()
    volume_20d_avg = data['volume'].rolling(20).mean()
    
    breakout_up = (data['close'] > high_20d.shift(1)).astype(int)
    breakout_down = (data['close'] < low_20d.shift(1)).astype(int)
    volume_filter = (data['volume'] > 1.5 * volume_20d_avg).astype(int)
    
    volume_breakout = (breakout_up - breakout_down) * volume_filter
    
    # Volatility-Adjusted Momentum
    def calculate_atr(data, window):
        high_low = data['high'] - data['low']
        high_prev_close = abs(data['high'] - data['close'].shift(1))
        low_prev_close = abs(data['low'] - data['close'].shift(1))
        true_range = pd.concat([high_low, high_prev_close, low_prev_close], axis=1).max(axis=1)
        return true_range.rolling(window).mean()
    
    atr_5d = calculate_atr(data, 5)
    atr_20d = calculate_atr(data, 20)
    volatility_regime = atr_5d / atr_20d.replace(0, np.nan)
    volatility_adjusted_momentum = price_5d_return / volatility_regime.replace(0, np.nan)
    
    # Volume-Weighted Acceleration
    price_3d_return = data['close'].pct_change(3)
    price_6d_return = data['close'].pct_change(6)
    price_acceleration = price_3d_return - price_6d_return
    
    volume_3d_avg = data['volume'].rolling(3).mean()
    volume_6d_avg = data['volume'].rolling(6).mean()
    volume_weighted_acceleration = price_acceleration * (volume_3d_avg / volume_6d_avg.replace(0, np.nan))
    
    # Order Flow Imbalance
    high_low_range = (data['high'] - data['low']).replace(0, np.nan)
    close_to_low = data['close'] - data['low']
    high_to_close = data['high'] - data['close']
    order_flow = ((close_to_low - high_to_close) / high_low_range) * data['volume']
    order_flow_imbalance = order_flow.rolling(5).sum()
    
    # Gap Analysis
    prev_extreme = pd.concat([data['high'].shift(1), data['low'].shift(1)], axis=1).max(axis=1)
    gap_magnitude = abs(data['open'] - prev_extreme)
    atr_10d = calculate_atr(data, 10)
    volume_10d_avg = data['volume'].rolling(10).mean()
    gap_analysis = -gap_magnitude / atr_10d.replace(0, np.nan) * (data['volume'] / volume_10d_avg.replace(0, np.nan))
    
    # Multi-Timeframe Volume
    volume_3d_change = data['volume'].pct_change(3)
    
    def volume_slope(series, window):
        x = np.arange(window)
        def calc_slope(window_data):
            if len(window_data) == window and not window_data.isna().any():
                return np.polyfit(x, window_data, 1)[0]
            return np.nan
        return series.rolling(window).apply(calc_slope, raw=False)
    
    volume_10d_slope = volume_slope(data['volume'], 10)
    volume_30d_avg = data['volume'].rolling(30).mean()
    multi_timeframe_volume = data['volume'] / volume_30d_avg.replace(0, np.nan)
    
    # Price-Volume Correlation
    def rolling_corr(x, y, window):
        return x.rolling(window).corr(y)
    
    price_volume_corr = rolling_corr(data['close'], data['volume'], 20).abs()
    
    rolling_corrs = []
    for i in range(20, len(data)):
        if i >= 25:
            corr_window = rolling_corr(data['close'], data['volume'], 20).iloc[i-5:i]
            rolling_corrs.append(corr_window.std())
        else:
            rolling_corrs.append(np.nan)
    
    corr_std = pd.Series(rolling_corrs, index=data.index[20:])
    corr_std = corr_std.reindex(data.index)
    price_volume_signal = price_volume_corr / corr_std.replace(0, np.nan)
    
    # Combine all factors with equal weights
    factors = pd.DataFrame({
        'momentum_divergence': momentum_divergence,
        'range_efficiency': range_efficiency,
        'volume_breakout': volume_breakout,
        'volatility_momentum': volatility_adjusted_momentum,
        'volume_acceleration': volume_weighted_acceleration,
        'order_flow': order_flow_imbalance,
        'gap_analysis': gap_analysis,
        'multi_timeframe_volume': multi_timeframe_volume,
        'price_volume_signal': price_volume_signal
    })
    
    # Standardize and combine
    factors_standardized = (factors - factors.rolling(60).mean()) / factors.rolling(60).std()
    combined_factor = factors_standardized.mean(axis=1)
    
    return combined_factor
