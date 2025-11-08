import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Generate novel alpha factors for stock return prediction using multiple heuristics.
    
    Parameters:
    df: DataFrame with columns ['open', 'high', 'low', 'close', 'amount', 'volume']
    
    Returns:
    Series: Combined alpha factor values indexed by date
    """
    
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Initialize factor storage
    factors = pd.DataFrame(index=data.index)
    
    # 1. Volatility-Adjusted Gap Momentum
    gap_magnitude = np.abs(data['open'] / data['close'].shift(1) - 1)
    intraday_vol = (data['high'] - data['low']) / data['close']
    intraday_vol = intraday_vol.replace(0, np.nan)  # Avoid division by zero
    factors['factor1'] = (gap_magnitude / intraday_vol) * np.sign(data['open'] / data['close'].shift(1) - 1)
    
    # 2. Volume-Weighted Price Acceleration
    price_accel = (data['close'] / data['close'].shift(5) - 1) - (data['close'].shift(5) / data['close'].shift(10) - 1)
    volume_surge = data['volume'] / data['volume'].rolling(window=10, min_periods=5).mean().shift(1)
    factors['factor2'] = price_accel * (volume_surge ** (1/3))
    
    # 3. Range Efficiency Oscillator
    max_potential_range = data['high'].rolling(window=6, min_periods=5).max() - data['low'].rolling(window=6, min_periods=5).min()
    actual_range_util = (data['high'] - data['low']).rolling(window=6, min_periods=5).mean()
    range_efficiency = 1 - (actual_range_util / max_potential_range.replace(0, np.nan))
    factors['factor3'] = range_efficiency * (data['close'] / data['close'].shift(1) - 1)
    
    # 4. Pressure Diffusion Indicator
    pressure_gradient = (data['close'] - data['low']) / (data['high'] - data['low']).replace(0, np.nan)
    
    def calc_pressure_momentum(series):
        if len(series) < 6:
            return np.nan
        x = np.arange(len(series))
        return np.polyfit(x, series.values, 1)[0]
    
    pressure_momentum = pressure_gradient.rolling(window=6, min_periods=5).apply(calc_pressure_momentum, raw=False)
    factors['factor4'] = np.tanh(pressure_gradient * pressure_momentum)
    
    # 5. Volume-Clustered Reversal
    low_volume_filter = data['volume'] < data['volume'].rolling(window=16, min_periods=15).quantile(0.2)
    
    def percentile_rank(current, window_data):
        if len(window_data) < 15:
            return np.nan
        return (current > window_data).sum() / len(window_data)
    
    volume_rank = []
    for i in range(len(data)):
        if i < 15:
            volume_rank.append(np.nan)
        else:
            window_volumes = data['volume'].iloc[i-15:i].values
            volume_rank.append(percentile_rank(data['volume'].iloc[i], window_volumes))
    
    factors['factor5'] = (-(data['close'] / data['close'].shift(1) - 1)) * volume_rank
    factors.loc[~low_volume_filter, 'factor5'] = 0
    
    # 6. Amplitude-Modulated Trend
    price_amplitude = (data['high'] - data['low']) / ((data['high'] + data['low']) / 2).replace(0, np.nan)
    
    def trend_persistence_corr(window_data):
        if len(window_data) < 5:
            return np.nan
        returns = window_data.pct_change().dropna()
        if len(returns) < 4:
            return np.nan
        x = np.arange(len(returns))
        return np.corrcoef(x, returns.values)[0, 1]
    
    trend_persistence = data['close'].rolling(window=6, min_periods=5).apply(trend_persistence_corr, raw=False)
    factors['factor6'] = (data['close'] / data['close'].shift(5) - 1) * price_amplitude * trend_persistence / np.sqrt(price_amplitude.replace(0, np.nan))
    
    # 7. Gap-Fill Momentum Divergence
    gap_fill_progress = (data['close'] - data['open']) / (data['close'].shift(1) - data['open']).replace(0, np.nan)
    volume_ratio = data['volume'] / data['volume'].rolling(window=5, min_periods=4).mean().shift(1)
    indicator_open_above = (data['open'] > data['close'].shift(1)).astype(int)
    factors['factor7'] = np.abs(gap_fill_progress * np.log(volume_ratio.replace(0, np.nan))) * indicator_open_above
    
    # 8. Efficiency-Regime Momentum
    market_efficiency = np.abs(data['close'] - data['close'].shift(1)) / (data['high'] - data['low']).replace(0, np.nan)
    regime_filter = market_efficiency < 0.5
    factors['factor8'] = ((data['close'] / data['close'].shift(5) - 1) / market_efficiency.replace(0, np.nan)) * ((1 - market_efficiency) ** 2) * regime_filter.astype(int)
    
    # 9. Pressure-Volume Congestion
    price_compression = (data['high'] - data['low']).rolling(window=6, min_periods=5).max() / (data['high'] - data['low']).rolling(window=6, min_periods=5).min().replace(0, np.nan)
    volume_compression = data['volume'].rolling(window=6, min_periods=5).max() / data['volume'].rolling(window=6, min_periods=5).min().replace(0, np.nan)
    factors['factor9'] = (data['close'] / data['close'].shift(1) - 1) / (price_compression * volume_compression)
    
    # 10. Range-Bound Momentum Capture
    range_position = (data['close'] - data['low'].rolling(window=21, min_periods=20).min()) / (data['high'].rolling(window=21, min_periods=20).max() - data['low'].rolling(window=21, min_periods=20).min()).replace(0, np.nan)
    range_momentum = range_position - range_position.shift(5)
    range_filter = ((range_position < 0.2) | (range_position > 0.8)).astype(int)
    factors['factor10'] = range_position * range_momentum * range_filter * (range_position ** 3)
    
    # Combine factors using equal weighting
    combined_factor = factors.mean(axis=1, skipna=True)
    
    return combined_factor
