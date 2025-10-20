import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Intraday Momentum Divergence
    # Calculate intraday ratio
    intraday_ratio = (data['high'] - data['open']) / (data['open'] - data['low']).replace(0, np.nan)
    
    # 5-day average ratio
    avg_ratio_5d = intraday_ratio.rolling(window=5, min_periods=3).mean()
    
    # Intraday divergence score
    intraday_divergence = intraday_ratio - avg_ratio_5d
    
    # Volatility regime calculation
    daily_range = (data['high'] - data['low']) / data['close']
    vol_20d = daily_range.rolling(window=20, min_periods=15).std()
    
    # Classify volatility regime
    vol_percentile = vol_20d.rolling(window=60, min_periods=40).apply(
        lambda x: (x.iloc[-1] - x.mean()) / x.std() if x.std() > 0 else 0, raw=False
    )
    
    # Regime-specific scaling
    current_range = (data['high'] - data['low']) / data['close']
    regime_scaling = np.where(
        vol_percentile > 1, 
        1 / (current_range + 0.001),  # High volatility: inverse range
        np.where(
            vol_percentile < -0.5,
            intraday_divergence,  # Low volatility: raw divergence
            intraday_divergence * (1 / (current_range + 0.001)) * 0.5  # Medium: blend
        )
    )
    
    # Volume-Weighted Price Acceleration
    # Volume-weighted price change
    avg_volume_20d = data['volume'].rolling(window=20, min_periods=15).mean()
    vol_weighted_change = (data['close'] - data['open']) * data['volume'] / (avg_volume_20d + 1)
    
    # 3-day momentum gradient
    def calc_slope_3d(series):
        if len(series) < 3:
            return np.nan
        x = np.arange(len(series))
        return np.polyfit(x, series.values, 1)[0]
    
    momentum_gradient = vol_weighted_change.rolling(window=3, min_periods=3).apply(
        calc_slope_3d, raw=False
    )
    
    # 5-day volume slope
    volume_slope = data['volume'].rolling(window=5, min_periods=4).apply(
        calc_slope_3d, raw=False
    )
    
    # Combined acceleration score
    acceleration_score = momentum_gradient * volume_slope
    
    # Opening Gap Momentum Persistence
    # Gap percentage
    gap_pct = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
    
    # Gap fill percentage
    gap_fill_pct = np.where(
        gap_pct > 0,
        (data['high'] - data['open']) / (gap_pct * data['close'].shift(1) + 0.001),
        (data['open'] - data['low']) / (-gap_pct * data['close'].shift(1) + 0.001)
    )
    gap_fill_pct = np.clip(gap_fill_pct, 0, 1)
    gap_remaining = 1 - gap_fill_pct
    
    # Volume ratio
    volume_ratio = data['volume'] / avg_volume_20d
    
    # Persistence score
    persistence_score = gap_pct * volume_ratio * gap_remaining
    
    # Price-Volume Efficiency Ratio
    # Price efficiency
    price_efficiency = abs(data['close'] - data['open']) / (data['high'] - data['low'] + 0.001)
    
    # Volume efficiency (10-day correlation)
    def calc_volume_corr(window):
        if len(window) < 5:
            return np.nan
        close_prices = window['close']
        volumes = window['volume']
        if close_prices.std() == 0 or volumes.std() == 0:
            return 0
        return abs(np.corrcoef(close_prices, volumes)[0, 1])
    
    volume_efficiency = data.rolling(window=10, min_periods=5).apply(
        calc_volume_corr, raw=False
    )
    
    # Composite efficiency score
    efficiency_score = price_efficiency * volume_efficiency
    
    # Range Breakout Confirmation
    # 10-day range levels
    rolling_high_10d = data['high'].rolling(window=10, min_periods=8).max()
    rolling_low_10d = data['low'].rolling(window=10, min_periods=8).min()
    
    # Breakout detection
    upper_breakout = (data['close'] > rolling_high_10d.shift(1)).astype(int)
    lower_breakout = (data['close'] < rolling_low_10d.shift(1)).astype(int)
    
    # Volume surge
    volume_surge = data['volume'] / avg_volume_20d
    
    # Intraday strength
    intraday_strength = (data['close'] - data['open']) / (data['high'] - data['low'] + 0.001)
    
    # Breakout score
    breakout_score = (upper_breakout - lower_breakout) * volume_surge * intraday_strength
    
    # Combine all factors with equal weights
    combined_factor = (
        regime_scaling.fillna(0) * 0.2 +
        acceleration_score.fillna(0) * 0.2 +
        persistence_score.fillna(0) * 0.2 +
        efficiency_score.fillna(0) * 0.2 +
        breakout_score.fillna(0) * 0.2
    )
    
    return pd.Series(combined_factor, index=data.index)
