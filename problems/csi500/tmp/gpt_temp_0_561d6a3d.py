import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Price Acceleration Components
    roc_5 = df['close'].pct_change(5)
    roc_10 = df['close'].pct_change(10)
    roc_20 = df['close'].pct_change(20)
    
    short_term_acc = roc_5 - roc_10
    medium_term_acc = roc_10 - roc_20
    acceleration_regime = np.sign(short_term_acc) * np.sign(medium_term_acc)
    
    # Volume Confirmation Patterns
    volume_decay_ratio = df['volume'] / df['volume'].shift(5)
    volume_trend_intensity = df['volume'] / df['volume'].rolling(20, min_periods=1).median()
    volume_price_corr_acc = df['close'].pct_change().rolling(10, min_periods=1).corr(df['volume'].pct_change())
    
    # Acceleration Signal
    acceleration_signal = acceleration_regime * volume_decay_ratio * volume_trend_intensity
    
    # Dual-Timeframe Efficiency Momentum
    close_diff_5 = df['close'] - df['close'].shift(5)
    close_diff_15 = df['close'] - df['close'].shift(15)
    
    abs_returns_5 = df['close'].diff().abs().rolling(5, min_periods=1).sum()
    abs_returns_15 = df['close'].diff().abs().rolling(15, min_periods=1).sum()
    
    short_term_efficiency = close_diff_5 / abs_returns_5
    medium_term_efficiency = close_diff_15 / abs_returns_15
    efficiency_momentum = (short_term_efficiency - medium_term_efficiency) * np.sign(close_diff_5)
    
    # Adaptive Volatility Scaling
    range_volatility = (df['high'] - df['low']).rolling(20, min_periods=1).std()
    return_volatility = (df['close'].pct_change()).rolling(20, min_periods=1).std()
    combined_volatility = np.sqrt(range_volatility * return_volatility)
    
    # Scaled Efficiency
    scaled_efficiency = efficiency_momentum / combined_volatility
    
    # Intraday Pressure Dynamics
    intraday_spread = (df['high'] - df['low']) / df['close']
    vwap_deviation = (df['close'] - (df['high'] + df['low'] + df['close']) / 3) / df['close'].abs()
    
    # Price Pressure (slope of close over 3 days)
    def rolling_slope(x):
        if len(x) < 3:
            return np.nan
        x_vals = np.arange(len(x))
        return np.polyfit(x_vals, x, 1)[0]
    
    price_pressure = df['close'].rolling(3, min_periods=1).apply(rolling_slope, raw=True)
    
    # Volume Spike Detection
    volume_spike_ratio = df['volume'] / df['volume'].rolling(20, min_periods=1).mean()
    volume_trend = (df['volume'].rolling(10, min_periods=1).mean() / 
                   df['volume'].rolling(20, min_periods=1).mean() - 1)
    
    # Pressure Signal
    pressure_signal = -intraday_spread * price_pressure * volume_spike_ratio
    
    # Gap Persistence with Volume Confirmation
    opening_gap = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
    
    def gap_persistence_score(gap_series, window=3):
        scores = []
        for i in range(len(gap_series)):
            if i < window:
                scores.append(0)
            else:
                window_data = gap_series.iloc[i-window+1:i+1]
                persistence_count = sum(np.sign(window_data.iloc[j]) == np.sign(window_data.iloc[j-1]) 
                                      for j in range(1, len(window_data)))
                scores.append(persistence_count / (window - 1))
        return pd.Series(scores, index=gap_series.index)
    
    persistence_score = gap_persistence_score(opening_gap, 3)
    
    # Volume Confirmation for Gap
    volume_confirmation_ratio = df['volume'] / df['volume'].rolling(5, min_periods=1).mean()
    volume_pattern = df['volume'] / df['volume'].rolling(20, min_periods=1).median()
    
    # Gap Signal
    gap_signal = opening_gap * persistence_score * volume_confirmation_ratio
    
    # Volume-Price Integration
    volume_price_correlation = df['volume'].rolling(10, min_periods=1).corr(df['close'])
    volume_trend_confirmation = volume_trend_intensity * volume_trend
    
    # Composite Factor Construction
    base_composite = acceleration_signal * scaled_efficiency * pressure_signal
    final_alpha = base_composite * gap_signal * volume_price_correlation
    
    return final_alpha
