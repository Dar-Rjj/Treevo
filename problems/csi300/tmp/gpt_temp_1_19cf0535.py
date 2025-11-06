import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Volatility-Adjusted Intraday Reversal
    intraday_return = data['close'] / data['open'] - 1
    daily_range = data['high'] / data['low'] - 1
    volatility_adjusted_reversal = -intraday_return / (daily_range + 1e-8)  # Add small constant to avoid division by zero
    
    # Range-Breakout Momentum
    recent_high = data['high'].rolling(window=5, min_periods=1).apply(lambda x: x[:-1].max() if len(x) > 1 else x[0], raw=True)
    breakout_distance = data['close'] / recent_high - 1
    vwap = data['amount'] / (data['volume'] + 1e-8)
    range_breakout_momentum = breakout_distance * vwap
    
    # Gap-Persistence Momentum
    opening_gap = data['open'] / data['close'].shift(1) - 1
    intraday_momentum = data['close'] / data['open'] - 1
    gap_persistence_momentum = opening_gap * intraday_momentum
    
    # Volume-Spike Reversal
    def rolling_percentile(series, window):
        result = pd.Series(index=series.index, dtype=float)
        for i in range(len(series)):
            if i >= window:
                window_data = series.iloc[i-window:i]
                current_val = series.iloc[i]
                result.iloc[i] = (window_data < current_val).sum() / len(window_data)
            else:
                result.iloc[i] = np.nan
        return result
    
    volume_percentile = rolling_percentile(data['volume'], 20)
    recent_high_5 = data['high'].rolling(window=5, min_periods=1).apply(lambda x: x[:-1].max() if len(x) > 1 else x[0], raw=True)
    price_reversal = data['close'] / recent_high_5 - 1
    volume_spike_reversal = -price_reversal * (volume_percentile > 0.95)
    
    # Combine factors (equal weighting)
    combined_factor = (
        volatility_adjusted_reversal + 
        range_breakout_momentum + 
        gap_persistence_momentum + 
        volume_spike_reversal
    )
    
    return combined_factor
