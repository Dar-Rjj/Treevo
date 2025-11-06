import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Intraday Reversal with Volume Spike
    # Compute Intraday Return
    intraday_return = (data['high'] - data['low']) / data['open']
    abs_intraday_return = intraday_return.abs()
    
    # Identify Volume Spike
    vol_ma_20 = data['volume'].rolling(window=20, min_periods=1).mean()
    volume_ratio = data['volume'] / vol_ma_20
    volume_spike_indicator = (volume_ratio > 2.0).astype(float)
    
    # Combine Signals with reversal
    reversal_factor = -1 * (abs_intraday_return * volume_spike_indicator)
    
    # Volatility-Adjusted Price Momentum
    # Calculate Price Momentum
    price_momentum = (data['close'] - data['close'].shift(5)) / data['close'].shift(5)
    
    # Compute Volatility Measure
    daily_returns = data['close'].pct_change()
    volatility_20 = daily_returns.rolling(window=20, min_periods=1).std()
    
    # Adjust Momentum by Volatility
    vol_adjusted_momentum = price_momentum / volatility_20.replace(0, np.nan)
    
    # Volume-Price Divergence Factor
    # Price Trend Component
    def calc_slope(series, window=5):
        x = np.arange(window)
        slopes = []
        for i in range(len(series)):
            if i >= window - 1:
                y = series.iloc[i-window+1:i+1].values
                slope = np.polyfit(x, y, 1)[0]
                slopes.append(slope)
            else:
                slopes.append(np.nan)
        return pd.Series(slopes, index=series.index)
    
    price_slope = calc_slope(data['close'], 5)
    volume_slope = calc_slope(data['volume'], 5)
    
    # Detect Divergence
    divergence_factor = price_slope * volume_slope
    
    # Amplitude-Weighted Accumulation
    # Calculate Daily Amplitude
    daily_amplitude = (data['high'] - data['low']) / ((data['high'] + data['low']) / 2)
    
    # Compute Accumulation Pattern
    intraday_movement = data['close'] - data['open']
    accumulation_3day = intraday_movement.rolling(window=3, min_periods=1).sum()
    
    # Weight Accumulation by Amplitude
    amplitude_weighted_factor = accumulation_3day * daily_amplitude
    
    # Gap Persistence Factor
    # Identify Price Gaps
    gap_size = (data['open'] / data['close'].shift(1)) - 1
    
    # Measure Gap Persistence
    gap_direction = np.sign(data['open'] - data['close'].shift(1))
    intraday_direction = np.sign(data['close'] - data['open'])
    persistence_indicator = gap_direction * intraday_direction
    
    # Combine Gap Size and Persistence
    gap_persistence_factor = gap_size * persistence_indicator
    
    # Combine all factors with equal weighting
    combined_factor = (
        reversal_factor.fillna(0) + 
        vol_adjusted_momentum.fillna(0) + 
        divergence_factor.fillna(0) + 
        amplitude_weighted_factor.fillna(0) + 
        gap_persistence_factor.fillna(0)
    )
    
    return combined_factor
