import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Initialize factor series
    factor = pd.Series(index=data.index, dtype=float)
    
    # Intraday Reversal with Volume Spike
    intraday_return = (data['high'] - data['low']) / data['open']
    volume_ma_20 = data['volume'].rolling(window=20, min_periods=1).mean()
    volume_spike = (data['volume'] > 2 * volume_ma_20).astype(int)
    factor1 = -1 * (intraday_return * volume_spike)
    
    # Volatility-Adjusted Price Momentum
    returns_5d = data['close'].pct_change(periods=5)
    returns_20d = data['close'].pct_change(periods=1).rolling(window=20, min_periods=1).std()
    factor2 = returns_5d / (returns_20d + 1e-8)
    
    # Volume-Price Divergence
    def linear_slope(series, window):
        x = np.arange(window)
        slopes = []
        for i in range(len(series)):
            if i >= window - 1:
                y = series.iloc[i-window+1:i+1].values
                slope = (window * np.sum(x * y) - np.sum(x) * np.sum(y)) / (window * np.sum(x**2) - np.sum(x)**2)
                slopes.append(slope)
            else:
                slopes.append(np.nan)
        return pd.Series(slopes, index=series.index)
    
    price_slope = linear_slope(data['close'], 5)
    volume_slope = linear_slope(data['volume'], 5)
    factor3 = price_slope * volume_slope
    
    # Amplitude-Weighted Accumulation
    daily_amplitude = (data['high'] - data['low']) / ((data['high'] + data['low']) / 2)
    accumulation_3d = (data['close'] - data['open']).rolling(window=3, min_periods=1).sum()
    factor4 = accumulation_3d * daily_amplitude
    
    # Gap Persistence
    gap_size = (data['open'] / data['close'].shift(1)) - 1
    persistence = np.sign(data['open'] - data['close'].shift(1)) * np.sign(data['close'] - data['open'])
    factor5 = gap_size * persistence
    
    # Volume-Weighted Price Range
    price_range_ratio = (data['high'] - data['low']) / data['close']
    volume_ma_10 = data['volume'].rolling(window=10, min_periods=1).mean()
    volume_ratio = data['volume'] / (volume_ma_10 + 1e-8)
    factor6 = price_range_ratio * volume_ratio
    
    # Momentum Acceleration
    momentum_5d = data['close'] / data['close'].shift(5) - 1
    momentum_10d = data['close'] / data['close'].shift(10) - 1
    factor7 = momentum_5d - momentum_10d
    
    # Opening Gap Reversal
    gap = (data['open'] / data['close'].shift(1)) - 1
    intraday_reversal = (data['close'] - data['open']) / data['open']
    factor8 = -1 * gap * intraday_reversal
    
    # Volume-Adjusted Volatility
    daily_volatility = (data['high'] - data['low']) / data['close']
    volume_change = data['volume'] / data['volume'].shift(1) - 1
    factor9 = daily_volatility * volume_change
    
    # Combine all factors with equal weights
    factors_df = pd.DataFrame({
        'f1': factor1, 'f2': factor2, 'f3': factor3, 'f4': factor4,
        'f5': factor5, 'f6': factor6, 'f7': factor7, 'f8': factor8, 'f9': factor9
    })
    
    # Fill NaN values with 0 and calculate final factor
    factors_df = factors_df.fillna(0)
    final_factor = factors_df.mean(axis=1)
    
    return final_factor
