import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy import stats

def heuristics_v2(df):
    """
    Price-Volume Confirmation Factor that combines volatility-adjusted momentum
    with volume confirmation signals for enhanced return prediction.
    """
    # Volatility-Adjusted Price Momentum
    # Short-term Price Trend (5-day)
    short_return = df['close'] / df['close'].shift(5) - 1
    short_vol = df['close'].pct_change().rolling(window=5).std()
    
    # Medium-term Price Trend (20-day)
    medium_return = df['close'] / df['close'].shift(20) - 1
    medium_vol = df['close'].pct_change().rolling(window=20).std()
    
    # Multi-timeframe Alignment Score
    momentum_score = (short_return * medium_return) / (short_vol * medium_vol + 1e-8)
    
    # Volume Confirmation Signals
    # Volume Breakout Detection
    volume_median = df['volume'].rolling(window=20).median()
    volume_breakout = df['volume'] / volume_median.shift(1)
    
    # Volume Trend Consistency
    def calc_volume_slope(volume_series, window):
        slopes = []
        for i in range(len(volume_series)):
            if i < window - 1:
                slopes.append(np.nan)
            else:
                y = volume_series.iloc[i-window+1:i+1].values
                x = np.arange(window)
                slope, _, _, _, _ = stats.linregress(x, y)
                slopes.append(slope)
        return pd.Series(slopes, index=volume_series.index)
    
    volume_slope_5d = calc_volume_slope(df['volume'], 5)
    volume_slope_20d = calc_volume_slope(df['volume'], 20)
    volume_trend = (volume_slope_5d + volume_slope_20d) / 2
    
    # Volume-Price Divergence
    price_returns = df['close'].pct_change()
    volume_changes = df['volume'].pct_change()
    
    def rolling_correlation(series1, series2, window):
        corr = series1.rolling(window=window).corr(series2)
        return corr
    
    volume_price_corr = rolling_correlation(price_returns, volume_changes, 10)
    volume_divergence = np.where(volume_price_corr < 0, -volume_price_corr, volume_price_corr)
    
    # Combined Alpha Factor
    alpha_factor = (momentum_score * volume_breakout * 
                   volume_trend * volume_divergence)
    
    return alpha_factor
