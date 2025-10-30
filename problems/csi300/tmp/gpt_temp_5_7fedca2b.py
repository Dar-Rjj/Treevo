import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy import stats

def heuristics_v2(df):
    """
    Generate alpha factor combining price acceleration, volume-price divergence,
    volatility-adjusted range breakout, and multi-timeframe price structure.
    """
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Price Acceleration Factor
    # Short-term momentum (5-day)
    ret_5d = data['close'].pct_change(5)
    ret_10d_prev = (data['close'].shift(5) - data['close'].shift(10)) / data['close'].shift(10)
    short_term_accel = ret_5d - ret_10d_prev
    
    # Medium-term momentum (20-day)
    ret_20d = data['close'].pct_change(20)
    ret_40d_prev = (data['close'].shift(20) - data['close'].shift(40)) / data['close'].shift(40)
    medium_term_accel = ret_20d - ret_40d_prev
    
    # Combine acceleration signals
    price_accel = 0.6 * short_term_accel + 0.4 * medium_term_accel
    
    # Volume-Price Divergence
    window = 10
    
    def calc_slope_r2(series, window):
        """Calculate slope and R-squared for a series over rolling window"""
        slopes = []
        r_squared = []
        
        for i in range(len(series)):
            if i < window - 1:
                slopes.append(np.nan)
                r_squared.append(np.nan)
                continue
                
            y = series.iloc[i-window+1:i+1].values
            x = np.arange(window)
            
            if len(y) == window:
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
                slopes.append(slope)
                r_squared.append(r_value**2)
            else:
                slopes.append(np.nan)
                r_squared.append(np.nan)
                
        return pd.Series(slopes, index=series.index), pd.Series(r_squared, index=series.index)
    
    # Price trend analysis
    price_slope, price_r2 = calc_slope_r2(data['close'], window)
    
    # Volume trend analysis
    volume_slope, volume_r2 = calc_slope_r2(data['volume'], window)
    
    # Volume-price ratio trend
    vpr = data['volume'] / data['close']
    vpr_slope, vpr_r2 = calc_slope_r2(vpr, window)
    
    # Detect divergence
    price_volume_divergence = np.where(
        (price_r2 > 0.7) & (volume_r2 < 0.3) & (price_slope * volume_slope < 0),
        -price_slope * (price_r2 - volume_r2),  # Negative when trends diverge
        0
    )
    
    # Volatility-Adjusted Range Breakout
    # Recent price range analysis
    def true_range(high, low, close_prev):
        return np.maximum(high - low, np.maximum(abs(high - close_prev), abs(low - close_prev)))
    
    tr = true_range(data['high'], data['low'], data['close'].shift(1))
    atr_10d = tr.rolling(window=10).mean()
    atr_20d = tr.rolling(window=20).mean()
    
    current_range_pct = (data['high'] - data['low']) / data['close']
    avg_range_pct_10d = current_range_pct.rolling(window=10).mean()
    
    # Breakout detection
    high_3d_max = data['high'].rolling(window=3).max().shift(1)
    volume_20d_avg = data['volume'].rolling(window=20).mean()
    
    breakout_signal = np.where(
        (data['close'] > high_3d_max) & 
        (current_range_pct > avg_range_pct_10d) & 
        (data['volume'] > volume_20d_avg),
        current_range_pct * (data['close'] / high_3d_max - 1),
        0
    )
    
    # Volatility scaling
    volatility_adj_breakout = breakout_signal / (atr_20d / data['close'])
    
    # Multi-Timeframe Price Structure
    # Short-term structure (5-day)
    high_5d = data['high'].rolling(window=5).max()
    low_5d = data['low'].rolling(window=5).min()
    days_since_high_5d = (data['high'].rolling(window=5, min_periods=1).apply(
        lambda x: len(x) - np.argmax(x) - 1, raw=True))
    days_since_low_5d = (data['low'].rolling(window=5, min_periods=1).apply(
        lambda x: len(x) - np.argmin(x) - 1, raw=True))
    price_pos_5d = (data['close'] - low_5d) / (high_5d - low_5d)
    
    # Medium-term structure (20-day)
    high_20d = data['high'].rolling(window=20).max()
    low_20d = data['low'].rolling(window=20).min()
    days_since_high_20d = (data['high'].rolling(window=20, min_periods=1).apply(
        lambda x: len(x) - np.argmax(x) - 1, raw=True))
    days_since_low_20d = (data['low'].rolling(window=20, min_periods=1).apply(
        lambda x: len(x) - np.argmin(x) - 1, raw=True))
    price_pos_20d = (data['close'] - low_20d) / (high_20d - low_20d)
    
    # Structure convergence
    range_5d = high_5d - low_5d
    range_20d = high_20d - low_20d
    range_compression = (range_5d / data['close']) / (range_20d / data['close'])
    
    structure_signal = np.where(
        (range_compression < 0.6) & (price_pos_5d > 0.8) & (price_pos_20d > 0.8),
        1,  # Strong bullish structure
        np.where(
            (range_compression < 0.6) & (price_pos_5d < 0.2) & (price_pos_20d < 0.2),
            -1,  # Strong bearish structure
            (price_pos_5d - 0.5) + (price_pos_20d - 0.5)  # Neutral structure score
        )
    )
    
    # Alpha Factor Combination
    # Equal weighting initially
    alpha_score = (
        price_accel.fillna(0) +
        price_volume_divergence +
        volatility_adj_breakout.fillna(0) +
        structure_signal
    ) / 4
    
    return pd.Series(alpha_score, index=data.index, name='alpha_factor')
