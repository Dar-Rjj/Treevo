import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(data):
    df = data.copy()
    
    # Momentum-Adjusted Volume Breakout
    # Calculate 5-day price momentum (t-5 to t-1)
    momentum = (df['close'].shift(1) - df['close'].shift(5)) / df['close'].shift(5)
    
    # Calculate volume deviation from 10-day median
    vol_median = df['volume'].rolling(window=10, min_periods=5).median()
    vol_deviation = (df['volume'] - vol_median) / vol_median
    
    # Combine momentum and volume signals
    factor1 = np.sign(momentum) * vol_deviation
    
    # Volatility-Scaled Return Reversal
    # Calculate 2-day return
    ret_2d = (df['close'] - df['close'].shift(2)) / df['close'].shift(2)
    
    # Calculate 10-day Average True Range
    tr1 = df['high'] - df['low']
    tr2 = abs(df['high'] - df['close'].shift(1))
    tr3 = abs(df['low'] - df['close'].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=10, min_periods=5).mean()
    
    # Scale returns by volatility with reversal effect
    factor2 = -ret_2d / atr
    
    # Intraday Strength Persistence
    # Calculate intraday strength
    intraday_strength = (df['close'] - df['low']) / (df['high'] - df['low']).replace(0, np.nan)
    
    # Calculate 5-day strength trend slope
    def calc_slope(series):
        if len(series) < 2:
            return np.nan
        x = np.arange(len(series))
        return np.polyfit(x, series, 1)[0]
    
    strength_trend = intraday_strength.rolling(window=5, min_periods=3).apply(calc_slope, raw=False)
    
    # Combine with volume confirmation
    factor3 = np.sqrt(abs(strength_trend * df['volume'])) * np.sign(strength_trend)
    
    # Relative Volume Price Efficiency
    # Calculate daily price efficiency ratio
    daily_efficiency = (df['close'] - df['open']) / (df['high'] - df['low']).replace(0, np.nan)
    efficiency_ratio = daily_efficiency.rolling(window=20, min_periods=10).mean()
    
    # Calculate volume percentile rank
    vol_percentile = df['volume'].rolling(window=50, min_periods=25).rank(pct=True)
    
    # Combine efficiency and volume
    factor4 = np.log(abs(efficiency_ratio * vol_percentile) + 1e-6) * np.sign(efficiency_ratio)
    
    # Acceleration-Deceleration Divergence
    # Calculate price acceleration (second derivative)
    price_roc = df['close'].pct_change()
    price_acceleration = price_roc.diff().rolling(window=5, min_periods=3).mean()
    
    # Calculate volume deceleration
    vol_roc = df['volume'].pct_change()
    vol_deceleration = -vol_roc.diff().rolling(window=5, min_periods=3).mean()
    
    # Calculate rolling correlation
    rolling_corr = price_roc.rolling(window=10, min_periods=5).corr(vol_roc)
    
    # Detect divergence pattern
    factor5 = price_acceleration * vol_deceleration * rolling_corr
    
    # Bid-Ask Spread Implied Alpha
    # Estimate effective spread
    spread_proxy = 2 * (df['high'] - df['close']) / (df['high'] - df['low']).replace(0, np.nan)
    
    # Calculate spread momentum
    spread_momentum = (spread_proxy - spread_proxy.shift(5)) / spread_proxy.shift(5)
    
    # Calculate 1-day return
    ret_1d = df['close'].pct_change()
    
    # Combine with price movement
    factor6 = np.cbrt(ret_1d * -spread_momentum)
    
    # Opening Gap Volume Confirmation
    # Calculate opening gap percentage
    gap_pct = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
    
    # Calculate volume ratio to 20-day average
    vol_ratio = df['volume'] / df['volume'].rolling(window=20, min_periods=10).mean()
    
    # Weight gap by volume support
    factor7 = np.tanh(gap_pct * vol_ratio)
    
    # Multi-timeframe Momentum Convergence
    # Calculate short-term momentum (3-day)
    mom_short = (df['close'] - df['close'].shift(3)) / df['close'].shift(3)
    
    # Calculate medium-term momentum (10-day)
    mom_medium = (df['close'] - df['close'].shift(10)) / df['close'].shift(10)
    
    # Detect convergence pattern
    denominator = abs(mom_short) + abs(mom_medium)
    factor8 = (mom_short * mom_medium) / denominator.replace(0, np.nan)
    
    # Volume-Weighted Price Range Efficiency
    # Calculate normalized price range
    norm_range = (df['high'] - df['low']) / ((df['high'] + df['low']) / 2)
    
    # Calculate volume-weighted efficiency
    vw_efficiency = (df['close'] - df['open']) * df['volume'] / (df['high'] - df['low']).replace(0, np.nan)
    
    # Combine range and efficiency
    factor9 = np.sqrt(abs(norm_range * vw_efficiency)) * np.sign(vw_efficiency)
    
    # Combine all factors with equal weights
    factors = pd.DataFrame({
        'f1': factor1, 'f2': factor2, 'f3': factor3, 'f4': factor4,
        'f5': factor5, 'f6': factor6, 'f7': factor7, 'f8': factor8, 'f9': factor9
    })
    
    # Z-score normalization for each factor
    factors_normalized = factors.apply(lambda x: (x - x.rolling(window=50, min_periods=25).mean()) / 
                                     x.rolling(window=50, min_periods=25).std())
    
    # Equal-weighted combination
    final_factor = factors_normalized.mean(axis=1)
    
    return final_factor
