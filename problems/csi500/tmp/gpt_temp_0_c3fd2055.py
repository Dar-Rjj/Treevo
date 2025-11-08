import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy import stats

def heuristics_v2(df):
    # Intraday Volatility to Volume Ratio
    intraday_vol = (df['high'] - df['low']) / df['close'].shift(1)
    volume_ma = df['volume'].shift(1).rolling(window=20, min_periods=10).mean()
    volume_ratio = np.log(df['volume'] / volume_ma)
    ivvr = intraday_vol / (volume_ratio + 1e-6)
    
    # Acceleration-Deceleration Momentum
    mom_5 = df['close'].pct_change(5)
    mom_10 = df['close'].pct_change(10)
    mom_diff = mom_5 - mom_10
    vol_20 = df['close'].pct_change().rolling(window=20, min_periods=10).std()
    adm = mom_diff / (vol_20 + 1e-6)
    
    # Volume-Price Divergence Factor
    def linear_reg_slope(series, window):
        slopes = pd.Series(index=series.index, dtype=float)
        for i in range(window-1, len(series)):
            if i >= window-1:
                y = series.iloc[i-window+1:i+1].values
                x = np.arange(len(y))
                slope, _, _, _, _ = stats.linregress(x, y)
                slopes.iloc[i] = slope
        return slopes
    
    price_slope = linear_reg_slope(df['close'], 5)
    volume_slope = linear_reg_slope(df['volume'], 5)
    vpdf = np.sign(price_slope * volume_slope)
    
    # Overnight Gap Persistence
    overnight_gap = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
    gap_ma = overnight_gap.abs().rolling(window=3, min_periods=2).mean()
    ogp = overnight_gap / (gap_ma + 1e-6)
    
    # High-Low Compression Indicator
    daily_range = (df['high'] - df['low']) / df['close'].shift(1)
    min_range = daily_range.rolling(window=10, min_periods=5).min()
    max_range = daily_range.rolling(window=10, min_periods=5).max()
    hlci = (daily_range - min_range) / (max_range - min_range + 1e-6)
    
    # Volume-Weighted Price Reversal
    price_reversal = -df['close'].pct_change(2)
    volume_std = df['volume'].rolling(window=20, min_periods=10).std()
    vwpr = price_reversal * (df['volume'] / (volume_std + 1e-6))
    
    # Intraday Efficiency Ratio
    net_movement = (df['close'] - df['open']).abs()
    total_movement = df['high'] - df['low']
    efficiency = net_movement / (total_movement + 1e-6)
    volume_percentile = df['volume'].rolling(window=20, min_periods=10).rank(pct=True)
    ier = efficiency * volume_percentile
    
    # Momentum Acceleration with Volume Confirmation
    mom_accel = mom_5 - mom_10
    volume_mom = df['volume'].pct_change(5)
    macv = mom_accel * np.sign(volume_mom)
    
    # Combine factors with equal weights
    factors = pd.DataFrame({
        'ivvr': ivvr,
        'adm': adm,
        'vpdf': vpdf,
        'ogp': ogp,
        'hlci': hlci,
        'vwpr': vwpr,
        'ier': ier,
        'macv': macv
    })
    
    # Standardize each factor
    factors_standardized = factors.apply(lambda x: (x - x.mean()) / x.std())
    
    # Equal-weighted combination
    final_factor = factors_standardized.mean(axis=1)
    
    return final_factor
