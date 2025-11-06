import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    data = df.copy()
    
    # Volatility-Adjusted Momentum with Volume Confirmation
    # Calculate 5-day return
    ret_5d = data['close'] / data['close'].shift(5) - 1
    
    # Compute 5-day rolling standard deviation of (High_t - Low_t)
    hl_range = data['high'] - data['low']
    vol_5d = hl_range.rolling(window=5).std()
    
    # Calculate volume acceleration
    vol_acc = data['volume'] / data['volume'].shift(5)
    
    # Alpha factor 1
    alpha1 = ret_5d / vol_5d * vol_acc
    
    # Volume-Weighted Price Acceleration with Regime Detection
    # Calculate price acceleration
    ret_1d = data['close'] / data['close'].shift(1) - 1
    price_acc = ret_1d - ret_1d.shift(1)
    
    # Identify volatility regime using High-Low range percentiles over 20 days
    hl_range_20d = hl_range.rolling(window=20)
    vol_regime = hl_range_20d.apply(lambda x: pd.qcut(x, 4, labels=False, duplicates='drop').iloc[-1] if len(x.dropna()) == 20 else np.nan)
    
    # Multiply acceleration by Volume_t and apply conditional weighting
    vol_weighted_acc = price_acc * data['volume']
    # High volatility regime gets higher weight (regime 3 gets 2x, regime 2 gets 1.5x, regime 1 gets 1x, regime 0 gets 0.5x)
    regime_weights = vol_regime.map({0: 0.5, 1: 1.0, 2: 1.5, 3: 2.0})
    alpha2 = vol_weighted_acc * regime_weights
    
    # Gap Persistence with Order Flow
    # Calculate daily gap
    daily_gap = data['open'] / data['close'].shift(1) - 1
    
    # Count consecutive same-sign gaps
    gap_sign = np.sign(daily_gap)
    consecutive_gaps = gap_sign.groupby((gap_sign != gap_sign.shift(1)).cumsum()).cumcount() + 1
    
    # Compute daily average trade size and deviation from 10-day rolling median
    trade_size = data['amount'] / data['volume']
    trade_size_dev = trade_size / trade_size.rolling(window=10).median() - 1
    
    # Alpha factor 3
    alpha3 = daily_gap * consecutive_gaps * trade_size_dev
    
    # Multi-Timeframe Breakout with Volume Expansion
    # Detect Close price breakthrough of 20-day rolling High
    rolling_high_20d = data['high'].rolling(window=20).max().shift(1)
    breakout_signal = (data['close'] > rolling_high_20d).astype(int)
    
    # Calculate short-term (3-day) and medium-term (10-day) returns
    ret_3d = data['close'] / data['close'].shift(3) - 1
    ret_10d = data['close'] / data['close'].shift(10) - 1
    
    # Compute volume ratio
    vol_avg_10d = data['volume'].rolling(window=10).mean().shift(1)
    vol_ratio = data['volume'] / vol_avg_10d
    
    # Calculate price efficiency
    daily_ret_abs = abs(data['close'] / data['close'].shift(1) - 1)
    price_efficiency = daily_ret_abs / (data['high'] - data['low'])
    
    # Alpha factor 4
    alpha4 = breakout_signal * (ret_3d * ret_10d) * vol_ratio * price_efficiency
    
    # Relative Strength with Intraday Persistence
    # Calculate Close price deviation from 20-day rolling median
    price_median_20d = data['close'].rolling(window=20).median()
    price_dev = data['close'] / price_median_20d - 1
    
    # Compute intraday strength
    intraday_strength = (data['close'] - data['low']) / (data['high'] - data['low'])
    
    # Apply weighted count of consecutive strong days (intraday strength > 0.6)
    strong_day = (intraday_strength > 0.6).astype(int)
    consecutive_strong = strong_day.groupby((strong_day != strong_day.shift(1)).cumsum()).cumcount() + 1
    weighted_strong = consecutive_strong * strong_day
    
    # Calculate volume deviation from 20-day rolling average
    vol_avg_20d = data['volume'].rolling(window=20).mean()
    vol_dev = data['volume'] / vol_avg_20d - 1
    
    # Alpha factor 5
    alpha5 = price_dev * weighted_strong * vol_dev
    
    # Combine all alpha factors with equal weights
    combined_alpha = (alpha1.fillna(0) + alpha2.fillna(0) + alpha3.fillna(0) + 
                     alpha4.fillna(0) + alpha5.fillna(0)) / 5
    
    return combined_alpha
