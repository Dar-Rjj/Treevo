import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # High-Low Volatility Adjusted Momentum
    # 5-day return divided by 10-day rolling std of High-Low range
    returns_5d = data['close'].pct_change(5)
    hl_range = (data['high'] - data['low']) / data['close']
    hl_volatility = hl_range.rolling(window=10, min_periods=5).std()
    factor1 = returns_5d / (hl_volatility + 1e-8)
    
    # Volume-Weighted Price Acceleration
    # Second derivative of Close multiplied by Volume
    price_change = data['close'].pct_change()
    price_acceleration = price_change.diff()
    factor2 = price_acceleration * data['volume']
    
    # Relative Strength with Volume Confirmation
    # Close deviation from 20-day median multiplied by Volume deviation from 20-day average
    close_median = data['close'].rolling(window=20, min_periods=10).median()
    close_deviation = (data['close'] - close_median) / close_median
    volume_avg = data['volume'].rolling(window=20, min_periods=10).mean()
    volume_deviation = (data['volume'] - volume_avg) / (volume_avg + 1e-8)
    factor3 = close_deviation * volume_deviation
    
    # Opening Gap Persistence Factor
    # Open/Previous Close gap multiplied by consecutive same-sign gap count
    gap = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
    gap_sign = np.sign(gap)
    
    # Calculate consecutive same-sign gaps
    consecutive_count = pd.Series(0, index=data.index)
    for i in range(1, len(data)):
        if gap_sign.iloc[i] == gap_sign.iloc[i-1] and not pd.isna(gap_sign.iloc[i]) and not pd.isna(gap_sign.iloc[i-1]):
            consecutive_count.iloc[i] = consecutive_count.iloc[i-1] + 1
        else:
            consecutive_count.iloc[i] = 1
    
    factor4 = gap * consecutive_count
    
    # Amount-Based Order Flow Imbalance
    # Daily average trade size deviation from rolling median
    avg_trade_size = data['amount'] / (data['volume'] + 1e-8)
    trade_size_median = avg_trade_size.rolling(window=20, min_periods=10).median()
    factor5 = (avg_trade_size - trade_size_median) / (trade_size_median + 1e-8)
    
    # Volatility-Regime Adjusted Reversal
    # Reverse negative returns in high volatility, continue positive returns otherwise
    daily_return = data['close'].pct_change()
    hl_range_pct = (data['high'] - data['low']) / data['close']
    vol_regime = hl_range_pct.rolling(window=20, min_periods=10).apply(
        lambda x: 1 if x.iloc[-1] > np.percentile(x.dropna(), 70) else 0
    )
    
    factor6 = pd.Series(0, index=data.index)
    for i in range(len(data)):
        if not pd.isna(vol_regime.iloc[i]) and not pd.isna(daily_return.iloc[i]):
            if vol_regime.iloc[i] == 1 and daily_return.iloc[i] < 0:
                factor6.iloc[i] = -daily_return.iloc[i]  # Reverse negative returns
            else:
                factor6.iloc[i] = daily_return.iloc[i]  # Continue positive returns
    
    # Volume-Weighted Support/Resistance Breakout
    # Close breakthrough of 20-day High/Low levels multiplied by Volume ratio
    rolling_high = data['high'].rolling(window=20, min_periods=10).max()
    rolling_low = data['low'].rolling(window=20, min_periods=10).min()
    volume_avg_20d = data['volume'].rolling(window=20, min_periods=10).mean()
    
    breakout_signal = pd.Series(0, index=data.index)
    for i in range(len(data)):
        if not pd.isna(rolling_high.iloc[i]) and not pd.isna(rolling_low.iloc[i]):
            if data['close'].iloc[i] > rolling_high.iloc[i]:
                breakout_signal.iloc[i] = 1
            elif data['close'].iloc[i] < rolling_low.iloc[i]:
                breakout_signal.iloc[i] = -1
    
    volume_ratio = data['volume'] / (volume_avg_20d + 1e-8)
    factor7 = breakout_signal * volume_ratio
    
    # Intraday Strength Persistence Factor
    # Intraday strength (Close relative to High and Low) with weighted consecutive count
    intraday_strength = (data['close'] - data['low']) / (data['high'] - data['low'] + 1e-8) - 0.5
    
    # Weighted consecutive count
    weighted_count = pd.Series(0, index=data.index)
    for i in range(1, len(data)):
        if intraday_strength.iloc[i] * intraday_strength.iloc[i-1] > 0:
            weighted_count.iloc[i] = weighted_count.iloc[i-1] + abs(intraday_strength.iloc[i])
        else:
            weighted_count.iloc[i] = abs(intraday_strength.iloc[i])
    
    factor8 = intraday_strength * weighted_count
    
    # Volume-Expansion Price Efficiency
    # Price efficiency (absolute return / High-Low range) multiplied by Volume ratio
    price_efficiency = abs(data['close'].pct_change()) / (hl_range + 1e-8)
    volume_ratio_10d = data['volume'] / data['volume'].rolling(window=10, min_periods=5).mean()
    factor9 = price_efficiency * volume_ratio_10d
    
    # Multi-Timeframe Momentum Convergence
    # Short-term (3-day) and medium-term (10-day) returns interaction
    returns_3d = data['close'].pct_change(3)
    returns_10d = data['close'].pct_change(10)
    
    factor10 = pd.Series(0, index=data.index)
    for i in range(len(data)):
        if not pd.isna(returns_3d.iloc[i]) and not pd.isna(returns_10d.iloc[i]):
            if returns_3d.iloc[i] * returns_10d.iloc[i] > 0:
                factor10.iloc[i] = returns_3d.iloc[i] * returns_10d.iloc[i]  # Multiply when same direction
            else:
                factor10.iloc[i] = -abs(returns_3d.iloc[i] * returns_10d.iloc[i])  # Penalize when opposite
    
    # Combine all factors with equal weighting
    factors = [factor1, factor2, factor3, factor4, factor5, factor6, factor7, factor8, factor9, factor10]
    
    # Standardize each factor and combine
    combined_factor = pd.Series(0, index=data.index)
    for factor in factors:
        factor_standardized = (factor - factor.mean()) / (factor.std() + 1e-8)
        combined_factor = combined_factor + factor_standardized
    
    # Final standardization
    final_factor = (combined_factor - combined_factor.mean()) / (combined_factor.std() + 1e-8)
    
    return final_factor
