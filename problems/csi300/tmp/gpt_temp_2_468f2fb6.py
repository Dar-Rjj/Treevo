import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Intraday Momentum Acceleration
    intraday_momentum = (df['high'] - df['low']) / df['close']
    prev_intraday_momentum = intraday_momentum.shift(1)
    momentum_change = intraday_momentum - prev_intraday_momentum
    volume_ratio = df['volume'] / df['volume'].shift(1)
    factor1 = momentum_change * volume_ratio
    
    # Volume-Adjusted Price Reversal
    price_reversal = df['close'] / df['close'].shift(1) - 1
    volume_median = df['volume'].rolling(window=20, min_periods=1).median()
    volume_surge = df['volume'] / volume_median
    factor2 = price_reversal * volume_surge
    
    # Relative Strength Volatility Breakout
    # Assuming market index close is not available, using close price as proxy
    stock_return = df['close'].pct_change()
    market_return = df['close'].pct_change()  # Using same as proxy
    relative_strength = stock_return - market_return
    
    true_range = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            abs(df['high'] - df['close'].shift(1)),
            abs(df['low'] - df['close'].shift(1))
        )
    )
    volatility = true_range.rolling(window=10, min_periods=1).std()
    breakout_signal = (true_range > volatility).astype(float)
    factor3 = relative_strength * breakout_signal
    
    # Amount-Based Order Flow Imbalance
    avg_trade_size = df['amount'] / df['volume']
    rolling_avg_trade = avg_trade_size.rolling(window=5, min_periods=1).mean()
    order_imbalance = avg_trade_size - rolling_avg_trade
    intraday_return = df['close'] / df['open'] - 1
    factor4 = order_imbalance * intraday_return
    
    # High-Low Compression Signal
    daily_range_pct = (df['high'] - df['low']) / df['close']
    min_range = daily_range_pct.rolling(window=15, min_periods=1).min()
    compression_ratio = daily_range_pct / min_range
    
    # Calculate volume slope using linear regression
    def volume_slope(volumes):
        if len(volumes) < 2:
            return 0
        x = np.arange(len(volumes))
        slope = np.polyfit(x, volumes, 1)[0]
        return slope
    
    volume_trend = df['volume'].rolling(window=5, min_periods=1).apply(
        volume_slope, raw=False
    )
    factor5 = compression_ratio * volume_trend
    
    # Open-Gap Volume Confirmation
    opening_gap = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
    avg_volume = df['volume'].rolling(window=10, min_periods=1).mean()
    volume_ratio_gap = df['volume'] / avg_volume
    factor6 = opening_gap * volume_ratio_gap
    
    # Close-to-Close Momentum Persistence
    returns_5d = df['close'].pct_change(5)
    
    def count_consecutive_signs(returns):
        if len(returns) < 2:
            return 0
        signs = np.sign(returns)
        current_sign = signs.iloc[-1]
        count = 0
        for i in range(len(signs)-1, -1, -1):
            if signs.iloc[i] == current_sign and signs.iloc[i] != 0:
                count += 1
            else:
                break
        return count
    
    persistence_score = returns_5d.rolling(window=10, min_periods=1).apply(
        count_consecutive_signs, raw=False
    )
    avg_volume_5d = df['volume'].rolling(window=5, min_periods=1).mean()
    volume_weight = df['volume'] / avg_volume_5d
    factor7 = returns_5d * persistence_score * volume_weight
    
    # Volume-Weighted Price Acceleration
    returns = df['close'].pct_change()
    price_acceleration = returns.diff()  # Second derivative approximation
    
    def volume_percentile(volumes):
        if len(volumes) < 2:
            return 0.5
        current_vol = volumes.iloc[-1]
        rank = (volumes < current_vol).sum() / len(volumes)
        return rank
    
    volume_rank = df['volume'].rolling(window=20, min_periods=1).apply(
        volume_percentile, raw=False
    )
    factor8 = price_acceleration * volume_rank
    
    # Combine all factors with equal weights
    factors = pd.DataFrame({
        'f1': factor1,
        'f2': factor2,
        'f3': factor3,
        'f4': factor4,
        'f5': factor5,
        'f6': factor6,
        'f7': factor7,
        'f8': factor8
    })
    
    # Remove any infinite values and fill NaN
    factors = factors.replace([np.inf, -np.inf], np.nan)
    factors = factors.fillna(0)
    
    # Z-score normalization for each factor
    factors_normalized = factors.apply(lambda x: (x - x.mean()) / x.std())
    
    # Equal-weighted combination
    combined_factor = factors_normalized.mean(axis=1)
    
    return combined_factor
