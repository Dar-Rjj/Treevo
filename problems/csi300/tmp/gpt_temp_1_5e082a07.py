import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Intraday Momentum Acceleration
    intraday_momentum = (df['high'] - df['low']) / df['close']
    intraday_momentum_accel = (intraday_momentum - intraday_momentum.shift(1)) * (df['volume'] / df['volume'].shift(1))
    
    # Volume-Adjusted Price Reversal
    price_reversal = df['close'] / df['close'].shift(1) - 1
    volume_median = df['volume'].rolling(window=20, min_periods=1).median()
    volume_adjusted_reversal = price_reversal * (df['volume'] / volume_median)
    
    # Relative Strength Volatility Breakout
    relative_strength = (df['close'] / df['close'].shift(1))
    true_range = pd.DataFrame({
        'hl': df['high'] - df['low'],
        'hcp': abs(df['high'] - df['close'].shift(1)),
        'lcp': abs(df['low'] - df['close'].shift(1))
    }).max(axis=1)
    true_range_std = true_range.rolling(window=10, min_periods=1).std()
    volatility_breakout = relative_strength * (true_range > true_range_std)
    
    # Amount-Based Order Flow Imbalance
    avg_trade_size = df['amount'] / df['volume']
    avg_trade_size_mean = avg_trade_size.rolling(window=5, min_periods=1).mean()
    order_flow_imbalance = (avg_trade_size / avg_trade_size_mean) * (df['close'] / df['open'] - 1)
    
    # High-Low Compression Signal
    range_pct = (df['high'] - df['low']) / df['close']
    min_range = range_pct.rolling(window=15, min_periods=1).min()
    volume_slope = df['volume'].rolling(window=5, min_periods=1).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0, raw=False
    )
    compression_signal = (range_pct / min_range) * volume_slope
    
    # Open-Gap Volume Confirmation
    gap_pct = df['open'] / df['close'].shift(1) - 1
    volume_mean = df['volume'].rolling(window=10, min_periods=1).mean()
    gap_confirmation = gap_pct * (df['volume'] / volume_mean)
    
    # Close-to-Close Momentum Persistence
    return_5d = df['close'] / df['close'].shift(5) - 1
    daily_return = df['close'] / df['close'].shift(1) - 1
    
    def count_consecutive(series):
        count = pd.Series(index=series.index, dtype=float)
        current_count = 0
        current_sign = 0
        
        for i, val in enumerate(series):
            if pd.isna(val):
                count.iloc[i] = np.nan
                current_count = 0
                current_sign = 0
            else:
                sign = 1 if val > 0 else (-1 if val < 0 else 0)
                if sign == current_sign and current_sign != 0:
                    current_count += 1
                else:
                    current_count = 1 if sign != 0 else 0
                    current_sign = sign
                count.iloc[i] = current_count
        return count
    
    consecutive_count = count_consecutive(daily_return)
    volume_mean_5 = df['volume'].rolling(window=5, min_periods=1).mean()
    momentum_persistence = consecutive_count * (df['volume'] / volume_mean_5)
    
    # Volume-Weighted Price Acceleration
    return_t = df['close'] / df['close'].shift(1) - 1
    acceleration = return_t - return_t.shift(1)
    
    def rolling_percentile(series, window, percentile):
        return series.rolling(window=window, min_periods=1).apply(
            lambda x: np.percentile(x, percentile) if len(x) > 0 else np.nan, raw=True
        )
    
    volume_percentile = rolling_percentile(df['volume'], 20, 20)
    volume_weighted_accel = acceleration * volume_percentile
    
    # Combine all factors with equal weights
    factors = pd.DataFrame({
        'factor1': intraday_momentum_accel,
        'factor2': volume_adjusted_reversal,
        'factor3': volatility_breakout,
        'factor4': order_flow_imbalance,
        'factor5': compression_signal,
        'factor6': gap_confirmation,
        'factor7': momentum_persistence,
        'factor8': volume_weighted_accel
    })
    
    # Z-score normalization for each factor
    factors_normalized = factors.apply(lambda x: (x - x.mean()) / x.std())
    
    # Equal-weighted combination
    combined_factor = factors_normalized.mean(axis=1)
    
    return combined_factor
