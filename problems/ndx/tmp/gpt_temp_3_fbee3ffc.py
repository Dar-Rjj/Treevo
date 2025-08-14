import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Enhanced High-to-Low Range
    high_low_range = df['high'] - df['low']
    
    # Calculate Price Change (Close price of day t - Open price of day t-1)
    price_change = df['close'] - df['open'].shift(1)
    
    # Combine Enhanced High-to-Low Range with Price Change
    combined_value = 100 * high_low_range + price_change
    
    # Smoothing and Momentum
    smoothed_momentum = combined_value.ewm(span=7).mean()
    momentum_diff = smoothed_momentum - smoothed_momentum.shift(7)
    
    # Daily Log Return
    log_return = np.log(df['close'] / df['close'].shift(1))
    
    # Multi-Day Price Momentum
    price_return_3d = (df['close'] - df['close'].shift(3)) / df['close'].shift(3)
    price_return_7d = (df['close'] - df['close'].shift(7)) / df['close'].shift(7)
    price_return_21d = (df['close'] - df['close'].shift(21)) / df['close'].shift(21)
    
    # Weighted Average Momentum
    weighted_avg_momentum = (3 * price_return_3d + 1 * price_return_7d + 1 * price_return_21d) / 5
    
    # Volume Adjustment
    volume_pct_change = (df['volume'] - df['volume'].shift(1)) / df['volume'].shift(1)
    final_adjustment = weighted_avg_momentum * (1 + volume_pct_change)
    
    # Volume-Weighted Price Momentum
    close_wma_5d = df['close'].rolling(window=5).apply(lambda x: np.average(x, weights=df['volume']), raw=True)
    vwp_momentum = (df['close'] - close_wma_5d) * log_return
    
    # High-Frequency Volatility
    tr = df[['high', 'low', 'close']].diff(axis=1).max(axis=1)
    tr_sma_15d = tr.rolling(window=15).mean()
    
    # Adjust for Volatility
    adjusted_vwp_momentum = vwp_momentum / tr_sma_15d
    
    # Final Factor
    final_factor = momentum_diff + final_adjustment + adjusted_vwp_momentum
    
    return final_factor.dropna()

# Example usage:
# df = pd.read_csv('your_data.csv', parse_dates=True, index_col='date')
# factor_values = heuristics_v2(df)
# print(factor_values)
