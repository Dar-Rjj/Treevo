import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Compute Daily Intraday Returns (Close - Open)
    intraday_returns = df['close'] - df['open']
    
    # Calculate 14-Day Volume-Weighted Intraday Return
    volume_weighted_intraday_return = intraday_returns.rolling(window=14).apply(
        lambda x: np.dot(x, df.loc[x.index, 'volume']) / df.loc[x.index, 'volume'].sum(), raw=False
    )
    
    # Compute Intraday High-Low Spread (High - Low)
    intraday_high_low_spread = df['high'] - df['low']
    
    # Compute Close-to-Open Return (Close - Open)
    close_to_open_return = df['close'] - df['open']
    
    # Combine: (Close-to-Open Return / Intraday High-Low Spread) * Intraday High-Low Spread
    intraday_reversal = (close_to_open_return / intraday_high_low_spread) * intraday_high_low_spread
    
    # Enhance with Volume-Weighted High-Low Difference
    high_low_diff = df['high'] - df['low']
    volume_weighted_high_low_diff = high_low_diff.rolling(window=14).apply(
        lambda x: np.dot(x, df.loc[x.index, 'volume']) / df.loc[x.index, 'volume'].sum(), raw=False
    )
    
    # Incorporate Volume and Amount Influence
    avg_volume = df['volume'].rolling(window=14).mean()
    intraday_volume_impact = df['volume'] / avg_volume
    amount_impact = df['amount'] / avg_volume
    combined_volume_amount_impact = intraday_volume_impact + amount_impact
    enhanced_reversal = intraday_reversal * combined_volume_amount_impact
    
    # Adjust for Volatility
    daily_log_returns = np.log(df['close'] / df['close'].shift(1))
    realized_volatility = daily_log_returns.rolling(window=14).std()
    volatility_adjusted_reversal = enhanced_reversal / realized_volatility
    
    # Introduce Lagged Close-to-Open Return
    lagged_close_to_open_return = (df['close'].shift(1) - df['open'].shift(1)).apply(np.sign)
    
    # Adjust for Lagged Momentum
    close_to_close_return = df['close'] - df['close'].shift(1)
    adjusted_lagged_momentum = lagged_close_to_open_return * close_to_close_return
    
    # Integrate Price Range Ratio
    intraday_price_range = df['high'] - df['low']
    long_term_price_range = df['high'].rolling(window=20).max() - df['low'].rolling(window=20).min()
    price_range_ratio = intraday_price_range / long_term_price_range
    
    # Adjust for Price Range Ratio
    adjusted_return_with_momentum = adjusted_lagged_momentum * price_range_ratio
    
    # Adjust Close-to-Close Return by Intraday Volatility
    intraday_volatility = (df['high'] - df['low']).rolling(window=14).sum()
    adjusted_close_to_close_return = close_to_close_return / intraday_volatility
    
    # Combine Adjusted Return and Volume-Weighted Intraday Return
    combined_factor = adjusted_return_with_momentum + volume_weighted_intraday_return
    
    # Synthesize Combined Factor
    final_factor = combined_factor + adjusted_close_to_close_return
    
    return final_factor.dropna()
