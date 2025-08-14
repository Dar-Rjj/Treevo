import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Enhanced High-to-Low Range
    high_low_range = df['high'] - df['low']
    
    # Adjust for Volume
    adjusted_high_low_range = high_low_range * np.sqrt(df['volume'])
    
    # Evaluate Volume Trend
    volume_moving_avg = df['volume'].rolling(window=10).mean()
    volume_trend = np.where(df['volume'] > volume_moving_avg, 1, -1)
    
    # Integrate Volume Trend with Adjusted Range
    integrated_adjusted_range = adjusted_high_low_range * volume_trend
    
    # Calculate Intraday Return
    intraday_return = (df['close'] - df['open']) / df['open']
    
    # Intraday Volatility Adjusted
    intraday_volatility = df['high'] - df['low']
    adjusted_intraday_return = intraday_return / intraday_volatility
    
    # Combine Adjusted Range, Intraday Return, and Volume Trend
    combined_factor_1 = integrated_adjusted_range + adjusted_intraday_return
    
    # Calculate 20-day Price Momentum
    price_momentum_20 = (df['close'] - df['close'].shift(20)) / df['close'].shift(20)
    
    # Identify Volume Shock
    avg_volume_30 = df['volume'].rolling(window=30).mean()
    volume_shock = np.where(df['volume'] > 1.5 * avg_volume_30, 1, 0)
    
    # Identify Amount Shock
    avg_amount_30 = df['amount'].rolling(window=30).mean()
    amount_shock = np.where(df['amount'] > 1.5 * avg_amount_30, 1, 0)
    
    # Combine High-Low Range and Close Price Momentum
    close_price_momentum_30 = (df['close'] - df['close'].shift(30)) / df['close'].shift(30)
    combined_high_low_momentum = (high_low_range + close_price_momentum_30) / 2
    
    # Integrate Momentum and Volume Shock
    integrated_momentum_volume_shock = combined_high_low_momentum * volume_shock
    
    # Adjust Intraday Return with Intraday Volatility
    adjusted_intraday_return_volatility = adjusted_intraday_return * intraday_volatility
    
    # Combine Intraday and Historical Factors
    volume_displacement = df['volume'] - df['volume'].shift(1)
    combined_intraday_historical = adjusted_intraday_return * volume_displacement
    
    # Final Factor Combination
    final_factor = (
        combined_factor_1 * integrated_momentum_volume_shock +
        price_momentum_20 +
        (df['volume'] - df['volume'].shift(10)) +
        df['high'].rolling(window=20).apply(lambda x: np.mean(np.abs(x - x.shift())), raw=False)
    )
    
    return final_factor
