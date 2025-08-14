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
    volume_mavg_10 = df['volume'].rolling(window=10).mean()
    volume_trend = np.where(df['volume'] > volume_mavg_10, 1, -1)
    
    # Integrate Volume Trend with Adjusted Range
    integrated_range = adjusted_high_low_range * volume_trend
    
    # Calculate Intraday Return
    intraday_return = (df['close'] - df['open']) / df['open']
    
    # Intraday Volatility Adjusted
    intraday_volatility = df['high'] - df['low']
    adjusted_intraday_return = intraday_return / intraday_volatility
    
    # Combine Adjusted Range, Intraday Return, and Volume Trend
    combined_factor = integrated_range + adjusted_intraday_return
    
    # Calculate 20-day Price Momentum
    price_momentum_20 = (df['close'] - df['close'].shift(20)) / df['close'].shift(20)
    
    # Identify Volume Shock
    volume_shock = np.where(df['volume'] > 1.5 * df['volume'].rolling(window=30).mean(), 1, 0)
    
    # Identify Amount Shock
    amount_shock = np.where(df['amount'] > 1.5 * df['amount'].rolling(window=30).mean(), 1, 0)
    
    # Combine High-Low Range and Close Price Momentum
    high_low_range_30 = df['high'] - df['low']
    close_momentum_30 = (df['close'] - df['close'].shift(30)) / df['close'].shift(30)
    combined_range_momentum = (high_low_range_30 + close_momentum_30) / 2
    
    # Integrate Momentum and Volume Shock
    integrated_momentum = combined_range_momentum * volume_shock
    
    # Adjust Intraday Return with Intraday Volatility
    adjusted_intraday_return_final = adjusted_intraday_return / intraday_volatility
    
    # Combine Intraday and Historical Factors
    volume_displacement = df['volume'] - df['volume'].shift(1)
    combined_intraday_historical = adjusted_intraday_return_final * volume_displacement
    
    # Incorporate Short-Term Price Reversal
    price_reversal_5 = (df['close'] - df['close'].shift(5)) / df['close'].shift(5)
    integrated_price_reversal = price_reversal_5 * volume_trend
    
    # Final Factor Combination
    final_factor = (combined_intraday_historical * integrated_momentum +
                    price_momentum_20 +
                    (df['volume'] - df['volume'].shift(10)) +
                    df['true_range'].rolling(window=20).mean() +
                    integrated_price_reversal)
    
    return final_factor
