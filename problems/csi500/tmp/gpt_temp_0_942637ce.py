import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate Intraday Range
    intraday_range = df['high'] - df['low']
    
    # Compute Intraday Momentum
    intraday_momentum = df['high'] - df['open']
    
    # Aggregate Close-to-Previous Close Return
    close_to_prev_close_return = df['close'] - df['close'].shift(1)
    
    # Determine Relative Strength
    relative_strength = df['close'] / df['close'].shift(1)
    
    # Measure Intraday Volatility
    intraday_volatility = (df['high'] - df['low']).std()
    
    # Incorporate Liquidity
    liquidity = df['volume'] + 0.5 * df['amount']
    
    # Combine Factors
    combined_factors = (intraday_range + 
                        intraday_momentum + 
                        close_to_prev_close_return) * relative_strength
    
    # Handle Division by Zero in Intraday Volatility
    combined_factors = combined_factors / intraday_volatility
    combined_factors[intraday_volatility == 0] = 0
    
    # Integrate Volume Synchronization
    volume_difference = df['volume'].diff().fillna(0)
    volume_ratio = df['volume'] / df['volume'].shift(1).fillna(1)
    volume_synchronization = volume_difference * volume_ratio
    
    # Final Alpha Factor
    final_alpha_factor = combined_factors * volume_synchronization
    final_alpha_factor[volume_synchronization == 0] = 0
    
    return final_alpha_factor
