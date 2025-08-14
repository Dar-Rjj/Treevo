import numpy as np
def heuristics_v2(data):
    # Calculate the log return of the close price
    log_return = data['close'].apply(lambda x: np.log(x) - np.log(data['close'].shift(1)))
    
    # Calculate the spread between the high and low prices
    hl_spread = (data['high'] - data['low']) / data['close']
    
    # Calculate the ratio of close to open price
    co_ratio = data['close'] / data['open']
    
    # Calculate the ratio of volume to amount
    va_ratio = data['volume'] / data['amount']
    
    # Combine the above features using a weighted sum
    alpha_factor = 0.4 * log_return + 0.3 * hl_spread + 0.2 * co_ratio + 0.1 * va_ratio
    
    # Return the alpha factor as a pandas Series
    return alpha_factor
