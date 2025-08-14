import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate the price range (High - Low)
    df['price_range'] = df['high'] - df['low']
    
    # Compute the True Range
    df['true_range'] = df[['high', 'low']].apply(lambda x: max(x['high'] - x['low'], 
                                                               abs(x['high'] - df['close'].shift(1)), 
                                                               abs(x['low'] - df['close'].shift(1))), axis=1)
    
    # Calculate the Average True Range (ATR) over a period
    atr_period = 14
    df['atr'] = df['true_range'].rolling(window=atr_period).mean()
    
    # Normalize the price range by ATR to get the Intensity Ratio
    df['intensity_ratio'] = df['price_range'] / df['atr']
    
    # Return the intensity ratio as the factor
    return df['intensity_ratio']

# Example usage:
# df = pd.DataFrame({
#     'open': [100, 102, 101, 99, 103],
#     'high': [105, 107, 106, 104, 108],
#     'low': [95, 97, 96, 94, 98],
#     'close': [100, 102, 101, 99, 103],
#     'amount': [1000, 1050, 1020, 980, 1010],
#     'volume': [10000, 10500, 10200, 9800, 10100]
# })
# factors = heuristics_v2(df)
# print(factors)
