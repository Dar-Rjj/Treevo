import pandas as pd
import pandas as pd

def heuristics_v2(df):
    """
    Generate a novel and interpretable alpha factor based on the provided DataFrame.
    
    Parameters:
    df (pd.DataFrame): A DataFrame with (date) as index and columns ['open', 'high', 'low', 'close', 'volume'].
    
    Returns:
    pd.Series: A Series with (date) as index and the calculated factor values.
    """
    # Calculate the typical price
    typical_price = (df['high'] + df['low'] + df['close']) / 3.0
    
    # Calculate the money flow, which is the product of the typical price and volume
    money_flow = typical_price * df['volume']
    
    # Calculate the 14-day positive and negative money flow
    positive_money_flow = money_flow.rolling(window=14).apply(lambda x: (x * (typical_price > typical_price.shift(1))).sum(), raw=False)
    negative_money_flow = money_flow.rolling(window=14).apply(lambda x: (x * (typical_price < typical_price.shift(1))).sum(), raw=False)
    
    # Calculate the money flow ratio
    money_flow_ratio = positive_money_flow / negative_money_flow
    
    # Calculate the money flow index
    money_flow_index = 100 - (100 / (1 + money_flow_ratio))
    
    # Calculate the 5-day moving average of the close price
    ma_close_5 = df['close'].rolling(window=5).mean()
    
    # Calculate the 20-day moving average of the close price
    ma_close_20 = df['close'].rolling(window=20).mean()
    
    # Calculate the trend factor as the ratio of the 5-day MA to the 20-day MA
    trend_factor = ma_close_5 / ma_close_20
    
    # Combine the money flow index and the trend factor
    factor = (money_flow_index + trend_factor) / 2.0
    
    return factor

# Example usage:
# df = pd.read_csv('stock_data.csv', index_col='date', parse_dates=True)
# factor_values = heuristics_v2(df)
# print(factor_values)
