import pandas as pd
import pandas as pd

def heuristics_v2(data):
    # Simple Moving Average (SMA) of the last 10 days close prices
    data['SMA_10'] = data['close'].rolling(window=10).mean()
    
    # Exponential Moving Average (EMA) with a span of 12 days on high prices
    data['EMA_12'] = data['high'].ewm(span=12, adjust=False).mean()
    
    # Volume to close price change ratio
    data['close_change'] = data['close'].diff()
    data['volume_to_close_change'] = data['volume'] / data['close_change']
    
    # Breakout indicator: current day's high exceeds the maximum high of the previous 5 days
    data['breakout_indicator'] = (data['high'] > data['high'].rolling(window=5).max()).astype(int)
    
    # Rate of Change (ROC) over 21 days on the close prices
    data['ROC_21'] = data['close'].pct_change(periods=21)
    
    # Momentum factor based on the difference between the latest 3-day SMA and the 7-day SMA of close prices
    data['SMA_3'] = data['close'].rolling(window=3).mean()
    data['SMA_7'] = data['close'].rolling(window=7).mean()
    data['momentum_factor'] = data['SMA_3'] - data['SMA_7']
    
    # Standard deviation of the daily returns over the past 20 days
    data['daily_return'] = data['close'].pct_change()
    data['std_dev_20'] = data['daily_return'].rolling(window=20).std()
    
    # True range indicator over the last 14 days
    data['true_range'] = data[['high', 'low', 'close']].apply(lambda x: max(x['high'] - x['low'], 
                                                                            abs(x['high'] - x['close'].shift(1)), 
                                                                            abs(x['low'] - x['close'].shift(1))), axis=1)
    data['true_range_14'] = data['true_range'].rolling(window=14).sum()
    
    # Composite strength score
    data['composite_strength_score'] = data['SMA_10'] + data['EMA_12'] + 2 * data['std_dev_20']
    
    # Directional Movement Index (DMI) over the last 14 days
    data['DM_pos'] = (data['high'] - data['high'].shift(1)).apply(lambda x: max(x, 0))
    data['DM_neg'] = (data['low'].shift(1) - data['low']).apply(lambda x: max(x, 0))
    data['DI_pos'] = data['DM_pos'].rolling(window=14).sum() / data['true_range_14']
    data['DI_neg'] = data['DM_neg'].rolling(window=14).sum() / data['true_range_14']
    data['DMI'] = data['DI_pos'] - data['DI_neg']
    
    # Money Flow Index (MFI) over the last 14 days
    data['typical_price'] = (data['high'] + data['low'] + data['close']) / 3
    data['raw_money_flow'] = data['typical_price'] * data['volume']
    data['positive_money_flow'] = data['raw_money_flow'] * (data['close'] > data['close'].shift(1))
    data['negative_money_flow'] = data['raw_money_flow'] * (data['close'] < data['close'].shift(1))
    data['money_ratio'] = data['positive_money_flow'].rolling(window=14).sum() / data['negative_money_flow'].rolling(window=14).sum()
    data['MFI'] = 100 - (100 / (1 + data['money_ratio']))
    
    # Custom volume-weighted price (VWP) over the last 20 days
    data['VWP'] = (data['close'] * data['volume']).rolling(window=20).sum() / data['volume'].rolling(window=20).sum()
    
    # Return the composite strength score as the alpha factor
    return data['composite_strength_score']

# Example usage:
# data = pd.read_csv('your_data.csv', index_col='date', parse_dates=True)
# factor_values = heuristics_v2(data)
# print(factor_values)
