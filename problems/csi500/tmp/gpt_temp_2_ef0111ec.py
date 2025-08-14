import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate Daily High-Low Range
    df['high_low_range'] = df['high'] - df['low']
    
    # Calculate 10-Day Sum of High-Low Ranges
    df['10d_sum_high_low_range'] = df['high_low_range'].rolling(window=10).sum()
    
    # Calculate Price Change over 10 Days
    df['price_change_10d'] = df['close'] - df['close'].shift(10)
    
    # Calculate Intraday Mean Price
    df['intraday_mean_price'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4
    
    # Measure Intraday Deviation
    df['intraday_deviation'] = df['high'] - df['low']
    
    # Calculate Price Momentum
    df['price_momentum'] = df['close'] - df['close'].shift(1)
    
    # Calculate Close Momentum
    n = 10
    for i in range(1, n):
        df[f'close_momentum_{i}'] = (df['close'] - df['close'].shift(i)) / df['close'].shift(i)
    df['close_momentum'] = df[[f'close_momentum_{i}' for i in range(1, n)]].sum(axis=1)
    
    # Classify Volume
    lookback_period = 30
    df['average_volume'] = df['volume'].rolling(window=lookback_period).mean()
    df['volume_classification'] = df['volume'] > df['average_volume']
    
    # Classify Amount
    df['average_amount'] = df['amount'].rolling(window=lookback_period).mean()
    df['amount_classification'] = df['amount'] > df['average_amount']
    
    # Combine Price Momentum with Volume and Amount Classification
    df['weight'] = 1.0
    df.loc[(df['volume_classification'] == True) & (df['amount_classification'] == True), 'weight'] = 1.5
    df.loc[(df['volume_classification'] == True) & (df['amount_classification'] == False), 'weight'] = 1.25
    df.loc[(df['volume_classification'] == False) & (df['amount_classification'] == True), 'weight'] = 1.25
    df.loc[(df['volume_classification'] == False) & (df['amount_classification'] == False), 'weight'] = 0.75
    df['weighted_price_momentum'] = df['price_momentum'] * df['weight']
    
    # Volume Trend
    m = 21
    df['volume_trend'] = df['volume'].rolling(window=m).mean()
    df['volume_score'] = (df['volume'] - df['volume'].shift(m)) / df['volume'].shift(m)
    
    # Integrate All Scores
    df['integrated_score'] = df['weighted_price_momentum'] * df['volume_score']
    df['integrated_score'] += df['10d_sum_high_low_range'] / df['intraday_mean_price']
    df['integrated_score'] *= df['close_momentum']
    
    # Price Volatility
    df['10d_close_std'] = df['close'].rolling(window=10).std()
    df['integrated_score'] += df['10d_close_std'] * 0.1  # Adjusting factor by volatility
    
    # Incorporate Open-Price Relative Strength
    df['open_price_rel_strength'] = ((df['close'] - df['open']) / df['open']).rolling(window=10).sum()
    df['integrated_score'] += df['open_price_rel_strength'] * 0.1  # Adjusting factor by open-price relative strength
    
    # Integrate Trading Volume and Price Trend
    df['10d_volume_close_corr'] = df[['volume', 'close']].rolling(window=10).corr().unstack().iloc[::2, :]['close']
    df['integrated_score'] += df['10d_volume_close_corr'] * 0.1  # Adjusting factor by correlation
    
    # Calculate Intraday Volatility
    df['intraday_volatility'] = df['intraday_deviation'] / df['intraday_mean_price']
    df['integrated_score'] += df['intraday_volatility'] * 0.1  # Adjusting factor by intraday volatility
    
    return df['integrated_score']
