def heuristics_v2(df):
    # Calculate intraday volatility measures
    df['daily_price_range'] = df['high'] - df['low']
    df['intraday_true_range'] = (df[['high', 'close', 'open']]
                                 .apply(lambda x: max(x['high'] - x['close'], 
                                                      x['high'] - x['open'],
                                                      x['close'] - x['open']), axis=1))
    df['daily_return'] = df['close'].pct_change()
