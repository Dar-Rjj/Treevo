import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Momentum Factors
    df['price_roc_20'] = df['close'].pct_change(20)
    df['volume_roc_20'] = df['volume'].pct_change(20)
    
    # Volatility Factors
    df['true_range'] = df.apply(lambda x: max(x['high'] - x['low'], abs(x['high'] - df['close'].shift(1)), abs(x['low'] - df['close'].shift(1))), axis=1)
    df['atr_14'] = df['true_range'].rolling(window=14).mean()
    df['volume_std_14'] = df['volume'].pct_change().rolling(window=14).std()
    
    # Trend Strength Factors
    df['+DM'] = (df['high'] - df['high'].shift(1)).apply(lambda x: max(x, 0))
    df['-DM'] = (df['low'].shift(1) - df['low']).apply(lambda x: max(x, 0))
    df['+DI'] = df['+DM'].rolling(window=14).mean() / df['atr_14']
    df['-DI'] = df['-DM'].rolling(window=14).mean() / df['atr_14']
    df['DX'] = (abs(df['+DI'] - df['-DI']) / (df['+DI'] + df['-DI'])) * 100
    df['ADX'] = df['DX'].ewm(span=14, adjust=False).mean()
    
    df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
    df['sma_tp_20'] = df['typical_price'].rolling(window=20).mean()
    df['mad_tp_20'] = df['typical_price'].rolling(window=20).apply(lambda x: (x - x.mean()).abs().mean(), raw=True)
    df['CCI'] = (df['typical_price'] - df['sma_tp_20']) / (0.015 * df['mad_tp_20'])
    
    # Market Sentiment Factors
    df['RSL'] = df['close'] / df['close'].rolling(window=252).mean()
    
    df['money_flow'] = df['typical_price'] * df['volume']
    df['positive_money_flow'] = df.apply(lambda x: x['money_flow'] if x['typical_price'] > df['typical_price'].shift(1) else 0, axis=1)
    df['negative_money_flow'] = df.apply(lambda x: x['money_flow'] if x['typical_price'] < df['typical_price'].shift(1) else 0, axis=1)
    df['MFI'] = 100 - (100 / (1 + (df['positive_money_flow'].rolling(window=14).sum() / df['negative_money_flow'].rolling(window=14).sum())))
    
    # Liquidity Factors
    df['dollar_volume'] = df['close'] * df['volume']
    
    # New Alpha Factor: Price-Volume Divergence
    df['cumulative_price_diff_20'] = (df['close'] - df['close'].shift(1)).rolling(window=20).sum()
    df['cumulative_volume_diff_20'] = (df['volume'] - df['volume'].shift(1)).rolling(window=20).sum()
    df['price_volume_divergence'] = df['cumulative_price_diff_20'] - df['cumulative_volume_diff_20']
    
    return df['price_volume_divergence']
