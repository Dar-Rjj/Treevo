import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Momentum Indicators
    short_sma = df['close'].rolling(window=5).mean()
    long_sma = df['close'].rolling(window=20).mean()
    sma_crossover = short_sma - long_sma
    
    roc = df['close'].pct_change(periods=14)
    
    # Volatility Indicators
    true_range = (df[['high', 'close']].shift(1).max(axis=1) - df[['low', 'close']].shift(1).min(axis=1))
    atr = true_range.rolling(window=14).mean()
    
    # Volume-Based Factors
    volume_trend = df['volume'] - df['volume'].rolling(window=10).mean()
    volume_spike = (df['volume'] / df['volume'].rolling(window=10).mean() > 3).rolling(window=10).sum()
    
    # Combined Price and Volume Factors
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    raw_money_flow = typical_price * df['volume']
    positive_money_flow = raw_money_flow.where((typical_price > typical_price.shift(1)), 0).rolling(window=14).sum()
    negative_money_flow = raw_money_flow.where((typical_price < typical_price.shift(1)), 0).rolling(window=14).sum()
    total_money_flow = positive_money_flow + negative_money_flow
    money_flow_index = positive_money_flow / total_money_flow
    
    clv = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
    adl = clv * df['volume'].cumsum()
    
    # Pattern Recognition
    doji_pattern = (df['open'] == df['close']) & (df['high'] - df['low'] > 0.001 * df['open'])
    hammer_pattern = (df['close'] > df['open']) & ((df['high'] - df['close']) <= 0.05 * (df['high'] - df['low']))
    engulfing_pattern = (df['close'] > df['open']) & (df['close'].shift(1) < df['open'].shift(1)) & (df['close'] > df['open'].shift(1)) & (df['open'] < df['close'].shift(1))
    
    # Other Novel Factors
    high_low_ratio = df['high'] / df['low']
    close_open_ratio = df['close'] / df['open']
    
    # Combine all factors into a single DataFrame
    factors = pd.DataFrame({
        'SMA_Crossover': sma_crossover,
        'ROC': roc,
        'ATR': atr,
        'Volume_Trend': volume_trend,
        'Volume_Spike': volume_spike,
        'Money_Flow_Index': money_flow_index,
        'ADL': adl,
        'Doji_Pattern': doji_pattern,
        'Hammer_Pattern': hammer_pattern,
        'Engulfing_Pattern': engulfing_pattern,
        'High_Low_Ratio': high_low_ratio,
        'Close_Open_Ratio': close_open_ratio
    })
    
    return factors
