import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Volatility Breakout Momentum
    volatility_ratio = (df['high'] - df['low']) / (df['high'].shift(5) - df['low'].shift(5))
    
    momentum_persistence = pd.Series(
        [sum(df['close'].shift(i) > df['close'].shift(i+1) for i in range(5)) / 5 
         for _ in range(len(df))], 
        index=df.index
    )
    
    signal1 = volatility_ratio * momentum_persistence * df['volume']
    
    # Gap Absorption Efficiency
    gap = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
    absorption_rate = (df['close'] - df['open']) / (df['high'] - df['low'])
    signal2 = gap * absorption_rate * df['volume']
    
    # Pressure Reversal
    pressure_change = (df['close'] - df['open']) * df['volume'] - (df['close'].shift(1) - df['open'].shift(1)) * df['volume'].shift(1)
    range_expansion = (df['high'] - df['low']) / (df['high'].shift(1) - df['low'].shift(1))
    signal3 = -pressure_change.apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0)) * range_expansion * df['volume']
    
    # Volume-Volatility Divergence
    volume_ratio = df['volume'] / df['volume'].rolling(window=10).mean()
    volatility_ratio2 = (df['high'] - df['low']) / (df['high'] - df['low']).rolling(window=10).mean()
    signal4 = (volume_ratio - volatility_ratio2) * (df['close'] / df['close'].shift(1) - 1)
    
    # Auction Momentum
    auction_strength = (df['open'] - df['close'].shift(1)) / (df['high'].shift(1) - df['low'].shift(1))
    follow_through = (df['close'] - df['open']) / (df['high'] - df['low'])
    signal5 = auction_strength * follow_through * (df['close'] - df['close'].shift(1))
    
    # Combine signals with equal weights
    combined_signal = (signal1 + signal2 + signal3 + signal4 + signal5) / 5
    
    return combined_signal
