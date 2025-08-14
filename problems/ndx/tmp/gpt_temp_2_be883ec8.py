import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate simple moving averages
    short_sma = df['close'].rolling(window=5).mean()
    long_sma = df['close'].rolling(window=20).mean()
    
    # Momentum signal
    momentum_signal = short_sma - long_sma
    
    # Daily VWAP
    df['vwap'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
    
    # Percentage change in VWAP
    vwap_change = df['vwap'].pct_change().fillna(0)
    
    # Volume spike indicator
    avg_volume = df['volume'].rolling(window=20).mean()
    volume_spike = df['volume'] / avg_volume
    
    # True range calculation
    df['prev_close'] = df['close'].shift(1)
    true_range = (df[['high', 'low']].diff(axis=1).abs().max(axis=1) + 
                  (df['high'] - df['prev_close']).abs() + 
                  (df['low'] - df['prev_close']).abs()).max(axis=1)
    
    # ATR over a period
    atr_14 = true_range.rolling(window=14).mean()
    
    # Volatility breakout
    ema_close = df['close'].ewm(span=20, adjust=False).mean()
    upper_band = ema_close + atr_14 * 2
    lower_band = ema_close - atr_14 * 2
    
    # Daily return
    daily_return = (df['close'] - df['open']) / df['open']
    
    # Cumulative return over a recent period
    cumulative_return = (1 + daily_return).rolling(window=10).apply(lambda x: x.prod()) - 1
    
    # Sentiment trend
    positive_returns = daily_return.where(daily_return > 0, 0)
    negative_returns = daily_return.where(daily_return < 0, 0)
    sentiment_trend = positive_returns.rolling(window=10).sum() - negative_returns.rolling(window=10).sum()
    
    # EMAs
    ema_5 = df['close'].ewm(span=5, adjust=False).mean()
    ema_20 = df['close'].ewm(span=20, adjust=False).mean()
    
    # Trend direction based on the slope of EMAs
    ema_5_slope = ema_5.diff()
    ema_20_slope = ema_20.diff()
    
    # EMA crossover
    ema_crossover = (ema_5 > ema_20).astype(int).diff()
    
    # Composite score
    composite_score = (momentum_signal * 0.3 + 
                       vwap_change * 0.2 + 
                       volume_spike * 0.1 + 
                       atr_14 * 0.1 + 
                       cumulative_return * 0.1 + 
                       sentiment_trend * 0.1 + 
                       ema_5_slope * 0.05 + 
                       ema_20_slope * 0.05 + 
                       ema_crossover * 0.05)
    
    return composite_score
