import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate Short-Term Return
    df['short_term_return'] = (df['close'] - df['close'].shift(7)) / df['close'].shift(7)
    
    # Calculate Long-Term Return
    df['long_term_return'] = (df['close'] - df['close'].shift(90)) / df['close'].shift(90)
    
    # Volume-Weighted Short-Term Return
    df['volume_weighted_short_term_return'] = df['short_term_return'] * df['volume']
    
    # Volume-Weighted Long-Term Return
    df['volume_weighted_long_term_return'] = df['long_term_return'] * df['volume']
    
    # Measure Intraday Volatility
    df['intraday_volatility'] = abs(df['high'] - df['low'])
    
    # Adjust Returns by Intraday Volatility
    df['adjusted_short_term_return'] = df['volume_weighted_short_term_return'] / df['intraday_volatility']
    df['adjusted_long_term_return'] = df['volume_weighted_long_term_return'] / df['intraday_volatility']
    
    # Combined Adjusted Momentum Factor
    df['momentum_factor'] = df['adjusted_long_term_return'] - df['adjusted_short_term_return']
    
    # Reversal Indicator
    df['reversal_indicator'] = -df['momentum_factor']
    
    # Calculate 5-Day and 90-Day Exponential Moving Average of Close Price
    df['ema_5'] = df['close'].ewm(span=5, adjust=False).mean()
    df['ema_90'] = df['close'].ewm(span=90, adjust=False).mean()
    
    # Trend Indicator
    df['trend_indicator'] = df['ema_5'] - df['ema_90']
    
    # Daily Price Movement
    df['daily_price_movement'] = df['close'] - df['open']
    
    # Uptrend Count
    df['uptrend_count'] = (df['close'] > df['open']).rolling(window=10).sum()
    
    # Average True Range (ATR)
    df['tr'] = df[['high' - 'low', abs('high' - df['close'].shift(1)), abs('low' - df['close'].shift(1))]].max(axis=1)
    df['atr_10'] = df['tr'].rolling(window=10).mean()
    
    # ATR Moving Average
    df['atr_20_ma'] = df['atr_10'].rolling(window=20).mean()
    
    # ATR Signal
    df['atr_signal'] = (df['atr_10'] < df['atr_20_ma']).astype(int)
    
    # Positive Volume Score
    df['avg_volume_30'] = df['volume'].rolling(window=30).mean()
    df['positive_volume_score'] = ((df['close'] > df['open']) & (df['volume'] > df['avg_volume_30'])).rolling(window=30).sum()
    
    # Intraday Bullishness
    df['intraday_bullishness'] = (df['high'] - df['open']) / (df['open'] - df['low'])
    
    # Bullishness Ratio
    df['bullishness_ratio'] = df['intraday_bullishness'].rolling(window=5).mean()
    
    # Combined Indicator
    df['combined_indicator'] = (df['momentum_factor'] + df['uptrend_count'] + df['atr_signal'] + df['positive_volume_score'] + df['bullishness_ratio']) / 5
    
    # Final Alpha Factor
    df['final_alpha_factor'] = df['reversal_indicator'] + df['trend_indicator'] + df['combined_indicator']
    
    # Incorporate Open and Close Price Difference
    df['final_alpha_factor'] += df['daily_price_movement']
    
    return df['final_alpha_factor']
