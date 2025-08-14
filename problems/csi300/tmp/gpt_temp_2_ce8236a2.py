import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Compute short-term price trend (5-day moving average)
    df['price_5d_ma'] = df['close'].rolling(window=5).mean()
    
    # Detect bearish and bullish reversals
    df['bearish_reversal'] = (df['close'].shift(1) > df['price_5d_ma'].shift(1)) & (df['close'] < df['price_5d_ma'])
    df['bullish_reversal'] = (df['close'].shift(1) < df['price_5d_ma'].shift(1)) & (df['close'] > df['price_5d_ma'])
    
    # Calculate short-term volume trend (5-day moving average of volume)
    df['volume_5d_ma'] = df['volume'].rolling(window=5).mean()
    
    # Identify significant volume increase and decrease
    df['volume_20d_ma'] = df['volume'].rolling(window=20).mean()
    df['significant_volume_increase'] = df['volume'] > 2 * df['volume_20d_ma']
    df['significant_volume_decrease'] = df['volume'] < 0.5 * df['volume_20d_ma']
    
    # Daily price-volume correlation
    df['price_change'] = df['close'].pct_change()
    df['daily_corr'] = df['price_change'].rolling(window=5).corr(df['volume'])
    
    # Positive and negative divergence
    df['positive_divergence'] = (df['price_change'] < 0) & (df['volume'] > df['volume'].shift(1))
    df['negative_divergence'] = (df['price_change'] > 0) & (df['volume'] < df['volume'].shift(1))
    
    # Hammer pattern
    df['hammer_pattern'] = (df['close'] - df['low']) / (df['high'] - df['low']) > 0.6 & (df['open'] - df['low']) / (df['high'] - df['low']) > 0.6
    
    # Engulfing bullish and bearish patterns
    df['engulfing_bullish'] = (df['close'] > df['open']) & (df['close'].shift(1) < df['open'].shift(1)) & (df['close'] > df['open'].shift(1))
    df['engulfing_bearish'] = (df['close'] < df['open']) & (df['close'].shift(1) > df['open'].shift(1)) & (df['close'] < df['open'].shift(1))
    
    # Hanging man pattern
    df['hanging_man'] = (df['close'] - df['low']) / (df['high'] - df['low']) > 0.6 & (df['open'] - df['low']) / (df['high'] - df['low']) > 0.6 & (df['close'] - df['open']) / (df['high'] - df['low']) < 0.1
    
    # Daily price range
    df['daily_price_range'] = df['high'] - df['low']
    df['price_range_20d_ma'] = df['daily_price_range'].rolling(window=20).mean()
    
    # Detect high and low volatility days
    df['high_volatility_day'] = df['daily_price_range'] > 2 * df['price_range_20d_ma']
    df['low_volatility_day'] = df['daily_price_range'] < 0.5 * df['price_range_20d_ma']
    
    # Generate alpha factor
    df['alpha_factor'] = (
        df['bullish_reversal'].astype(int) 
        - df['bearish_reversal'].astype(int) 
        + df['significant_volume_increase'].astype(int) 
        - df['significant_volume_decrease'].astype(int) 
        + df['positive_divergence'].astype(int) 
        - df['negative_divergence'].astype(int) 
        + df['engulfing_bullish'].astype(int) 
        - df['engulfing_bearish'].astype(int) 
        + df['hammer_pattern'].astype(int) 
        - df['hanging_man'].astype(int) 
        + df['high_volatility_day'].astype(int) 
        - df['low_volatility_day'].astype(int)
    )
    
    return df['alpha_factor']
