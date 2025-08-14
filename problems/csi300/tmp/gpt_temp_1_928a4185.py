import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate short-term momentum (5-day change)
    df['short_term_momentum'] = df['close'].pct_change(5)
    
    # Calculate medium-term momentum (20-day change)
    df['medium_term_momentum'] = df['close'].pct_change(20)
    
    # Calculate 14-day Relative Strength Index (RSI)
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # Calculate 50-day moving average of volume
    df['volume_trend'] = df['volume'].rolling(window=50).mean()
    
    # Identify bullish and bearish engulfing patterns
    df['bullish_engulfing'] = ((df['open'] > df['close'].shift(1)) & (df['close'] > df['open']) & 
                               (df['open'].shift(1) > df['close'].shift(1)) & 
                               (df['close'] > df['open'].shift(1)) & (df['close'].shift(1) >= df['open']))
    df['bearish_engulfing'] = ((df['open'] < df['close'].shift(1)) & (df['close'] < df['open']) & 
                               (df['open'].shift(1) < df['close'].shift(1)) & 
                               (df['close'] < df['open'].shift(1)) & (df['close'].shift(1) <= df['open']))
    
    # Combine short-term momentum with volume trends
    df['composite_alpha'] = df['short_term_momentum'] * (df['volume'] / df['volume_trend'])
    
    # Integrate overbought/oversold conditions with pattern recognition
    df['enhanced_signal'] = (df['rsi'] < 30) * df['bullish_engulfing'] - (df['rsi'] > 70) * df['bearish_engulfing']
    
    # Final alpha factor: weighted sum of the derived factors
    df['alpha_factor'] = (df['short_term_momentum'] + df['medium_term_momentum'] + 
                          df['composite_alpha'] + df['enhanced_signal'])
    
    return df['alpha_factor'].dropna()

# Example usage:
# df = pd.DataFrame({
#     'open': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
#     'high': [105, 106, 107, 108, 109, 110, 111, 112, 113, 114],
#     'low': [95, 96, 97, 98, 99, 100, 101, 102, 103, 104],
#     'close': [103, 104, 105, 106, 107, 108, 109, 110, 111, 112],
#     'amount': [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900],
#     'volume': [100, 110, 120, 130, 140, 150, 160, 170, 180, 190]
# })
# alpha_factor = heuristics_v2(df)
# print(alpha_factor)
