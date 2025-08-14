import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate daily, weekly, and monthly closing price changes
    df['daily_return'] = df['close'] - df['close'].shift(1)
    df['weekly_return'] = df['close'] - df['close'].shift(7)
    df['monthly_return'] = df['close'] - df['close'].shift(30)
    
    # Calculate ADX (Average Directional Index) over 14 days
    def calculate_adx(df):
        df['high_low'] = df['high'] - df['low']
        df['high_close_prev'] = abs(df['high'] - df['close'].shift(1))
        df['low_close_prev'] = abs(df['low'] - df['close'].shift(1))
        df['true_range'] = df[['high_low', 'high_close_prev', 'low_close_prev']].max(axis=1)
        
        df['plus_dm'] = df['high'].diff()
        df['minus_dm'] = -df['low'].diff()
        df['plus_dm'] = df['plus_dm'].apply(lambda x: x if x > 0 else 0)
        df['minus_dm'] = df['minus_dm'].apply(lambda x: x if x < 0 else 0)
        
        df['plus_di'] = 100 * (df['plus_dm'].rolling(window=14).sum() / df['true_range'].rolling(window=14).sum())
        df['minus_di'] = 100 * (df['minus_dm'].rolling(window=14).abs().sum() / df['true_range'].rolling(window=14).sum())
        df['adx'] = 100 * (abs(df['plus_di'] - df['minus_di']) / (df['plus_di'] + df['minus_di'])).rolling(window=14).mean()
        return df['adx']
    df['adx'] = calculate_adx(df)
    
    # Calculate Price-Volume Trend (PVT)
    df['pvt'] = (df['close'].diff() / df['close'].shift(1)) * df['volume']
    df['pvt'] = df['pvt'].cumsum()
    
    # Calculate Advance-Decline Line (ADL)
    df['adl'] = ((df['close'] > df['close'].shift(1)).astype(int) - (df['close'] < df['close'].shift(1)).astype(int)).cumsum()
    
    # Calculate True Range (TR) and Average True Range (ATR)
    df['tr'] = df[['high', 'close']].diff(axis=1).max(axis=1).fillna(0).abs() - df[['low', 'close']].diff(axis=1).min(axis=1).fillna(0).abs()
    df['atr'] = df['tr'].ewm(span=14, adjust=False).mean()
    
    # Calculate High-Low Ratio
    df['hl_ratio'] = (df['high'] - df['low']) / df['low']
    
    # Calculate Volume Weighted Average Price (VWAP)
    df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
    df['cumulative_volume_price'] = (df['typical_price'] * df['volume']).cumsum()
    df['cumulative_volume'] = df['volume'].cumsum()
    df['vwap'] = df['cumulative_volume_price'] / df['cumulative_volume']
    
    # Combine all factors into a single alpha factor
    df['alpha_factor'] = (df['daily_return'] + df['weekly_return'] + df['monthly_return'] + 
                          df['adx'] + df['pvt'] + df['adl'] + df['atr'] + df['hl_ratio'] + df['vwap'])
    
    return df['alpha_factor']
