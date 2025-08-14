import pandas as pd
import pandas as pd

def heuristics(df):
    # Trend Following
    df['5_day_SMA'] = df['close'].rolling(window=5).mean()
    df['20_day_SMA'] = df['close'].rolling(window=20).mean()
    df['SMA_Crossover_Factor'] = df['20_day_SMA'] - df['5_day_SMA']
    
    df['12_day_Momentum'] = df['close'].pct_change(periods=12)
    
    # Mean Reversion
    df['50_day_MA'] = df['close'].rolling(window=50).mean()
    df['Price_Relative_to_MA'] = (df['50_day_MA'] - df['close']) / df['50_day_MA']
    
    df['20_day_MA'] = df['close'].rolling(window=20).mean()
    df['20_day_STD'] = df['close'].rolling(window=20).std()
    df['Upper_Band'] = df['20_day_MA'] + 2 * df['20_day_STD']
    df['Lower_Band'] = df['20_day_MA'] - 2 * df['20_day_STD']
    df['Bollinger_Bands_Reversal'] = (df['close'] - df['Lower_Band']) / (df['Upper_Band'] - df['Lower_Band'])
    
    # Volume Indicators
    df['OBV'] = (df['close'] > df['close'].shift(1)).astype(int) * df['volume'] - (df['close'] < df['close'].shift(1)).astype(int) * df['volume']
    df['OBV'] = df['OBV'].cumsum()
    df['OBV_Change_10_days'] = df['OBV'].pct_change(periods=10)
    
    df['Money_Flow_Multiplier'] = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
    df['Money_Flow_Volume'] = df['Money_Flow_Multiplier'] * df['volume']
    df['Chaikin_Money_Flow'] = df['Money_Flow_Volume'].rolling(window=20).sum() / df['volume'].rolling(window=20).sum()
    
    # Volatility
    df['Daily_Returns'] = df['close'].pct_change()
    df['Historical_Volatility'] = df['Daily_Returns'].rolling(window=30).std()
    
    # Pattern Recognition
    def engulfing_pattern(row, prev_row):
        if (row['open'] > row['close'] and prev_row['open'] < prev_row['close'] and
            row['close'] > prev_row['open'] and row['open'] < prev_row['close']):
            return 1
        elif (row['open'] < row['close'] and prev_row['open'] > prev_row['close'] and
              row['close'] < prev_row['open'] and row['open'] > prev_row['close']):
            return -1
        else:
            return 0
    
    df['Engulfing_Pattern'] = [engulfing_pattern(row, prev_row) for row, prev_row in zip(df.iterrows(), df.shift(1).iterrows())]
    
    # Combine all factors into a single series
    alpha_factors = pd.Series({
        'SMA_Crossover_Factor': df['SMA_Crossover_Factor'],
        '12_day_Momentum': df['12_day_Momentum'],
        'Price_Relative_to_MA': df['Price_Relative_to_MA'],
        'Bollinger_Bands_Reversal': df['Bollinger_Bands_Reversal'],
        'OBV_Change_10_days': df['OBV_Change_10_days'],
        'Chaikin_Money_Flow': df['Chaikin_Money_Flow'],
        'Historical_Volatility': df['Historical_Volatility'],
        'Engulfing_Pattern': df['Engulfing_Pattern']
    })
    
    return alpha_factors
