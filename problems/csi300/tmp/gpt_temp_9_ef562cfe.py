import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate High-Low Spread
    df['High_Low_Spread'] = df['high'] - df['low']
    
    # Calculate Daily Volume Trend
    df['Volume_MA_10'] = df['volume'].rolling(window=10).mean()
    df['Volume_Trend'] = df['volume'] - df['Volume_MA_10']
    
    # Calculate Short-Term Price Trend
    df['Close_EMA_10'] = df['close'].ewm(span=10, adjust=False).mean()
    df['Short_Term_Price_Trend'] = df['close'] - df['Close_EMA_10']
    
    # Calculate Medium-Term Price Trend
    df['Close_EMA_30'] = df['close'].ewm(span=30, adjust=False).mean()
    df['Medium_Term_Price_Trend'] = df['close'] - df['Close_EMA_30']
    
    # Calculate Long-Term Price Trend
    df['Close_EMA_50'] = df['close'].ewm(span=50, adjust=False).mean()
    df['Long_Term_Price_Trend'] = df['close'] - df['Close_EMA_50']
    
    # Calculate Dynamic Volatility
    df['Volatility'] = df['close'].rolling(window=20).std()
    df['Volatility_Label'] = np.where(df['Volatility'] > df['Volatility'].quantile(0.75), 'high',
                                       np.where(df['Volatility'] < df['Volatility'].quantile(0.25), 'low', 'medium'))
    
    # Integrate Momentum and Relative Strength
    df['Close_EMA_5'] = df['close'].ewm(span=5, adjust=False).mean()
    df['Relative_Strength'] = df['Close_EMA_5'] / df['Close_EMA_20']
    df['Relative_Strength_Label'] = np.where(df['Relative_Strength'] > 1.1, 'strong', 'weak')
    
    # Combine Spread, Volume, Multi-Period Price Trends, and Volatility
    df['Adjusted_Spread'] = df['High_Low_Spread'] * (1.5 if df['Volume_Trend'] > 0 else 0.5)
    df['Adjusted_Short_Term'] = df['Adjusted_Spread'] * (1.2 if df['Short_Term_Price_Trend'] > 0 else 0.8)
    df['Adjusted_Medium_Term'] = df['Adjusted_Short_Term'] * (1.1 if df['Medium_Term_Price_Trend'] > 0 else 0.9)
    df['Adjusted_Long_Term'] = df['Adjusted_Medium_Term'] * (1.3 if df['Long_Term_Price_Trend'] > 0 else 0.7)
    
    # Incorporate Dynamic Volatility
    df['Adjusted_Volatility'] = df['Adjusted_Long_Term'] * (1.4 if df['Volatility_Label'] == 'high' else 0.6)
    
    # Incorporate Relative Strength
    df['Final_Factor'] = df['Adjusted_Volatility'] * (1.5 if df['Relative_Strength_Label'] == 'strong' else 0.5)
    
    # Consider Market Context
    market_trend = 'bullish'  # This should be determined by a separate function or input
    if market_trend == 'bullish':
        df['Final_Factor'] *= 1.5
    elif market_trend == 'bearish':
        df['Final_Factor'] *= 0.5
    
    return df['Final_Factor']

# Example usage
df = pd.DataFrame({
    'date': pd.date_range(start='2023-01-01', periods=100),
    'open': np.random.rand(100) * 100,
    'high': np.random.rand(100) * 100,
    'low': np.random.rand(100) * 100,
    'close': np.random.rand(100) * 100,
    'amount': np.random.rand(100) * 1000,
    'volume': np.random.randint(1000, 5000, size=100)
})
df.set_index('date', inplace=True)

alpha_factor = heuristics_v2(df)
print(alpha_factor)
