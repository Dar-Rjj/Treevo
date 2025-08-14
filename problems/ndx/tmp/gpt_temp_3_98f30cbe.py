import numpy as np
def heuristics_v2(df):
    # Short-term Momentum
    short_term_momentum = df['close'].rolling(window=5).mean() - df['close'].rolling(window=20).mean()
    
    # Long-term Momentum
    long_term_momentum = df['close'].rolling(window=100).mean() - df['close'].rolling(window=200).mean()
    
    # Relative Momentum
    relative_momentum = (df['close'] - df['close'].rolling(window=5).mean()) / df['close'].rolling(window=5).mean()
    relative_momentum += (df['close'] - df['close'].rolling(window=20).mean()) / df['close'].rolling(window=20).mean()
    relative_momentum += (df['close'] - df['close'].rolling(window=100).mean()) / df['close'].rolling(window=100).mean()
    
    # Price Volatility
    price_volatility = (df['high'] - df['low']).rolling(window=10).std()
    
    # Volume Volatility
    volume_volatility = df['volume'].rolling(window=10).std()
    
    # Combined Volatility
    combined_volatility = np.sqrt(price_volatility * volume_volatility)
    
    # Short-term Trend
    def calculate_slope(series):
        x = np.arange(len(series))
        y = series.values
        slope, _, _, _, _ = linregress(x, y)
        return slope
