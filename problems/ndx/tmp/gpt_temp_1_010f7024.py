import numpy as np
def heuristics_v2(df):
    # Calculate Intraday Volatility
    df['intraday_volatility'] = df['high'] - df['low']
    
    # Calculate Long-Term Momentum (20-day Moving Average)
    df['long_term_momentum'] = df['close'].rolling(window=20).mean()
    
    # Calculate Short-Term Momentum (5-day Moving Average)
    df['short_term_momentum'] = df['close'].rolling(window=5).mean()
    
    # Calculate Price Momentum
    df['price_momentum'] = df['long_term_momentum'] - df['short_term_momentum']
    
    # Detect Volume Surge
    df['volume_surge'] = ((df['volume'] - df['volume'].shift(1)) / df['volume'].shift(1) * 100).fillna(0)
    df['volume_surge_indicator'] = (df['volume_surge'] > 10).astype(int)  # Threshold for volume surge
    
    # Combine Momentum and Volume Surge
    df['combined_momentum'] = df['price_momentum'] * df['volume_surge_indicator']
    
    # Adjust Final Factor by Intraday Volatility
    df['adjusted_momentum'] = df['combined_momentum'] / df['intraday_volatility']
    
    # Calculate Price Oscillator
    df['price_oscillator'] = (df['high'] - df['low']) / df['close'].shift(1)
    
    # Integrate Adjusted Momentum and Price Oscillator
    df['integrated_momentum_oscillator'] = df['adjusted_momentum'] * df['price_oscillator']
    
    # Incorporate Price Change Rate
    df['price_change_rate'] = (df['close'] - df['close'].shift(1)) / df['close'].shift(1) * 100
    df['integrated_factor'] = df['integrated_momentum_oscillator'] + df['price_change_rate']
    
    # Calculate Daily Log Return
    df['log_return'] = (df['close'] / df['open']).apply(np.log)
