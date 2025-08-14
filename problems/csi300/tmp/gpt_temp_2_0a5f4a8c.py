import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate Daily Price Range
    df['price_range'] = df['high'] - df['low']
    
    # Compute Volume-Adjusted Momentum
    df['price_change'] = (df['close'] - df['close'].shift(10)) / df['close'].shift(10)
    df['volume_adjusted_momentum'] = df['price_change'] * df['volume']
    
    # Calculate Intraday Momentum
    df['intraday_momentum'] = df['high'] - df['low']
    
    # Integrate Volume-Adjusted Momentum and Intraday Momentum
    df['integrated_momentum'] = df['volume_adjusted_momentum'] * df['intraday_momentum']
    df['integrated_momentum'] = df.apply(lambda x: x['integrated_momentum'] * 2 if x['volume'] > 1.5 * df['volume'].shift(1) else x['integrated_momentum'], axis=1)
    
    # Apply Directional Bias
    df['factor'] = df['integrated_momentum']
    df['factor'] = df.apply(lambda x: x['factor'] * 1.5 if x['close'] > x['open'] else x['factor'] * 0.5, axis=1)
    
    # Incorporate Open-Close Trend
    df['open_close_diff'] = df['close'] - df['open']
    df['factor'] = df.apply(lambda x: x['factor'] * 1.2 if x['open_close_diff'] > 0 else x['factor'] * 0.8, axis=1)
    
    # Calculate Moving Averages
    df['short_ma'] = df['close'].rolling(window=5).mean()
    df['long_ma'] = df['close'].rolling(window=20).mean()
    
    # Compute Crossover Signal
    df['crossover_signal'] = df['short_ma'] - df['long_ma']
    
    # Generate Alpha Factor
    df['alpha_factor'] = df.apply(lambda x: 1 if x['crossover_signal'] > 0 else -1, axis=1)
    
    # Final Alpha Factor
    df['alpha_factor'] = df['alpha_factor'] * df['factor']
    
    return df['alpha_factor']
