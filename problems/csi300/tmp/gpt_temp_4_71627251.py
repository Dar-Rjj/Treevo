import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate Intraday Momentum
    df['intraday_range'] = df['high'] - df['low']
    
    # Calculate Volume Spike
    df['volume_spike'] = (df['volume'] > 1.5 * df['volume'].shift(1)).astype(int)
    
    # Adjust Intraday Range for Spike
    df['adjusted_intraday_range'] = df['intraday_range'] * (2 if df['volume_spike'] == 1 else 1)
    
    # Adjust for Price Direction
    df['price_direction'] = (df['close'] > df['open']).astype(int) * 2 - 1
    df['combined_indicator'] = df['adjusted_intraday_range'] * df['price_direction']
    
    # Calculate Moving Averages
    df['short_term_ma'] = df['close'].rolling(window=5).mean()
    df['long_term_ma'] = df['close'].rolling(window=20).mean()
    
    # Compute Crossover Signal
    df['crossover_signal'] = df['short_term_ma'] - df['long_term_ma']
    
    # Generate Alpha Factor
    df['alpha_factor'] = (df['crossover_signal'] > 0).astype(int) * 2 - 1
    
    # Integrate Combined Indicator and Alpha Factor
    df['alpha_factor'] = df.apply(lambda row: row['alpha_factor'] + row['combined_indicator'] if row['alpha_factor'] == 1 else row['alpha_factor'] - row['combined_indicator'], axis=1)
    
    return df['alpha_factor']
