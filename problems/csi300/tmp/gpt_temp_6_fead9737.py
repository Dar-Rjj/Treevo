import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate Intraday Momentum
    df['intraday_range'] = df['high'] - df['low']
    
    # Calculate Volume Spike
    df['volume_spike'] = (df['volume'] > 1.5 * df['volume'].shift(1)).astype(int)
    
    # Combine Intraday Momentum and Volume Spike
    df['combined_indicator'] = df['intraday_range'] * (2 if df['volume_spike'] == 1 else 1)
    
    # Adjust for Price Direction
    df['price_direction'] = (df['close'] > df['open']).astype(int) * 2 - 1
    df['combined_indicator'] *= df['price_direction']
    
    # Calculate Short-Term Moving Average
    df['short_term_ma'] = df['close'].rolling(window=5).mean()
    
    # Calculate Long-Term Moving Average
    df['long_term_ma'] = df['close'].rolling(window=20).mean()
    
    # Compute Crossover Signal
    df['crossover_signal'] = df['short_term_ma'] - df['long_term_ma']
    
    # Generate Alpha Factor
    df['alpha_factor'] = (df['crossover_signal'] > 0).astype(int) * 2 - 1
    
    # Integrate Combined Indicator and Alpha Factor
    df['alpha_factor'] += df['combined_indicator'] * (1 if df['alpha_factor'] == 1 else -1)
    
    return df['alpha_factor']

# Example usage:
# df = pd.read_csv('your_data.csv', index_col='date')
# alpha_factor = heuristics_v2(df)
# print(alpha_factor)
