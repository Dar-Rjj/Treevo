import pandas as pd

def heuristics_v2(df):
    # Calculate the daily return
    df['Return'] = df['close'].pct_change()
    
    # Shift the return to align with the factors for prediction
    df['Future_Return'] = df['Return'].shift(-1)
    
    # Drop rows with NaN values resulting from the shift
    df = df.dropna()

    # Calculate a set of technical indicators
    def calculate_momentum(df, window):
        return (df['close'] / df['close'].shift(window)) - 1

    def calculate_range(df, window):
        high = df['high'].rolling(window=window).max()
        low = df['low'].rolling(window=window).min()
        return (high - low) / ((high + low) / 2)

    momentum = calculate_momentum(df, 20)
    range_ = calculate_range(df, 20)

    # Initialize an empty DataFrame to store the dynamic weights
    weights = pd.DataFrame(index=df.index, columns=['momentum', 'range'], dtype='float64')

    # Compute the dynamic weights
    for col in ['momentum', 'range']:
        corr_with_return = df[col].rolling(window=20).corr(df['Future_Return'])
        vol = df[col].rolling(window=20).std()
        weights[col] = (corr_with_return / vol)  # Adjust weights based on volatility and correlation
    
    # Fill any remaining NaNs in the weights matrix
    weights = weights.fillna(0)

    # Combine the weighted factors
    heuristics_matrix = (momentum * weights['momentum']) + (range_ * weights['range'])

    return heuristics_matrix
