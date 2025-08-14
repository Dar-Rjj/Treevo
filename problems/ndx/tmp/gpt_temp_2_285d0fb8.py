import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Obtain Last 5 Days' Close Prices
    last_5_close = df['close'].rolling(window=5, min_periods=1).apply(lambda x: x[-5:], raw=True)
    
    # Calculate Moving Averages
    short_term_ma = last_5_close.apply(lambda x: x[-3:].mean(), raw=True)  # 3-day MA
    long_term_ma = df['close'].rolling(window=10, min_periods=1).mean()  # 10-day MA
    
    # Compute Momentum Signal
    momentum_signal = long_term_ma - short_term_ma
    momentum_direction = (momentum_signal > momentum_signal.shift(1)).astype(int)  # 1 if increasing, 0 otherwise
    
    # Calculate Volume Ratio
    volume_last_5 = df['volume'].rolling(window=5, min_periods=1).sum()
    avg_volume_last_20 = df['volume'].rolling(window=20, min_periods=1).mean()
    volume_ratio = volume_last_5 / avg_volume_last_20
    
    # Apply Volume Filter
    volume_threshold = 1.5  # Threshold for high volume
    alpha_factor = pd.Series(index=df.index, dtype=float)
    
    for i in range(len(df)):
        if momentum_direction.iloc[i] == 1 and volume_ratio.iloc[i] > volume_threshold:
            alpha_factor.iloc[i] = 2  # Strong Positive Factor
        elif momentum_direction.iloc[i] == 0 and volume_ratio.iloc[i] <= volume_threshold:
            alpha_factor.iloc[i] = -1  # Weak Negative Factor
        elif momentum_direction.iloc[i] == 1 and volume_ratio.iloc[i] <= volume_threshold:
            alpha_factor.iloc[i] = 0  # Neutral Factor
        elif momentum_direction.iloc[i] == 0 and volume_ratio.iloc[i] > volume_threshold:
            alpha_factor.iloc[i] = -2  # Negative Factor
        else:
            alpha_factor.iloc[i] = 0  # Default to Neutral Factor
    
    return alpha_factor
