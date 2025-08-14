import pandas as pd
import pandas as pd

def heuristics_v2(df, window_size=20):
    # Calculate Daily Return
    df['daily_return'] = df['close'].pct_change()

    # Calculate Volume Change
    df['volume_change'] = df['volume'].diff()

    # Weighted Price Momentum
    df['weighted_momentum'] = df['daily_return'] * df['volume']
    df['weighted_momentum_avg'] = df['weighted_momentum'].rolling(window=window_size).mean()

    # Relative Strength Adjustment
    df['lowest_low'] = df['low'].rolling(window=window_size).min()
    df['range'] = df['high'].rolling(window=window_size).max() - df['low'].rolling(window=window_size).min()
    df['relative_strength'] = (df['high'] - df['lowest_low']) / df['range']

    df['adjusted_vpmi'] = df['weighted_momentum_avg'] * df['relative_strength']

    # Intraday Volatility-Adjusted Momentum
    df['true_range'] = df.apply(
        lambda x: max(x['high'] - x['low'], 
                      abs(x['high'] - df['close'].shift(1).iloc[x.name]), 
                      abs(x['low'] - df['close'].shift(1).iloc[x.name])),
        axis=1
    )
    df['intraday_volatility'] = df['true_range'].rolling(window=window_size).mean()

    df['volatility_adjusted_momentum'] = df['weighted_momentum_avg'] / (df['intraday_volatility'] + 1e-8)

    # Final Alpha Factor
    df['final_alpha_factor'] = df['adjusted_vpmi'] + df['volatility_adjusted_momentum']

    return df['final_alpha_factor']
