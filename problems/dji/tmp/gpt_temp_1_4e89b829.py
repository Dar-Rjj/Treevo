import pandas as pd
import pandas as pd

def heuristics_v2(df, window=10):
    # Calculate Daily Return
    df['daily_return'] = df['close'] - df['close'].shift(1)
    
    # Calculate Volume Change
    df['volume_change'] = df['volume'] - df['volume'].shift(1)
    
    # Weighted Price Momentum
    df['weighted_momentum'] = df['daily_return'] * df['volume']
    df['weighted_momentum_rolling'] = df['weighted_momentum'].rolling(window=window).mean()
    
    # Relative Strength Adjustment
    df['range'] = df['high'] - df['low']
    df['relative_strength'] = (df['high'] - df['low'].rolling(window=window).min()) / df['range'].rolling(window=window).sum()
    df['adjusted_vpmi'] = df['weighted_momentum_rolling'] * df['relative_strength']
    
    # Intraday Volatility-Adjusted Momentum
    df['true_range'] = df[['high', 'low', 'close']].apply(
        lambda x: max(x['high'] - x['low'], abs(x['high'] - x['close'].shift(1)), abs(x['low'] - x['close'].shift(1))), axis=1
    )
    df['intraday_volatility_adjusted_momentum'] = df['weighted_momentum_rolling'] / df['true_range']
    
    # Final Alpha Factor
    df['alpha_factor'] = df['adjusted_vpmi'] + df['intraday_volatility_adjusted_momentum']
    
    return df['alpha_factor']
