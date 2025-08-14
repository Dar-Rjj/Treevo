import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate Daily Return
    df['daily_return'] = df['close'] - df['close'].shift(1)
    
    # Calculate Volume Change
    df['volume_change'] = df['volume'] - df['volume'].shift(1)
    
    # Weighted Price Momentum
    df['weighted_price_momentum'] = (df['daily_return'] * df['volume']).rolling(window=20).mean()
    
    # Relative Strength
    high_low_range = df['high'].rolling(window=20).max() - df['low'].rolling(window=20).min()
    highest_high = df['high'].rolling(window=20).max()
    df['relative_strength'] = high_low_range / highest_high
    
    # Intraday Volatility
    df['true_range'] = df[['high' - 'low', 
                           'high' - df['close'].shift(1), 
                           df['close'].shift(1) - 'low']].max(axis=1)
    df['intraday_volatility'] = df['true_range'].rolling(window=20).mean()
    
    # Adjusted VPMI
    df['adjusted_vpmi'] = df['weighted_price_momentum'] * df['relative_strength']
    
    # Intraday Volatility-Adjusted Momentum
    df['intraday_volatility_adjusted_momentum'] = df['adjusted_vpmi'] / df['intraday_volatility']
    
    # Final Alpha Factor
    alpha_factor = 0.5 * df['adjusted_vpmi'] + 0.5 * df['intraday_volatility_adjusted_momentum']
    
    return alpha_factor
