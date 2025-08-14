import pandas as pd
import pandas as pd

def heuristics_v2(df, lookback=10):
    # Calculate Intraday Volatility
    intraday_volatility = df['high'] - df['low']
    
    # Calculate Volume-to-Price Ratio
    volume_to_price_ratio = df['volume'] / ((df['high'] + df['low']) / 2)
    
    # Calculate Weighted Intraday Volatility
    weighted_intraday_volatility = intraday_volatility * volume_to_price_ratio
    
    # Calculate Close-to-Open Change
    close_to_open_change = df['close'] - df['open']
    
    # Calculate Enhanced Factor
    enhanced_factor = weighted_intraday_volatility - close_to_open_change
    
    # Calculate Exponential Moving Average of (Close - Open) over the lookback period
    ema_close_open_spread = df['close'] - df['open']
    ema_close_open_spread_momentum = ema_close_open_spread.ewm(span=lookback, adjust=False).mean()
    
    # Initialize Final Factor
    final_factor = pd.Series(index=df.index, dtype=float)
    
    # Conditional calculation of Final Factor
    for i in df.index:
        if ema_close_open_spread_momentum.loc[i] > 0:
            final_factor.loc[i] = enhanced_factor.loc[i] / ema_close_open_spread_momentum.loc[i]
        else:
            final_factor.loc[i] = enhanced_factor.loc[i] * ema_close_open_spread_momentum.loc[i]
    
    return final_factor
