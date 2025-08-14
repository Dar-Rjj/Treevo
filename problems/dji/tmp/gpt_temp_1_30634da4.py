import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Intraday High-Low Spread
    intraday_high_low_spread = df['high'] - df['low']
    
    # Calculate Intraday Open-Close Return
    intraday_open_close_return = (df['close'] - df['open']) / df['open']
    
    # Calculate Momentum Indicator
    daily_returns = (df['close'].pct_change()).fillna(0)
    weights = np.sqrt(df['volume'])
    weighted_moving_average = (daily_returns.rolling(window=10).apply(lambda x: (x * weights.iloc[x.index]).sum() / weights.iloc[x.index].sum(), raw=False))
    
    # Calculate Price Deviation
    median_price = (df['open'] + df['high'] + df['low'] + df['close']) / 4
    price_deviation = df['close'] - median_price
    
    # Synthesize Combined Factors
    combined_intraday_factors = intraday_high_low_spread + (intraday_open_close_return * np.log(df['volume']))
    
    combined_momentum_and_price_deviation = (weighted_moving_average * np.sqrt(df['volume'])) + price_deviation
    
    final_alpha_factor = combined_intraday_factors + combined_momentum_and_price_deviation + (df['close'].shift(1) * np.log(df['volume']).ewm(span=10, adjust=False).mean())
    
    return final_alpha_factor
