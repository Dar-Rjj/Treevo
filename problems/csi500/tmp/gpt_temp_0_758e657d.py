import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Intraday Return
    df['intraday_return'] = (df['high'] - df['low']) / df['close']

    # Calculate Volume-Weighted Intraday Return
    df['volume_weighted_intraday_return'] = df['intraday_return'] * df['volume']
    
    # Adaptive Exponential Moving Average of Volume-Weighted Intraday Return
    def adaptive_ema(series, span=30):
        return series.ewm(span=span, adjust=False).mean()
    df['adaptive_ema_volume_weighted_intraday_return'] = adaptive_ema(df['volume_weighted_intraday_return'])

    # Calculate Intraday Volatility
    df['intraday_volatility'] = df['intraday_return'].pow(2)
    df['intraday_volatility'] = df['intraday_volatility'].rolling(window=60).sum().pow(0.5)

    # Simple Moving Average of Intraday Volatility
    df['sma_intraday_volatility'] = df['intraday_volatility'].rolling(window=60).mean()

    # Adjust for Recent Volatility
    df['recent_volatility_sma'] = df['intraday_volatility'].rolling(window=10).mean()

    # Assume we have a 'sector' column in the DataFrame
    if 'sector' not in df.columns:
        raise ValueError("Sector information is required and must be included in the DataFrame as a 'sector' column.")
    
    # Calculate Sector-Specific Intraday Volatility
    sector_volatility = df.groupby(['date', 'sector'])['intraday_return'].agg(lambda x: np.sqrt(np.sum(x**2))).reset_index(name='sector_intraday_volatility')
    df = df.merge(sector_volatility, on=['date', 'sector'], how='left')

    # Calculate Volume-Weighted Intraday Momentum
    df['volume_weighted_intraday_momentum'] = df['intraday_return'] * df['volume']

    # Adaptive Exponential Moving Average of Volume-Weighted Intraday Momentum
    df['adaptive_ema_volume_weighted_intraday_momentum'] = adaptive_ema(df['volume_weighted_intraday_momentum'])

    # Combine Factors
    df['final_factor'] = (
        df['adaptive_ema_volume_weighted_intraday_return'] +
        df['sma_intraday_volatility'] +
        df['recent_volatility_sma'] +
        df['sector_intraday_volatility'] +
        df['adaptive_ema_volume_weighted_intraday_momentum']
    )

    # Drop intermediate columns
    df.drop(columns=[
        'intraday_return', 'volume_weighted_intraday_return',
        'intraday_volatility', 'sma_intraday_volatility',
        'recent_volatility_sma', 'sector_intraday_volatility',
        'volume_weighted_intraday_momentum'
    ], inplace=True)

    return df['final_factor']
