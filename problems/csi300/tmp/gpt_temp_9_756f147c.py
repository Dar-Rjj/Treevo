import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Daily Return
    df['daily_return'] = df['close'].pct_change()

    # Compute Exponential Moving Average (EMA) of Returns
    ema_param = 10
    df['ema_returns'] = df['daily_return'].ewm(span=ema_param, adjust=False).mean()

    # Identify High and Low Volatility Days
    df['true_range'] = df[['high', 'low']].sub(df['close'].shift(), axis=0).abs().max(axis=1)
    atr_period = 14
    df['atr'] = df['true_range'].rolling(window=atr_period).mean()
    volatility_threshold = df['atr'].quantile(0.75)  # Example threshold
    df['is_high_volatility'] = df['atr'] > volatility_threshold

    # Filter Days by Volatility
    high_vol_days = df[df['is_high_volatility']]
    low_vol_days = df[~df['is_high_volatility']]

    # Calculate Volume Weighted Momentum
    df['volume_weighted_momentum'] = df['daily_return'] * df['volume']
    high_vol_momentum = (high_vol_days['volume_weighted_momentum']).sum() / high_vol_days['volume'].sum()
    low_vol_momentum = (low_vol_days['volume_weighted_momentum']).sum() / low_vol_days['volume'].sum()

    # Compute Momentum Difference
    momentum_difference = high_vol_momentum - low_vol_momentum

    # Adjust for Overall Market Trend
    market_index_close = df['close']  # Assuming close price is the market index
    market_index_return = market_index_close.pct_change()
    adjusted_momentum = momentum_difference - market_index_return[-1]

    # Generate Final Alpha Factor
    alpha_factor = adjusted_momentum

    return alpha_factor
