import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import linregress

def heuristics_v2(df, N=5, p=10, q=10, m=14):
    # Calculate Daily Returns
    df['daily_return'] = df['close'].pct_change()

    # Calculate Volume-Weighted Average Price (VWAP)
    df['vwap'] = (df['high'] + df['low']) / 2 * df['volume']
    vwap_sum = df['vwap'].rolling(window=N).sum()
    volume_sum = df['volume'].rolling(window=N).sum()
    df['vwap'] = vwap_sum / volume_sum

    # Calculate Volume-Weighted Momentum
    df['return_volume'] = df['daily_return'] * df['volume']
    return_volume_sum = df['return_volume'].rolling(window=N).sum()
    volume_sum = df['volume'].rolling(window=N).sum()
    df['vwam'] = return_volume_sum / volume_sum

    # Smooth the Daily Return using VWAP
    df['vwap_return'] = (df['close'] - df['vwap']) / df['vwap']
    df['smoothed_return'] = df['vwap_return'].ewm(span=5).mean()
    df['alpha_factor_vwam'] = df['smoothed_return'] * df['volume']

    # Calculate Price Momentum
    df['sma_close'] = df['close'].rolling(window=N).mean()
    df['price_diff'] = df['close'] - df['sma_close']
    df['momentum_score'] = df['price_diff'] / df['sma_close']

    # Calculate Adjusted Price Momentum
    df['cumulative_volume'] = df['volume'].rolling(window=N).sum()
    max_cumulative_volume = df['cumulative_volume'].rolling(window=N).max()
    df['normalized_volume'] = df['cumulative_volume'] / max_cumulative_volume
    df['adjusted_momentum'] = df['momentum_score'] * df['normalized_volume']

    # Calculate High-to-Low Price Range
    df['price_range'] = df['high'] - df['low']

    # Calculate Trading Intensity
    df['volume_change'] = df['volume'].diff()
    df['amount_change'] = df['amount'].diff()
    df['trading_intensity'] = df['volume_change'] / df['amount_change']

    # Combine Volume and Amount Momentum
    df['combined_momentum'] = df['volume_change'] + df['amount_change']

    # Weight the Range by Combined Momentum
    df['weighted_range'] = (df['combined_momentum'] / 1000) * df['price_range']

    # Adjust for Price Volatility
    df['true_range'] = df[['high', 'low']].apply(lambda x: max(x['high'] - x['low'], abs(x['high'] - df['close'].shift(1)), abs(x['low'] - df['close'].shift(1))), axis=1)
    df['atr'] = df['true_range'].rolling(window=m).mean()
    df['enhanced_atr'] = df['atr'] * (1 + 0.5 * (df['high'] - df['low']) / df['close'].shift(1))
    df['volatility_adjusted_weighted_range'] = df['weighted_range'] - df['enhanced_atr']

    # Incorporate Trend Strength
    df['moving_average'] = df['close'].rolling(window=p).mean()
    trend_slopes = [linregress(df['close'][i-q+1:i+1].index, df['close'][i-q+1:i+1]).slope if i >= q else np.nan for i in range(len(df))]
    df['trend_slope'] = trend_slopes
    df['trend_strength'] = df['weighted_range'] * df['trend_slope']

    # Combine Adjusted Momentum, Weighted Range, and VWAM
    df['alpha_factor'] = df['adjusted_momentum'] + df['volatility_adjusted_weighted_range'] + (df['vwam'] * df['smoothed_return']) + df['trend_strength']

    return df['alpha_factor']
