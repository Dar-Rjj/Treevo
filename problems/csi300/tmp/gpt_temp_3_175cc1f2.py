import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate Raw Returns
    df['returns'] = df['close'].pct_change()
    
    # Compute 14-Day Sum of Upward and Downward Returns
    positive_returns = df[df['returns'] > 0]['returns']
    negative_returns = df[df['returns'] < 0]['returns'].abs()
    df['14_day_sum_up'] = positive_returns.rolling(window=14).sum()
    df['14_day_sum_down'] = negative_returns.rolling(window=14).sum()

    # Calculate Relative Strength
    df['relative_strength'] = df['14_day_sum_up'] / df['14_day_sum_down']

    # Smooth with Exponential Moving Average on Volume
    df['ema_volume'] = df['volume'].ewm(span=14, adjust=False).mean()
    df['smoothed_relative_strength'] = df['relative_strength'] * df['ema_volume']

    # Adjust Relative Strength with Price Trend
    df['21_sma_close'] = df['close'].rolling(window=21).mean()
    df['price_trend_adjustment'] = df['close'] / df['21_sma_close']
    df['adjusted_relative_strength'] = df['smoothed_relative_strength'] * df['price_trend_adjustment']

    # VWAP Calculation
    df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
    df['vwap'] = (df['typical_price'] * df['volume']).cumsum() / df['volume'].cumsum()

    # Integrate VWAP Differences
    df['vwap_diff_from_open'] = df['vwap'] - df['open']
    df['vwap_diff_from_close'] = df['vwap'] - df['close']

    # Combine VWAP Differences with Adjusted Relative Strength
    df['vwap_diff_from_open_adj'] = df['vwap_diff_from_open'] * df['adjusted_relative_strength']
    df['vwap_diff_from_close_adj'] = df['vwap_diff_from_close'] * df['adjusted_relative_strength']

    # Calculate High-Low Spread
    df['high_low_spread'] = df['high'] - df['low']

    # Calculate Volume-Weighted High-Low Spread
    df['vol_weighted_high_low_spread'] = df['volume'] * df['high_low_spread']

    # Combine Factors
    df['combined_factors'] = (df['adjusted_relative_strength'] + 
                              df['vwap_diff_from_open_adj'] + 
                              df['vwap_diff_from_close_adj']) / df['vol_weighted_high_low_spread']

    # Incorporate Volume Acceleration
    df['5_day_roc_volume'] = df['volume'].pct_change(periods=5)
    df['10_sma_vol_roc'] = df['5_day_roc_volume'].rolling(window=10).mean()
    df['combined_with_vol_acc'] = df['combined_factors'] * df['10_sma_vol_roc']

    # Synthesize Final Alpha Factor
    df['final_alpha_factor'] = (df['smoothed_relative_strength'] + 
                                df['combined_with_vol_acc'] + 
                                df['vwap_diff_from_open_adj'] + 
                                df['vwap_diff_from_close_adj'])

    # Incorporate Price Momentum
    df['14_day_momentum'] = df['close'].pct_change(periods=14)
    df['final_alpha_factor'] = df['final_alpha_factor'] * df['14_day_momentum']

    # Incorporate Open-Close Spread
    df['open_close_spread'] = df['open'] - df['close']
    df['final_alpha_factor'] = df['final_alpha_factor'] * df['open_close_spread']

    return df['final_alpha_factor'].dropna()
