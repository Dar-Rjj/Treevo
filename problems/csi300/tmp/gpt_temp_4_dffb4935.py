import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # 10-day calculations
    df['HL_vol_10'] = (df['high'] - df['low']) * df['volume']
    df['HL_vol_10_sum'] = df['HL_vol_10'].rolling(window=10).sum()
    df['volume_10_sum'] = df['volume'].rolling(window=10).sum()
    df['long_term_weighted_price_diff'] = df['HL_vol_10_sum'] / df['volume_10_sum']
    df['long_term_momentum'] = df['close'] - df['open'].shift(10)
    df['long_term_cumulative_avg'] = df['long_term_weighted_price_diff'] / df['volume_10_sum']
    df['avg_daily_return_10'] = (df['close'] - df['close'].shift(1)).rolling(window=10).sum() / 10
    df['daily_returns'] = (df['close'] - df['close'].shift(1)) / df['close'].shift(1)
    df['ewm_std_10'] = df['daily_returns'].ewm(span=10, adjust=False).std()
    
    # Final 10-day alpha factor
    df['alpha_factor_10'] = (df['long_term_cumulative_avg'] - df['avg_daily_return_10']) / df['ewm_std_10']

    # 5-day calculations
    df['HL_vol_5'] = (df['high'] - df['low']) * df['volume']
    df['HL_vol_5_sum'] = df['HL_vol_5'].rolling(window=5).sum()
    df['volume_5_sum'] = df['volume'].rolling(window=5).sum()
    df['short_term_weighted_price_diff'] = df['HL_vol_5_sum'] / df['volume_5_sum']
    df['short_term_momentum'] = df['close'] - df['open'].shift(5)
    df['short_term_cumulative_avg'] = df['short_term_weighted_price_diff'] / df['volume_5_sum']
    df['avg_daily_return_5'] = (df['close'] - df['close'].shift(1)).rolling(window=5).sum() / 5
    df['ewm_std_5'] = df['daily_returns'].ewm(span=5, adjust=False).std()
    
    # Final 5-day alpha factor
    df['alpha_factor_5'] = (df['short_term_cumulative_avg'] - df['avg_daily_return_5']) / df['ewm_std_5']

    # Dynamic weights
    df['relative_volatility_ratio'] = df['ewm_std_5'] / df['ewm_std_10']
    df['weight_10'] = 1 / (1 + df['relative_volatility_ratio'])
    df['weight_5'] = df['relative_volatility_ratio'] / (1 + df['relative_volatility_ratio'])

    # Combined alpha factor
    df['combined_alpha_factor'] = (df['weight_10'] * df['alpha_factor_10']) + (df['weight_5'] * df['alpha_factor_5'])

    # Adjust for volume and turnover rate
    df['volume_5_avg'] = df['volume'].rolling(window=5).mean()
    df['turnover_rate_5_avg'] = (df['volume_5_avg'] / df['free_float_shares'])  # Assume 'free_float_shares' is a column in the DataFrame
    df['final_alpha_factor'] = df['combined_alpha_factor'] * df['volume_5_avg'] * df['turnover_rate_5_avg']

    return df['final_alpha_factor'].dropna()
