import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Daily VWAP
    df['total_volume'] = df['volume']
    df['total_dollar_value'] = df['close'] * df['volume']
    df['vwap'] = df['total_dollar_value'].rolling(window=1, min_periods=1).sum() / df['total_volume'].rolling(window=1, min_periods=1).sum()

    # Calculate VWAP Deviation
    df['vwap_deviation'] = df['close'] - df['vwap']

    # Calculate Cumulative VWAP Deviation
    df['cumulative_vwap_deviation'] = df['vwap_deviation'].cumsum()

    # Integrate Adaptive Short-Term Momentum
    def determine_dynamic_period(volatility, short_threshold, medium_threshold):
        if volatility < short_threshold:
            return 5  # Short-term
        elif volatility < medium_threshold:
            return 20  # Medium-term
        else:
            return 60  # Long-term

    # Calculate Intraday Volatility
    df['high_low_range'] = df['high'] - df['low']
    df['abs_vwap_deviation'] = (df['close'] - df['vwap']).abs()
    df['intraday_volatility'] = df['high_low_range'] + df['abs_vwap_deviation']

    # Determine Dynamic Periods based on recent volatility
    df['volatility'] = df['close'].rolling(window=5, min_periods=1).std()
    short_threshold = df['volatility'].quantile(0.33)
    medium_threshold = df['volatility'].quantile(0.66)

    df['short_term_period'] = df['volatility'].apply(lambda x: determine_dynamic_period(x, short_threshold, medium_threshold))
    df['medium_term_period'] = df['volatility'].apply(lambda x: determine_dynamic_period(x, short_threshold, medium_threshold))
    df['long_term_period'] = df['volatility'].apply(lambda x: determine_dynamic_period(x, short_threshold, medium_threshold))

    # Calculate Momentum for the chosen period
    def calculate_momentum(deviations, period):
        return deviations.rolling(window=period, min_periods=1).sum()

    df['short_term_momentum'] = calculate_momentum(df['vwap_deviation'], df['short_term_period'])
    df['medium_term_momentum'] = calculate_momentum(df['vwap_deviation'], df['medium_term_period'])
    df['long_term_momentum'] = calculate_momentum(df['vwap_deviation'], df['long_term_period'])

    # Combine all momentum and cumulative VWAP deviation
    df['combined_factor'] = (df['cumulative_vwap_deviation'] 
                             + df['short_term_momentum'] 
                             + df['medium_term_momentum'] 
                             + df['long_term_momentum'] 
                             + df['intraday_volatility'])

    return df['combined_factor']

# Example usage:
# df = pd.DataFrame({
#     'open': [10, 11, 12, 13, 14],
#     'high': [11, 12, 13, 14, 15],
#     'low': [9, 10, 11, 12, 13],
#     'close': [10.5, 11.5, 12.5, 13.5, 14.5],
#     'amount': [1000, 1100, 1200, 1300, 1400],
#     'volume': [100, 110, 120, 130, 140]
# })
# alpha_factor = heuristics_v2(df)
# print(alpha_factor)
