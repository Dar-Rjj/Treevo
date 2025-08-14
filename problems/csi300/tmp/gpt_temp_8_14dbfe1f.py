import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Long-Term (10-day) and Short-Term (5-day) Metrics
    def calculate_weighted_price_difference(df, window):
        return (df['high'] - df['low']) * df['volume'].rolling(window=window).sum() / df['volume'].rolling(window=window).sum()

    def determine_momentum(df, window):
        return df['close'] - df['open'].shift(window)

    def calculate_cumulative_weighted_average(df, window):
        return calculate_weighted_price_difference(df, window) / df['volume'].rolling(window=window).sum()

    def average_daily_return(df, window):
        return (df['close'] - df['close'].shift(1)).rolling(window=window).mean()

    def ewm_std(df, window, span):
        daily_returns = (df['close'] - df['close'].shift(1)) / df['close'].shift(1)
        return daily_returns.ewm(span=span, adjust=False).std()

    # 10-Day Metrics
    long_term_wpd = calculate_cumulative_weighted_average(df, 10)
    long_term_momentum = determine_momentum(df, 10)
    long_term_avg_daily_return = average_daily_return(df, 10)
    long_term_ewm_std = ewm_std(df, 10, 0.94)

    # 5-Day Metrics
    short_term_wpd = calculate_cumulative_weighted_average(df, 5)
    short_term_momentum = determine_momentum(df, 5)
    short_term_avg_daily_return = average_daily_return(df, 5)
    short_term_ewm_std = ewm_std(df, 5, 0.94)

    # Final 10-Day Alpha Factor
    long_term_alpha = (long_term_wpd + long_term_momentum - long_term_avg_daily_return) / long_term_ewm_std

    # Final 5-Day Alpha Factor
    short_term_alpha = (short_term_wpd + short_term_momentum - short_term_avg_daily_return) / short_term_ewm_std

    # Dynamic Weights Based on Recent Volatility
    relative_volatility_ratio = short_term_ewm_std / long_term_ewm_std
    long_term_weight = 1 / (1 + relative_volatility_ratio)
    short_term_weight = relative_volatility_ratio / (1 + relative_volatility_ratio)

    # Integrate Trading Volume Effectively
    five_day_avg_volume = df['volume'].rolling(window=5).mean()
    combined_alpha = (long_term_weight * long_term_alpha + short_term_weight * short_term_alpha) * five_day_avg_volume

    # Final Alpha Factor
    final_alpha_factor = combined_alpha

    return final_alpha_factor.dropna()
