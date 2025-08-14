import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate Daily VWAP
    df['total_volume'] = df['volume']
    df['total_dollar_value'] = df['volume'] * df['close']
    df['vwap'] = df.groupby(df.index.date)['total_dollar_value'].transform('sum') / df.groupby(df.index.date)['total_volume'].transform('sum')
    
    # Calculate VWAP Deviation
    df['vwap_deviation'] = df['close'] - df['vwap']
    
    # Calculate Cumulative VWAP Deviation
    df['cumulative_vwap_deviation'] = df['vwap_deviation'].cumsum()
    
    # Integrate Adaptive Short-Term Momentum (5 days)
    short_term_momentum_period = 5
    df['short_term_return'] = df['close'].pct_change(short_term_momentum_period)
    
    # Integrate Adaptive Medium-Term Momentum (10 days)
    medium_term_momentum_period = 10
    df['medium_term_return'] = df['close'].pct_change(medium_term_momentum_period)
    
    # Calculate Intraday Volatility
    df['high_low_range'] = df['high'] - df['low']
    df['abs_vwap_deviation'] = abs(df['close'] - df['vwap'])
    df['intraday_volatility'] = df['high_low_range'] + df['abs_vwap_deviation']
    
    # Integrate Long-Term Trend (20-Day Moving Average)
    df['long_term_trend'] = df['close'] - df['close'].rolling(window=20).mean()
    
    # Final Alpha Factor
    df['alpha_factor'] = (
        0.3 * df['cumulative_vwap_deviation'] +
        0.2 * df['short_term_return'] +
        0.2 * df['medium_term_return'] +
        0.2 * df['intraday_volatility'] +
        0.1 * df['long_term_trend']
    )
    
    return df['alpha_factor']

# Example usage:
# df = pd.DataFrame({
#     'open': [10, 10.5, 11, 10.8, 11.2, 11.5, 12, 12.5, 13, 13.5],
#     'high': [11, 11.5, 12, 11.9, 12.2, 12.5, 13, 13.5, 14, 14.5],
#     'low': [9, 9.5, 10, 10.7, 11, 11.3, 11.8, 12.3, 12.8, 13.3],
#     'close': [10.5, 11, 11.5, 12, 12.5, 13, 13.5, 14, 14.5, 15],
#     'amount': [1000, 1050, 1100, 1150, 1200, 1250, 1300, 1350, 1400, 1450],
#     'volume': [100, 110, 120, 130, 140, 150, 160, 170, 180, 190]
# })
# df.index = pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05',
#                            '2023-01-06', '2023-01-07', '2023-01-08', '2023-01-09', '2023-01-10'])
# alpha_factor = heuristics_v2(df)
# print(alpha_factor)
