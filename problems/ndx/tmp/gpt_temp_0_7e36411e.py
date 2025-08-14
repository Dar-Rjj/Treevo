import pandas as pd
import numpy as np
import pandas as pd

def heuristics_v2(df):
    # Calculate Intraday Volatility and Momentum Impact
    df['intraday_range'] = df['high'] - df['low']
    df['price_change'] = df['close'] - df['close'].shift(1)
    df['volume_impact'] = df['volume'] * df['price_change'].abs()

    # Integrate Historical Momentum and Volatility
    df['momentum_impact'] = df['volume_impact']
    df['sum_momentum_impact'] = df['momentum_impact'].rolling(window=5).sum()
    df['max_min_change'] = (df['high'].rolling(window=5).max() - df['low'].rolling(window=5).min()).apply(lambda x: 1 if x > 0 else 0)
    df['enhanced_momentum_impact'] = df['sum_momentum_impact'] * df['max_min_change']
    df['avg_intraday_range'] = df['intraday_range'].rolling(window=5).mean()
    df['deviation_intraday_range'] = df['intraday_range'] - df['avg_intraday_range']

    # Compute Volume-Weighted Average Price (VWAP)
    df['vwap_total'] = (df['open'] + df['high'] + df['low'] + df['close']) * df['volume']
    df['cumulative_volume'] = df['volume'].expanding().sum()
    df['vwap'] = df['vwap_total'].expanding().sum() / df['cumulative_volume']

    # Analyze VWAP Deviation from Close
    df['vwap_deviation'] = df['vwap'] - df['close']

    # Construct Integrated Alpha Factor Components
    df['adjusted_intraday_range'] = df['intraday_range'] * df['volume']
    df['preliminary_alpha_factor'] = (
        df['adjusted_intraday_range'] + 
        df['deviation_intraday_range'] + 
        df['vwap_deviation']
    ) * df['close']

    # Adjust for Short-Term Volatility
    df['volatility_threshold'] = ((df['high'] - df['low']) / df['close']).rolling(window=5).mean()
    df['volatility_adjusted_alpha_factor'] = df['preliminary_alpha_factor'].apply(
        lambda x: x * 1.1 if x > df['volatility_threshold'] else x * 0.9
    )

    # Incorporate Long-Term Trend
    df['20_day_ma'] = df['close'].rolling(window=20).mean()
    df['long_term_trend'] = df['20_day_ma'].diff()
    df['long_term_adjusted_alpha_factor'] = df['volatility_adjusted_alpha_factor'].apply(
        lambda x: x * 1.1 if df['long_term_trend'][-1] > 0 else x * 0.9
    )

    # Evaluate Intraday and Overnight Market Signals
    df['intraday_high_low_ratio'] = df['high'] / df['low']
    df['intraday_close_open_ratio'] = df['close'] / df['open']
    df['overnight_return'] = df['open'] / df['close'].shift(1)
    df['log_volume'] = df['volume'].apply(lambda x: np.log(x) if x > 0 else 0)

    # Synthesize Final Alpha Factor
    df['final_alpha_factor'] = (
        df['long_term_adjusted_alpha_factor'] + 
        df['intraday_high_low_ratio'] + 
        df['intraday_close_open_ratio'] + 
        df['overnight_return']
    ) * df['log_volume']

    return df['final_alpha_factor']

# Example usage:
# df = pd.read_csv('stock_data.csv', parse_dates=True, index_col='date')
# alpha_factor = heuristics_v2(df)
# print(alpha_factor)
