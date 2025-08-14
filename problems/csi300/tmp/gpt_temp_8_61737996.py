import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # High-Low Range Momentum
    df['daily_range'] = df['high'] - df['low']
    df['range_momentum'] = (df['daily_range'] > df['daily_range'].shift(1)).astype(int) * 2 - 1
    
    # Volume-Amount Ratio Trend
    df['volume_amount_ratio'] = df['volume'] / df['amount']
    df['volume_amount_ratio_5d_sum'] = df['volume_amount_ratio'].rolling(window=5).sum()
    df['volume_amount_ratio_trend'] = (df['volume_amount_ratio_5d_sum'] > df['volume_amount_ratio_5d_sum'].shift(5)).astype(int) * 2 - 1
    
    # Volume Trend
    df['volume_5d_sum'] = df['volume'].rolling(window=5).sum()
    df['volume_trend'] = (df['volume_5d_sum'] > df['volume_5d_sum'].shift(5)).astype(int) * 2 - 1
    
    # Intraday High-Low Ratio
    df['intraday_high_low_ratio'] = (df['high'] - df['low']) / df['low']
    
    # Close-to-Open Return
    df['close_to_open_return'] = (df['close'] - df['open']) / df['open']
    
    # Synthesize Factors
    df['high_low_range_momentum_x_volume_amount_ratio_trend'] = df['range_momentum'] * df['volume_amount_ratio_trend']
    df['volume_trend_x_intraday_high_low_ratio'] = df['volume_trend'] * df['intraday_high_low_ratio']
    
    # Average True Range (ATR)
    df['tr'] = df[['high' - 'low', abs('high' - 'close').shift(1), abs('low' - 'close').shift(1)]].max(axis=1)
    df['atr_10d'] = df['tr'].rolling(window=10).mean()
    
    df['close_to_open_return_x_intraday_high_low_ratio_div_atr'] = (df['close_to_open_return'] * df['intraday_high_low_ratio']) / (1 + df['atr_10d'])
    
    # Adjusted Price Momentum
    df['10_day_average_return'] = df['close'].pct_change().rolling(window=10).mean()
    df['5_day_vol_ave'] = df['volume'].rolling(window=5).mean()
    df['1_day_vol_change'] = df['volume'] - df['5_day_vol_ave']
    df['adjusted_price_momentum'] = df['10_day_average_return'] * df['1_day_vol_change'].apply(lambda x: -1 if x < 0 else 1)
    
    # Final Factor Combination
    df['final_factor'] = (
        df['high_low_range_momentum_x_volume_amount_ratio_trend'] +
        df['volume_trend_x_intraday_high_low_ratio'] +
        df['close_to_open_return_x_intraday_high_low_ratio_div_atr'] +
        df['adjusted_price_momentum']
    )
    
    # Incorporate Open-to-Close Return
    df['open_to_close_return'] = (df['close'] - df['open']) / df['open']
    df['volume_weighted_high_low_range'] = (df['high'] - df['low']) * df['volume']
    df['open_to_close_return_x_volume_weighted_high_low_range'] = df['open_to_close_return'] * df['volume_weighted_high_low_range']
    
    # Final Alpha Factor
    df['final_alpha_factor'] = (
        df['final_factor'] +
        df['open_to_close_return_x_volume_weighted_high_low_range']
    )
    
    return df['final_alpha_factor']

# Example usage:
# data = pd.read_csv('your_data.csv', parse_dates=['date'], index_col='date')
# alpha_factor = heuristics_v2(data)
