import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # High-Low Range Momentum
    df['range'] = df['high'] - df['low']
    df['range_momentum'] = (df['range'] > df['range'].shift(1)).astype(int) - (df['range'] <= df['range'].shift(1)).astype(int)
    
    # Volume-Amount Ratio Trend
    df['volume_amount_ratio'] = df['volume'] / df['amount']
    df['rolling_volume_amount_ratio_5'] = df['volume_amount_ratio'].rolling(window=5).sum()
    df['volume_amount_trend'] = (df['rolling_volume_amount_ratio_5'] > df['rolling_volume_amount_ratio_5'].shift(5)).astype(int) - (df['rolling_volume_amount_ratio_5'] <= df['rolling_volume_amount_ratio_5'].shift(5)).astype(int)
    
    # Volume Trend
    df['rolling_volume_5'] = df['volume'].rolling(window=5).sum()
    df['volume_trend'] = (df['rolling_volume_5'] > df['rolling_volume_5'].shift(5)).astype(int) - (df['rolling_volume_5'] <= df['rolling_volume_5'].shift(5)).astype(int)
    
    # Intraday High-Low Ratio
    df['intraday_high_low_ratio'] = (df['high'] - df['low']) / df['low']
    
    # Close-to-Open Return
    df['close_to_open_return'] = (df['close'] - df['open']) / df['open']
    
    # Volume Weighted Return
    df['volume_weighted_return'] = df['volume'] * df['close_to_open_return']
    
    # Average True Range
    df['true_range'] = df[['high', 'low', 'close']].rolling(window=2).apply(lambda x: max(x) - min(x), raw=True)
    df['average_true_range'] = df['true_range'].rolling(window=5).mean()
    
    # Synthesize Factors
    df['high_low_momentum_and_volume_amount_trend'] = df['range_momentum'] * df['volume_amount_trend']
    df['volume_trend_and_intraday_high_low_ratio'] = df['volume_trend'] * df['intraday_high_low_ratio']
    df['volume_weighted_return_adjusted'] = (df['volume_weighted_return'] * df['intraday_high_low_ratio']) / (1 + df['average_true_range'])
    
    # Adjusted Price Momentum
    df['10_day_avg_return'] = df['close'].pct_change().rolling(window=10).mean()
    df['5_day_vol_ma'] = df['volume'].rolling(window=5).mean()
    df['1_day_vol_change'] = df['volume'] - df['5_day_vol_ma']
    df['volume_reversal'] = df['10_day_avg_return'] * (-1 * (df['1_day_vol_change'] < 0) + 1 * (df['1_day_vol_change'] >= 0))
    
    # Final Factor Combination
    df['final_factor'] = df['high_low_momentum_and_volume_amount_trend'] + df['volume_trend_and_intraday_high_low_ratio'] + df['volume_weighted_return_adjusted'] + df['volume_reversal']
    
    return df['final_factor'].dropna()
