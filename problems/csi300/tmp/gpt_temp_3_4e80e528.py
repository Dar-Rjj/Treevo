import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Momentum factor
    df['close_5d_pct_change'] = (df['close'] / df['close'].shift(5) - 1)
    df['close_10d_pct_change'] = (df['close'] / df['close'].shift(10) - 1)
    df['close_20d_pct_change'] = (df['close'] / df['close'].shift(20) - 1)
    df['momentum_factor'] = (df['close_5d_pct_change'] * 0.5 + 
                             df['close_10d_pct_change'] * 0.3 + 
                             df['close_20d_pct_change'] * 0.2)

    # Volatility factor (ATR)
    df['true_range'] = df[['high', 'low', 'close']].apply(
        lambda x: max(x['high'] - x['low'], abs(x['high'] - x['close'].shift()), abs(x['low'] - x['close'].shift())), axis=1)
    df['atr_14d'] = df['true_range'].rolling(window=14).mean()

    # Volume-based factor
    df['avg_volume_30d'] = df['volume'].rolling(window=30).mean()
    df['volume_ratio'] = df['volume'] / df['avg_volume_30d']
    df['high_volume_days'] = (df['volume_ratio'] > 2.5).astype(int).rolling(window=10).sum()

    # Reversal indicator
    df['20d_high'] = df['high'].rolling(window=20).max()
    df['20d_low'] = df['low'].rolling(window=20).min()
    df['reversal_score'] = (abs(df['close'] - df['20d_high']) / df['20d_high'] < 0.05).astype(int) - \
                           (abs(df['close'] - df['20d_low']) / df['20d_low'] < 0.05).astype(int)
    
    # Price-to-volume ratio
    df['price_to_volume_ratio'] = df['close'] / df['volume']
    df['pvr_10d_ma'] = df['price_to_volume_ratio'].rolling(window=10).mean()
    df['pvr_signal'] = df['price_to_volume_ratio'] - df['pvr_10d_ma']

    # Trend consistency factor
    df['up_day'] = (df['close'] > df['open']).astype(int)
    df['down_day'] = (df['close'] < df['open']).astype(int)
    df['consecutive_up'] = df['up_day'].groupby((df['up_day'] != df['up_day'].shift()).cumsum()).cumsum()
    df['consecutive_down'] = df['down_day'].groupby((df['down_day'] != df['down_day'].shift()).cumsum()).cumsum()
    df['trend_strength'] = (df['consecutive_up'] > 5).astype(int) - (df['consecutive_down'] > 5).astype(int)

    # Combine factors
    df['alpha_factor'] = (df['momentum_factor'] + df['atr_14d'] + df['high_volume_days'] + 
                          df['reversal_score'] + df['pvr_signal'] + df['trend_strength'])

    return df['alpha_factor']
