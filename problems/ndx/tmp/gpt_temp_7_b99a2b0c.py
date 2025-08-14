import pandas as pd
import pandas as pd

def heuristics_v2(df, n=20):
    # Momentum factor based on price trends
    df['Close_t_minus_n'] = df['close'].shift(n)
    df['Momentum'] = (df['close'] - df['Close_t_minus_n']) / df['Close_t_minus_n']

    # Short-term and long-term momentum
    short_term_window = 5
    long_term_window = 20
    df['ShortTermMomentum'] = (df['close'].rolling(window=short_term_window).mean() / df['close'].shift(short_term_window-1) - 1)
    df['LongTermMomentum'] = (df['close'].rolling(window=long_term_window).mean() / df['close'].shift(long_term_window-1) - 1)
    df['RelativeMomentum'] = df['ShortTermMomentum'] - df['LongTermMomentum']

    # Volatility factor using price movement variability
    df['DailyReturn'] = (df['close'] - df['close'].shift(1)) / df['close'].shift(1)
    df['Volatility'] = df['DailyReturn'].rolling(window=n).std()

    # Volume and price change relationship
    df['UpDay'] = (df['close'] > df['open']).astype(int)
    df['DownDay'] = (df['close'] < df['open']).astype(int)
    avg_volume_up_days = df[df['UpDay'] == 1]['volume'].rolling(window=n).mean()
    avg_volume_down_days = df[df['DownDay'] == 1]['volume'].rolling(window=n).mean()
    df['VolumeRatio'] = avg_volume_up_days / avg_volume_down_days

    # High and low prices impact
    df['Range'] = df['high'] - df['low']
    df['RangeOverClose'] = df['Range'] / df['close']

    # Trading volume and price correlation
    df['VolumeChange'] = (df['volume'] - df['volume'].shift(1)) / df['volume'].shift(1)
    df['PriceVolumeCorrelation'] = df['VolumeChange'].rolling(window=n).corr(df['DailyReturn'])

    # Gap openings effect
    df['GapSize'] = df['open'] - df['close'].shift(1)
    df['GapDirection'] = (df['open'] > df['close'].shift(1)).astype(int)

    # Combine all factors into a single alpha factor
    df['AlphaFactor'] = (
        df['Momentum'] +
        df['RelativeMomentum'] +
        df['Volatility'] * -1 +
        df['VolumeRatio'] +
        df['RangeOverClose'] +
        df['PriceVolumeCorrelation'] +
        df['GapSize'] * df['GapDirection']
    )

    return df['AlphaFactor'].dropna()
