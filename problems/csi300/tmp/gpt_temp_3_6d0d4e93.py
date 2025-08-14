import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Adjusted Price Momentum
    df['daily_change'] = df['close'] - df['close'].shift(1)
    df['3d_avg_change'] = df['daily_change'].rolling(window=3).mean()
    df['7d_avg_change'] = df['daily_change'].rolling(window=7).mean()
    df['21d_avg_change'] = df['daily_change'].rolling(window=21).mean()

    # Volume-Adjusted Momentum and Liquidity
    n = 21
    df['long_term_momentum'] = (df['close'] - df['close'].shift(n)) / df['close'].shift(n)
    df['volume_ratio'] = df['volume'] / df['volume'].shift(n)
    df['volume_adjusted_momentum'] = df['long_term_momentum'] * df['volume_ratio']

    df['3d_avg_volume'] = df['volume'].rolling(window=3).mean()
    df['7d_avg_volume'] = df['volume'].rolling(window=7).mean()
    df['21d_avg_volume'] = df['volume'].rolling(window=21).mean()

    # Enhanced Intraday Momentum
    df['high_indicator'] = df['high'] - df['open']
    df['low_indicator'] = df['open'] - df['low']
    df['avg_intraday_momentum'] = (df['high_indicator'] + df['low_indicator']) / 2
    df['volume_ma'] = df['volume'].rolling(window=21).mean()
    df['volume_normalized'] = df['volume'] / df['volume_ma']
    df['intraday_momentum_adjusted'] = df['avg_intraday_momentum'] * df['volume_normalized']

    # Gaps and Breaks
    df['up_gap'] = (df['open'] - df['close'].shift(1)) > 0
    df['down_gap'] = (df['open'] - df['close'].shift(1)) < 0
    df['gap_intensity_up'] = df['open'] - df['close'].shift(1)
    df['gap_intensity_down'] = df['open'] - df['close'].shift(1)

    df['upper_breakout'] = df['high'] > df['high'].rolling(window=21).max().shift(1)
    df['lower_breakout'] = df['low'] < df['low'].rolling(window=21).min().shift(1)
    df['breakout_intensity_up'] = df['high'] - df['high'].rolling(window=21).max().shift(1)
    df['breakout_intensity_down'] = df['low'].rolling(window=21).min().shift(1) - df['low']

    # Intraday Patterns
    df['intraday_range'] = df['high'] - df['low']
    df['intraday_momentum'] = df['close'] - df['open']
    df['intraday_strength'] = df['intraday_momentum'] / df['intraday_range']
    df['intraday_volatility'] = df[['high', 'low']].rolling(window=21).std()

    # Add Volume Spike Filter
    spike_threshold = 1.5
    df['volume_spike'] = df['volume'] / df['volume'].shift(1)
    df['volume_spike_filtered'] = (df['volume_spike'] > spike_threshold).astype(int)

    # Final Alpha Factor Synthesis
    df['adjusted_close'] = df['3d_avg_change'] + df['7d_avg_change'] + df['21d_avg_change']
    df['volume_impacted_intraday_momentum'] = df['intraday_momentum_adjusted']
    df['volume_adjusted_long_term_momentum'] = df['volume_adjusted_momentum']
    df['total_factor'] = (df['adjusted_close'] + df['volume_impacted_intraday_momentum'] + df['volume_adjusted_long_term_momentum']) / df['open']
    df['final_alpha'] = df['total_factor'] * df['volume_spike_filtered']

    return df['final_alpha']
