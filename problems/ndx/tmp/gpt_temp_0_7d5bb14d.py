import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Exponential Momentum
    df['exp_momentum'] = (df['close'] / df['close'].shift(30)) - 1

    # Calculate 30-Day Average True Range for Stability
    df['true_range'] = df[['high', 'low', 'close']].apply(
        lambda x: max(x['high'] - x['low'], abs(x['high'] - x['close'].shift(1)), abs(x['close'].shift(1) - x['low'])), 
        axis=1
    )
    df['atr_30'] = df['true_range'].rolling(window=30).mean()

    # Calculate Short-Term and Long-Term Exponential Moving Averages
    df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
    df['ema_200'] = df['close'].ewm(span=200, adjust=False).mean()

    # Derive Moving Average Crossover Signal
    df['crossover_signal'] = np.where(df['ema_50'] > df['ema_200'], 1, np.where(df['ema_50'] < df['ema_200'], -1, 0))

    # Adjust Momentum Factor with Inverse of Volatility
    df['adjusted_momentum'] = df['exp_momentum'] * (1 / df['atr_30'])

    # Incorporate Volume into the Momentum Adjustment
    df['avg_volume_30'] = df['volume'].rolling(window=30).mean()
    df['volume_factor'] = df['volume'] / df['avg_volume_30']
    df['momentum_volume_adjusted'] = df['adjusted_momentum'] * df['volume_factor']

    # Integrate Price Position Relative to Exponential Moving Averages
    df['price_position_to_emas'] = np.where(df['close'] > df['ema_50'], 
                                            np.where(df['close'] > df['ema_200'], 1, 0), 
                                            np.where(df['close'] < df['ema_50'], 
                                                    np.where(df['close'] < df['ema_200'], -1, 0), 0)
                                           )

    # Final Alpha Factor Composition
    df['alpha_factor'] = df['momentum_volume_adjusted'] + df['crossover_signal'] + df['price_position_to_emas']

    return df['alpha_factor']
