import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Raw Returns
    df['raw_returns'] = df['close'].pct_change()

    # Momentum Component
    df['10_day_ema_returns'] = df['raw_returns'].ewm(span=10, adjust=False).mean()
    df['5_day_ema_returns'] = df['raw_returns'].ewm(span=5, adjust=False).mean()
    df['momentum_component'] = df['10_day_ema_returns'] - df['5_day_ema_returns']

    # Reversal Component
    df['21_day_wma_returns'] = (df['raw_returns'] * np.arange(1, 22)).rolling(window=21).sum() / (21 * 22) / 2
    df['reversal_component'] = -1 * df['21_day_wma_returns']

    # Scale Components
    momentum_max = df['momentum_component'].abs().max()
    reversal_max = df['reversal_component'].abs().max()

    df['scaled_momentum'] = df['momentum_component'] / momentum_max
    df['scaled_reversal'] = df['reversal_component'] / reversal_max

    # Combine Scaled Components
    df['combined_factor'] = (df['scaled_momentum'] + df['scaled_reversal']) * np.sqrt(df['volume'])

    return df['combined_factor']
