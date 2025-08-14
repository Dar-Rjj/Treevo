import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Intraday Momentum
    df['intraday_return'] = (df['close'] - df['open']) / df['open']
    intraday_momentum = df['intraday_return'].rolling(window=5).mean()
    intraday_momentum = intraday_momentum.ewm(span=5, adjust=False).mean()

    # Evaluate Liquidity
    avg_volume = df['volume'].rolling(window=10).mean()
    price_impact = df['close'].diff().abs()
    liquidity = avg_volume * price_impact / 1000  # Scale by a constant

    # Calculate Daily Price Momentum
    ema_12 = df['close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['close'].ewm(span=26, adjust=False).mean()

    # Adjust Momentum by Volume-Weighted True Range
    true_range = df[['high', 'low', 'close']].join(df['close'].shift(1))
    true_range = true_range.rename(columns={'close': 'prev_close'})
    true_range['tr1'] = true_range['high'] - true_range['low']
    true_range['tr2'] = abs(true_range['high'] - true_range['prev_close'])
    true_range['tr3'] = abs(true_range['low'] - true_range['prev_close'])
    true_range['true_range'] = true_range[['tr1', 'tr2', 'tr3']].max(axis=1)
    tr_sma_14 = true_range['true_range'].rolling(window=14).mean()
    volume_change = df['volume'].diff().rolling(window=14).mean()
    vol_weighted_tr = tr_sma_14 * abs(volume_change)

    momentum_adjustment = (ema_12 - ema_26) / vol_weighted_tr

    # Combine Intraday Momentum, Liquidity, and Price Momentum
    alpha_factor = intraday_momentum + intraday_momentum * liquidity + momentum_adjustment

    # Generate Alpha Factor
    alpha_factor = alpha_factor.clip(lower=0)
    alpha_factor = alpha_factor.where(alpha_factor > 0.01, 0)  # Apply threshold for stability

    return alpha_factor
