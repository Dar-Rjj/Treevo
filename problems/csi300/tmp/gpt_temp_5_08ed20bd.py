import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Novel alpha factor combining volatility regime asymmetry, liquidity stress dynamics,
    and information flow hierarchy patterns.
    """
    # Volatility Regime Asymmetry
    # High-low range skew vs close-to-close volatility
    df['hl_range'] = df['high'] - df['low']
    df['cc_vol'] = df['close'].pct_change().abs()
    df['range_skew'] = (df['hl_range'] / df['close'].shift(1)).rolling(window=5).skew()
    df['vol_regime'] = df['range_skew'] / (df['cc_vol'].rolling(window=5).std() + 1e-8)
    
    # Multi-day range curvature patterns
    df['range_ma3'] = df['hl_range'].rolling(window=3).mean()
    df['range_ma5'] = df['hl_range'].rolling(window=5).mean()
    df['range_curvature'] = (df['range_ma3'] - df['range_ma5']) / (df['range_ma5'] + 1e-8)
    
    # Liquidity Stress Dynamics
    # Open-close reversal intensity with volume
    df['oc_reversal'] = np.where(
        (df['close'] > df['open']) & (df['close'].shift(1) < df['open'].shift(1)),
        (df['close'] - df['open']) / df['open'],
        np.where(
            (df['close'] < df['open']) & (df['close'].shift(1) > df['open'].shift(1)),
            (df['open'] - df['close']) / df['open'],
            0
        )
    )
    df['reversal_intensity'] = df['oc_reversal'] * df['volume'] / df['volume'].rolling(window=10).mean()
    
    # Volume-price divergence speed
    df['price_change'] = df['close'].pct_change()
    df['volume_change'] = df['volume'].pct_change()
    df['vp_divergence'] = (df['price_change'] - df['volume_change']).rolling(window=5).std()
    
    # Information Flow Hierarchy
    # Large-small trade lead-lag timing (using amount/volume as proxy for trade size)
    df['avg_trade_size'] = df['amount'] / (df['volume'] + 1e-8)
    df['large_trade_lead'] = df['avg_trade_size'].rolling(window=3).corr(df['close'].pct_change().shift(-1))
    df['small_trade_lag'] = df['avg_trade_size'].rolling(window=3).corr(df['close'].pct_change().shift(1))
    df['trade_timing'] = df['large_trade_lead'] - df['small_trade_lag']
    
    # Volume-return correlation decay rate
    corr_window = 10
    rolling_corrs = []
    for i in range(len(df)):
        if i >= corr_window - 1:
            window_data = df.iloc[i-corr_window+1:i+1]
            corr = window_data['volume'].corr(window_data['close'].pct_change())
            rolling_corrs.append(corr)
        else:
            rolling_corrs.append(np.nan)
    
    df['vol_ret_corr'] = pd.Series(rolling_corrs, index=df.index)
    df['corr_decay'] = df['vol_ret_corr'].diff() / df['vol_ret_corr'].shift(1)
    
    # Combine all components with appropriate weights
    factor = (
        0.3 * df['vol_regime'].fillna(0) +
        0.2 * df['range_curvature'].fillna(0) +
        0.25 * df['reversal_intensity'].fillna(0) +
        0.15 * df['vp_divergence'].fillna(0) +
        0.1 * df['trade_timing'].fillna(0)
    )
    
    # Remove any potential lookahead bias and return
    factor = factor.replace([np.inf, -np.inf], np.nan).fillna(0)
    return factor
