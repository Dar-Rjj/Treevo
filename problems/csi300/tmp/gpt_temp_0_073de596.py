import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Dynamic Volatility-Adjusted Momentum
    momentum_5d = df['close'] / df['close'].shift(5) - 1
    vol_20d = (df['high'] - df['low']).rolling(window=20).std()
    volume_sma_20d = df['volume'].rolling(window=20).mean()
    factor1 = momentum_5d / vol_20d * (df['volume'] / volume_sma_20d)
    
    # Price-Volume Divergence Oscillator
    price_roc_10d = df['close'] / df['close'].shift(10) - 1
    volume_roc_10d = df['volume'] / df['volume'].shift(10) - 1
    factor2 = np.sign(price_roc_10d - volume_roc_10d)
    
    # Liquidity-Adjusted Reversal Factor
    ret_3d = df['close'] / df['close'].shift(3) - 1
    dollar_volume = df['close'] * df['volume']
    extreme_ret = ret_3d.quantile(0.9)  # top decile threshold
    reversal_signal = -np.sign(ret_3d) * (np.abs(ret_3d) >= extreme_ret) * dollar_volume
    factor3 = reversal_signal
    
    # Intraday Strength Persistence
    intraday_strength = (df['close'] - df['open']) / (df['high'] - df['low'])
    strength_sign = np.sign(intraday_strength)
    consecutive_days = strength_sign.groupby((strength_sign != strength_sign.shift(1)).cumsum()).cumcount() + 1
    factor4 = intraday_strength * consecutive_days * df['volume']
    
    # Volatility Regime Breakout
    vol_20d_close = df['close'].rolling(window=20).std()
    vol_percentile = vol_20d_close.rolling(window=50).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)
    regime_break = (vol_percentile > 0.8) | (vol_percentile < 0.2)
    factor5 = (df['close'] / df['close'].shift(1) - 1) * regime_break
    
    # Combine factors with equal weights
    combined_factor = (factor1.fillna(0) + factor2.fillna(0) + factor3.fillna(0) + 
                       factor4.fillna(0) + factor5.fillna(0))
    
    return combined_factor
