import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    """
    Novel alpha factor combining momentum acceleration, volume persistence, 
    and volatility regime adaptation with microstructure insights.
    """
    
    # Momentum acceleration: 3-day vs 6-day momentum difference
    mom_3d = (df['close'].shift(1) - df['close'].shift(3)) / df['close'].shift(3)
    mom_6d = (df['close'].shift(1) - df['close'].shift(6)) / df['close'].shift(6)
    momentum_accel = mom_3d - mom_6d
    
    # Volume persistence: current volume vs recent trend
    vol_ma_5 = df['volume'].rolling(window=5, min_periods=1).mean()
    vol_ma_10 = df['volume'].rolling(window=10, min_periods=1).mean()
    volume_persistence = (vol_ma_5 / vol_ma_10) * np.sign(df['volume'] - vol_ma_5)
    
    # Volatility regime adaptation using rolling percentiles
    intraday_range = (df['high'] - df['low']) / df['close']
    vol_regime = intraday_range.rolling(window=20, min_periods=1).apply(
        lambda x: (x.iloc[-1] - np.percentile(x, 30)) / (np.percentile(x, 70) - np.percentile(x, 30) + 1e-7)
    )
    
    # Trade size dynamics: large trade concentration
    trade_size = df['amount'] / (df['volume'] + 1e-7)
    large_trade_ratio = trade_size.rolling(window=10, min_periods=1).apply(
        lambda x: np.sum(x > np.percentile(x, 75)) / len(x)
    )
    
    # Price efficiency: close relative to intraday range
    price_efficiency = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-7)
    
    # Combine factors with economic rationale:
    # Accelerating momentum amplified by persistent volume,
    # adapted to volatility regime, with large trade confirmation
    # and price efficiency adjustment
    alpha_factor = (
        momentum_accel * 
        np.tanh(volume_persistence) * 
        (2 - np.exp(vol_regime)) * 
        np.sqrt(large_trade_ratio + 0.1) * 
        (2 * price_efficiency - 1)
    )
    
    return alpha_factor
