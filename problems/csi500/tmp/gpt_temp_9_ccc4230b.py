import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    # Non-linear momentum with volatility-adaptive volume confirmation
    # Combines asymmetric price behavior, volume persistence, and dynamic volatility scaling
    
    # Asymmetric momentum: stronger weight on upward moves with volume confirmation
    price_change = df['close'] / df['close'].shift(1) - 1
    volume_change = df['volume'] / df['volume'].shift(1) - 1
    
    # Non-linear momentum with volume acceleration
    momentum_component = np.sign(price_change) * np.sqrt(np.abs(price_change)) * np.tanh(volume_change * 10)
    
    # Volatility-adaptive volume persistence
    intraday_range = (df['high'] - df['low']) / df['close']
    volume_persistence = df['volume'].rolling(window=5).apply(
        lambda x: np.corrcoef(range(len(x)), x)[0,1] if len(x) > 1 else 0
    )
    
    # Dynamic volatility scaling using rolling percentiles
    volatility_scale = intraday_range.rolling(window=20).apply(
        lambda x: (x.iloc[-1] - np.percentile(x, 25)) / (np.percentile(x, 75) - np.percentile(x, 25) + 1e-7)
    )
    
    # Amount concentration factor (non-linear transformation)
    trade_concentration = df['amount'] / (df['volume'] + 1e-7)
    concentration_power = np.log1p(trade_concentration) / np.log1p(trade_concentration.rolling(window=10).std() + 1)
    
    # Non-linear combination with volatility-adaptive scaling
    alpha_factor = (
        momentum_component * 
        (1 + np.arctan(volume_persistence * 2)) * 
        np.exp(-volatility_scale * 3) * 
        concentration_power
    )
    
    return alpha_factor
