import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Price Efficiency Divergence
    # Short-term (5-day)
    data['price_eff_short'] = (data['close'] - data['close'].shift(5)) / (
        data['high'].rolling(6).max() - data['low'].rolling(6).min()
    )
    
    # Medium-term (13-day)
    data['price_eff_medium'] = (data['close'] - data['close'].shift(13)) / (
        data['high'].rolling(14).max() - data['low'].rolling(14).min()
    )
    
    # Long-term (34-day)
    data['price_eff_long'] = (data['close'] - data['close'].shift(34)) / (
        data['high'].rolling(35).max() - data['low'].rolling(35).min()
    )
    
    # Volume Efficiency Divergence
    # Short-term
    data['vol_eff_short'] = data['volume'] / data['volume'].shift(1).rolling(4).mean()
    
    # Medium-term
    data['vol_eff_medium'] = data['volume'] / data['volume'].shift(1).rolling(12).mean()
    
    # Long-term
    data['vol_eff_long'] = data['volume'] / data['volume'].shift(1).rolling(33).mean()
    
    # Intraday Structure
    # Opening Gap
    data['opening_gap'] = data['open'] / data['close'].shift(1) - 1
    
    # Range Utilization
    data['range_util'] = (data['close'] - data['open']) / (data['high'] - data['low'])
    
    # Volume Distribution
    # Volume Concentration
    data['vol_concentration'] = data['volume'] / data['volume'].rolling(6).max()
    
    # Volume-Volatility Ratio
    data['vol_volatility_ratio'] = data['volume'] / (data['high'] - data['low'])
    
    # Efficiency Divergence Score
    data['eff_div_score'] = (
        data['price_eff_short'] * data['vol_eff_short'] +
        data['price_eff_medium'] * data['vol_eff_medium'] +
        data['price_eff_long'] * data['vol_eff_long']
    ) / 3
    
    # Regime-Weighted Composite
    # Calculate volatility regime (using 20-day rolling std of returns)
    data['returns'] = data['close'].pct_change()
    data['volatility_regime'] = data['returns'].rolling(20).std()
    
    # Normalize volatility regime for weighting
    vol_norm = data['volatility_regime'].rolling(60).apply(
        lambda x: (x.iloc[-1] - x.mean()) / x.std() if x.std() > 0 else 0
    )
    
    # Calculate trend regime (using 20-day moving average)
    data['trend_regime'] = data['close'].rolling(20).mean() / data['close'] - 1
    
    # Final composite factor with regime weighting
    regime_weight = 1 / (1 + np.exp(-vol_norm * data['trend_regime']))
    
    data['factor'] = (
        regime_weight * data['eff_div_score'] +
        (1 - regime_weight) * (
            data['opening_gap'] + 
            data['range_util'] + 
            data['vol_concentration'] + 
            data['vol_volatility_ratio']
        ) / 4
    )
    
    return data['factor']
