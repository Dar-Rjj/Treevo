import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Multi-Timeframe Momentum Acceleration
    # 5-day momentum
    mom_5d = (data['close'] / data['close'].shift(5)) - 1
    
    # 20-day momentum
    mom_20d = (data['close'] / data['close'].shift(20)) - 1
    
    # Momentum acceleration
    mom_accel = mom_5d - mom_20d
    
    # Volume Regime Classification
    # Volume momentum
    vol_mom = (data['volume'] / data['volume'].shift(20)) - 1
    
    # Volume regime (2x for high volume, 1x for low volume)
    vol_regime_weight = np.where(vol_mom > 0, 2.0, 1.0)
    
    # Volume confirmation strength
    vol_ma_20 = data['volume'].rolling(window=20, min_periods=1).mean()
    vol_ratio = data['volume'] / vol_ma_20
    
    # Intraday vs Close-to-Close Return Divergence
    # Close-to-close return
    c2c_return = (data['close'] / data['close'].shift(1)) - 1
    
    # Intraday return
    intraday_return = (data['close'] / data['open']) - 1
    
    # Return divergence
    return_divergence = c2c_return - intraday_return
    
    # Range Efficiency with Volume Clustering
    # True range
    tr1 = data['high'] - data['low']
    tr2 = abs(data['high'] - data['close'].shift(1))
    tr3 = abs(data['low'] - data['close'].shift(1))
    true_range = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Price movement efficiency
    price_change = abs(data['close'] - data['close'].shift(1))
    efficiency_ratio = price_change / true_range
    
    # Volume-clustered efficiency
    vol_percentile = data['volume'].rolling(window=20, min_periods=1).apply(
        lambda x: (x.iloc[-1] > np.percentile(x, 80)) if len(x) == 20 else False
    )
    high_vol_efficiency = (vol_percentile > 0) & (efficiency_ratio > 0.7)
    efficiency_multiplier = np.where(high_vol_efficiency, 1.5, 1.0)
    
    # Generate Composite Alpha Signal
    # Core momentum acceleration component
    core_momentum = mom_accel * vol_ratio * vol_regime_weight
    
    # Apply return divergence adjustment
    divergence_adjusted = core_momentum * (1 + return_divergence)
    
    # Apply volume-clustered efficiency multiplier
    final_alpha = divergence_adjusted * efficiency_multiplier
    
    return pd.Series(final_alpha, index=data.index)
