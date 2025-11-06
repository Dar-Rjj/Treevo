import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Volume-Efficiency Momentum Divergence factor that captures the relationship between 
    price momentum and volume efficiency across multiple time horizons.
    """
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Volume Efficiency Calculation
    # Price movement per unit volume across different time horizons
    data['dollar_volume'] = data['volume'] * data['close']
    
    # 5-day volume efficiency (price change per unit volume)
    data['price_change_5d'] = data['close'] / data['close'].shift(5) - 1
    data['volume_eff_5d'] = data['price_change_5d'] / (data['dollar_volume'].rolling(5).mean() + 1e-8)
    
    # 10-day volume efficiency
    data['price_change_10d'] = data['close'] / data['close'].shift(10) - 1
    data['volume_eff_10d'] = data['price_change_10d'] / (data['dollar_volume'].rolling(10).mean() + 1e-8)
    
    # 20-day volume efficiency
    data['price_change_20d'] = data['close'] / data['close'].shift(20) - 1
    data['volume_eff_20d'] = data['price_change_20d'] / (data['dollar_volume'].rolling(20).mean() + 1e-8)
    
    # Comparison of volume efficiency in up vs down days
    data['up_day'] = (data['close'] > data['open']).astype(int)
    data['down_day'] = (data['close'] < data['open']).astype(int)
    
    # Rolling efficiency comparison (up days efficiency minus down days efficiency)
    data['up_eff_5d'] = data['volume_eff_5d'].where(data['up_day'] == 1).rolling(10).mean()
    data['down_eff_5d'] = data['volume_eff_5d'].where(data['down_day'] == 1).rolling(10).mean()
    data['eff_spread_5d'] = data['up_eff_5d'] - data['down_eff_5d']
    
    # Momentum Divergence Detection
    # Price momentum vs volume efficiency momentum comparison
    data['price_momentum_10d'] = data['close'] / data['close'].shift(10) - 1
    data['eff_momentum_10d'] = data['volume_eff_10d'] / (data['volume_eff_10d'].shift(5) + 1e-8) - 1
    
    # Efficiency breakdown during price acceleration
    data['price_accel_5d'] = data['price_momentum_10d'] - data['price_momentum_10d'].shift(5)
    data['eff_breakdown'] = data['eff_momentum_10d'] - data['price_accel_5d']
    
    # Cross-sectional efficiency ranking persistence
    data['eff_rank_5d'] = data['volume_eff_5d'].rolling(20).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) == 20 else np.nan, 
        raw=False
    )
    data['eff_rank_persistence'] = data['eff_rank_5d'].rolling(10).std()
    
    # Signal Construction
    # Efficiency-adjusted momentum factors
    data['eff_adj_momentum'] = data['price_momentum_10d'] * (1 + data['volume_eff_10d'])
    
    # Efficiency regime transition timing
    data['eff_regime_change'] = (
        data['volume_eff_10d'].rolling(5).std() / 
        (data['volume_eff_10d'].rolling(20).std() + 1e-8)
    )
    
    # Divergence magnitude as return predictor
    data['momentum_divergence'] = (
        data['price_momentum_10d'] - 
        data['eff_momentum_10d'].rolling(5).mean()
    )
    
    # Combine signals into final factor
    factor = (
        0.4 * data['eff_adj_momentum'] +
        0.3 * data['eff_spread_5d'] +
        0.2 * data['momentum_divergence'] -
        0.1 * data['eff_rank_persistence']
    )
    
    # Normalize the factor
    factor = (factor - factor.rolling(60).mean()) / (factor.rolling(60).std() + 1e-8)
    
    return factor
