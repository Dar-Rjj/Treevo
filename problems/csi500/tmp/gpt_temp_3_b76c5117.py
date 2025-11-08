import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying the original dataframe
    data = df.copy()
    
    # Calculate returns
    data['returns'] = data['close'].pct_change()
    
    # Price Momentum Gap
    price_momentum_3 = (data['close'] - data['close'].shift(3)) / data['close'].shift(3)
    price_momentum_8 = (data['close'] - data['close'].shift(8)) / data['close'].shift(8)
    price_momentum_gap = price_momentum_3 - price_momentum_8
    
    # Volume Momentum Gap
    volume_momentum_3 = (data['volume'] - data['volume'].shift(3)) / data['volume'].shift(3)
    volume_momentum_8 = (data['volume'] - data['volume'].shift(8)) / data['volume'].shift(8)
    volume_momentum_gap = volume_momentum_3 - volume_momentum_8
    
    # Calculate volatility asymmetry
    def calculate_volatility_skew(returns_series):
        positive_returns = returns_series[returns_series > 0]
        negative_returns = returns_series[returns_series < 0]
        
        pos_vol = positive_returns.std() if len(positive_returns) > 1 else 0
        neg_vol = negative_returns.std() if len(negative_returns) > 1 else 0
        
        return pos_vol - neg_vol
    
    # Calculate rolling volatility skew over 7 days
    volatility_skew = data['returns'].rolling(window=7, min_periods=2).apply(
        calculate_volatility_skew, raw=False
    )
    
    # Price-Volume Divergence
    price_volume_divergence = price_momentum_gap - volume_momentum_gap
    
    # Final alpha factor construction
    alpha_factor = price_volume_divergence * (1 + volatility_skew * np.sign(price_volume_divergence))
    
    return alpha_factor
