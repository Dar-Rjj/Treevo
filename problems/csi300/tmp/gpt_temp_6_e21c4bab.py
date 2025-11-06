import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate daily returns
    data['daily_return'] = data['close'].pct_change()
    
    # Calculate sector return (using rolling mean of all stocks as proxy)
    # In practice, this would be replaced with actual sector data
    data['sector_return'] = data['daily_return'].rolling(window=5, min_periods=1).mean()
    
    # Calculate market return (using broader rolling mean as proxy)
    data['market_return'] = data['daily_return'].rolling(window=10, min_periods=1).mean()
    
    # Relative Performance Calculation
    data['rel_sector'] = (data['daily_return'] / data['sector_return'].replace(0, np.nan)) - 1
    data['rel_market'] = (data['daily_return'] / data['market_return'].replace(0, np.nan)) - 1
    
    # Momentum Persistence Analysis
    # Short-term sign consistency (3 days)
    data['sign_3d'] = np.sign(data['daily_return'].rolling(window=3, min_periods=1).sum())
    data['sign_consistency_3d'] = (data['sign_3d'].rolling(window=3, min_periods=1).apply(
        lambda x: np.mean(x == x.iloc[-1]) if len(x) > 0 else np.nan
    ))
    
    # Medium-term sign consistency (5 days)
    data['sign_5d'] = np.sign(data['daily_return'].rolling(window=5, min_periods=1).sum())
    data['sign_consistency_5d'] = (data['sign_5d'].rolling(window=5, min_periods=1).apply(
        lambda x: np.mean(x == x.iloc[-1]) if len(x) > 0 else np.nan
    ))
    
    # Volatility-Adjusted Momentum
    data['daily_volatility'] = data['high'] - data['low']
    data['momentum_3d'] = data['daily_return'].rolling(window=3, min_periods=1).sum()
    data['momentum_5d'] = data['daily_return'].rolling(window=5, min_periods=1).sum()
    
    # Momentum-to-volatility ratios
    data['momentum_vol_ratio_3d'] = data['momentum_3d'] / data['daily_volatility'].replace(0, np.nan)
    data['momentum_vol_ratio_5d'] = data['momentum_5d'] / data['daily_volatility'].replace(0, np.nan)
    
    # Combine factors with weights
    data['cross_asset_factor'] = (
        0.3 * data['rel_sector'].fillna(0) +
        0.3 * data['rel_market'].fillna(0) +
        0.2 * data['sign_consistency_3d'].fillna(0) +
        0.1 * data['sign_consistency_5d'].fillna(0) +
        0.05 * data['momentum_vol_ratio_3d'].fillna(0) +
        0.05 * data['momentum_vol_ratio_5d'].fillna(0)
    )
    
    return data['cross_asset_factor']
