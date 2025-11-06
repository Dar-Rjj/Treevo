import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Generate alpha factor based on intraday volatility structure analysis and volume-price elasticity.
    Combines volatility regime detection with volume response to price changes.
    """
    # Make a copy to avoid modifying original dataframe
    data = df.copy()
    
    # Calculate daily returns
    data['returns'] = data['close'].pct_change()
    
    # Calculate intraday price ranges as volatility proxies
    data['morning_range'] = (data['high'] - data['low']) / data['open']  # Full day range as proxy
    data['intraday_vol'] = (data['high'] - data['low']) / data['close'].shift(1)
    
    # Volume-price elasticity calculation
    data['price_change'] = data['close'].pct_change()
    data['volume_change'] = data['volume'].pct_change()
    
    # Avoid division by zero
    mask = (data['price_change'].abs() > 1e-8) & (data['volume_change'].abs() > 1e-8)
    data['volume_elasticity'] = 0.0
    data.loc[mask, 'volume_elasticity'] = (
        data.loc[mask, 'volume_change'] / data.loc[mask, 'price_change']
    )
    
    # Rolling volatility regime detection (5-day window)
    data['vol_regime'] = data['intraday_vol'].rolling(window=5).mean()
    data['vol_regime_std'] = data['intraday_vol'].rolling(window=5).std()
    
    # Volume distribution entropy proxy
    # Use rolling volume statistics to measure concentration
    data['volume_ma_5'] = data['volume'].rolling(window=5).mean()
    data['volume_std_5'] = data['volume'].rolling(window=5).std()
    data['volume_entropy'] = data['volume_std_5'] / (data['volume_ma_5'] + 1e-8)
    
    # Price-level memory effect
    # Calculate how current price relates to recent volume-weighted average price
    data['vwap_5'] = (
        (data['close'] * data['volume']).rolling(window=5).sum() / 
        data['volume'].rolling(window=5).sum()
    )
    data['price_memory'] = (data['close'] - data['vwap_5']) / data['vwap_5']
    
    # Order flow imbalance proxy using OHLC relationship
    data['typical_price'] = (data['high'] + data['low'] + data['close']) / 3
    data['price_pressure'] = (data['close'] - data['open']) / (data['high'] - data['low'] + 1e-8)
    
    # Combine components into final alpha factor
    # Higher weight to periods with clear volatility regimes and responsive volume
    data['alpha_factor'] = (
        data['volume_elasticity'].rolling(window=3).mean() * 
        np.sign(data['price_memory']) * 
        data['vol_regime'] * 
        (1 - data['volume_entropy'])
    )
    
    # Normalize the factor using rolling z-score (20-day window)
    alpha_series = data['alpha_factor'].rolling(window=20).apply(
        lambda x: (x.iloc[-1] - x.mean()) / (x.std() + 1e-8), raw=False
    )
    
    return alpha_series
