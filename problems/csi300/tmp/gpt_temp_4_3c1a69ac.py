import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Multi-Scale Fractal Momentum with Liquidity Gradient Dynamics
    Combines fractal pattern recognition, liquidity analysis, and microstructure dynamics
    """
    # Copy dataframe to avoid modifying original
    data = df.copy()
    
    # Fractal Price Pattern Recognition
    # Multi-timeframe fractal dimension analysis (5,10,20 days)
    def fractal_dimension(high, low, window):
        """Calculate fractal dimension using Hurst exponent approximation"""
        returns = np.log(high / low).replace([np.inf, -np.inf], np.nan)
        lags = range(2, min(window, len(returns)))
        tau = [np.std(np.subtract(returns[lag:], returns[:-lag])) for lag in lags]
        poly = np.polyfit(np.log(lags), np.log(tau), 1)
        return poly[0]
    
    # Calculate fractal dimensions for different timeframes
    for window in [5, 10, 20]:
        data[f'fractal_dim_{window}'] = data['high'].rolling(window=window).apply(
            lambda x: fractal_dimension(x, data['low'].loc[x.index], window), raw=False
        )
    
    # Fractal momentum divergence across scales
    data['fractal_momentum_div'] = (
        data['fractal_dim_5'] - data['fractal_dim_20']
    ) / data['fractal_dim_10'].replace(0, np.nan)
    
    # Liquidity Gradient Field Analysis
    # Volume gradient computation across price levels
    data['price_range'] = (data['high'] - data['low']) / data['close']
    data['volume_intensity'] = data['volume'] / data['volume'].rolling(20).mean()
    data['liquidity_gradient'] = (
        data['volume_intensity'] * data['price_range']
    ).rolling(5).mean()
    
    # Cross-timeframe liquidity momentum divergence
    data['liquidity_momentum_short'] = data['volume'].pct_change(5)
    data['liquidity_momentum_long'] = data['volume'].pct_change(20)
    data['liquidity_momentum_div'] = (
        data['liquidity_momentum_short'] - data['liquidity_momentum_long']
    )
    
    # Microstructure Imbalance Dynamics
    # Price impact asymmetry analysis
    data['intraday_return'] = (data['close'] - data['open']) / data['open']
    data['volume_clustering'] = (
        data['volume'].rolling(10).std() / data['volume'].rolling(10).mean()
    )
    
    # Order flow persistence
    data['amount_per_trade'] = data['amount'] / data['volume'].replace(0, np.nan)
    data['order_flow_persistence'] = (
        data['amount_per_trade'].pct_change(3).rolling(5).std()
    )
    
    # Microstructure pressure build-up
    data['micro_pressure'] = (
        data['volume_clustering'] * data['order_flow_persistence'] * 
        np.abs(data['intraday_return'])
    )
    
    # Adaptive Signal Integration Framework
    # Fractal-liquidity regime classification
    data['fractal_regime'] = np.where(
        data['fractal_dim_10'] > data['fractal_dim_10'].rolling(20).mean(), 
        1, -1
    )
    data['liquidity_regime'] = np.where(
        data['liquidity_gradient'] > data['liquidity_gradient'].rolling(20).mean(),
        1, -1
    )
    
    # Multi-scale signal validation and weighting
    fractal_weight = 0.4
    liquidity_weight = 0.35
    micro_weight = 0.25
    
    # Dynamic signal aggregation with regime adaptation
    regime_multiplier = data['fractal_regime'] * data['liquidity_regime']
    
    # Final factor calculation
    factor = (
        fractal_weight * data['fractal_momentum_div'] +
        liquidity_weight * data['liquidity_momentum_div'] +
        micro_weight * data['micro_pressure']
    ) * regime_multiplier
    
    # Clean and return the factor series
    factor = factor.replace([np.inf, -np.inf], np.nan).fillna(method='ffill')
    
    return factor
