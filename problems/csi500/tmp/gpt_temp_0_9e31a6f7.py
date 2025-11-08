import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(data):
    """
    Volatility-Normalized Momentum with Bounded Volume Regimes factor
    
    Parameters:
    data: pandas DataFrame with columns ['open', 'high', 'low', 'close', 'amount', 'volume']
    
    Returns:
    pandas Series with factor values
    """
    df = data.copy()
    
    # Multi-Period Return Calculation
    df['short_return'] = df['close'] / df['close'].shift(3) - 1
    df['medium_return'] = df['close'] / df['close'].shift(8) - 1
    df['long_return'] = df['close'] / df['close'].shift(20) - 1
    
    # Volatility Adjustment - Realized Volatility Estimation
    df['daily_range'] = (df['high'] - df['low']) / df['close'].shift(1)
    
    # Rolling Volatility (10-day std of daily ranges)
    df['short_vol'] = df['daily_range'].rolling(window=10, min_periods=8).std()
    df['medium_vol'] = df['daily_range'].rolling(window=10, min_periods=8).std()
    df['long_vol'] = df['daily_range'].rolling(window=10, min_periods=8).std()
    
    # Return Normalization
    df['short_mom'] = df['short_return'] / df['short_vol']
    df['medium_mom'] = df['medium_return'] / df['medium_vol']
    df['long_mom'] = df['long_return'] / df['long_vol']
    
    # Bounded Volume Regime Signals
    df['volume_ratio'] = df['volume'] / df['volume'].shift(5)
    df['volume_momentum'] = np.cbrt(df['volume_ratio'])
    
    # Percentile-Based Regime Classification
    df['volume_percentile'] = df['volume_momentum'].rolling(window=20, min_periods=15).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )
    
    # Regime Assignment
    df['volume_regime'] = 1.0  # Default medium regime
    df.loc[df['volume_percentile'] > 0.8, 'volume_regime'] = 1.5  # High regime
    df.loc[df['volume_percentile'] < 0.2, 'volume_regime'] = 0.5  # Low regime
    
    # Adaptive Factor Combination - Multiplicative Momentum Integration
    df['combined_momentum'] = np.cbrt(df['short_mom'] * df['medium_mom'] * df['long_mom'])
    
    # Volume-Regime Weighted Final Factor
    df['factor'] = df['combined_momentum'] * df['volume_regime']
    
    return df['factor']
