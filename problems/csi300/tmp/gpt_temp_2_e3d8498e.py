import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Cross-Asset Volatility Transmission Alpha Factor
    Combines intraday volatility patterns, volume-volatility dynamics, and volatility persistence
    to generate predictive signals for future returns
    """
    # Calculate basic volatility measures
    df['daily_range'] = (df['high'] - df['low']) / df['close'].shift(1)
    df['close_to_close_vol'] = df['close'].pct_change().abs()
    
    # Intraday volatility patterns
    df['open_vol'] = (df['open'] - df['close'].shift(1)).abs() / df['close'].shift(1)
    df['close_vol'] = (df['close'] - df['open']).abs() / df['open']
    df['intraday_vol_ratio'] = df['close_vol'] / (df['open_vol'] + 1e-8)
    
    # Volume-volatility dynamics
    df['volume_ma'] = df['volume'].rolling(window=20, min_periods=10).mean()
    df['abnormal_volume'] = df['volume'] / (df['volume_ma'] + 1e-8)
    df['volatility_volume_correlation'] = df['close_to_close_vol'].rolling(window=10).corr(df['volume'])
    
    # Volatility persistence and clustering
    df['volatility_ma_5'] = df['close_to_close_vol'].rolling(window=5, min_periods=3).mean()
    df['volatility_ma_20'] = df['close_to_close_vol'].rolling(window=20, min_periods=10).mean()
    df['volatility_regime'] = df['volatility_ma_5'] / (df['volatility_ma_20'] + 1e-8)
    
    # Large trade impact (using amount as proxy)
    df['amount_ma'] = df['amount'].rolling(window=20, min_periods=10).mean()
    df['large_trade_impact'] = df['amount'] / (df['amount_ma'] + 1e-8) * df['daily_range']
    
    # Volatility momentum and mean reversion signals
    df['volatility_momentum'] = df['close_to_close_vol'] / (df['close_to_close_vol'].shift(5) + 1e-8)
    df['volatility_zscore'] = (df['close_to_close_vol'] - df['volatility_ma_20']) / (df['close_to_close_vol'].rolling(window=20).std() + 1e-8)
    
    # Cross-sectional volatility transmission proxy (using own volatility structure)
    df['volatility_autocorr'] = df['close_to_close_vol'].rolling(window=10).apply(lambda x: x.autocorr(), raw=False)
    
    # Combine signals into final alpha factor
    # High volatility regime + abnormal volume + positive volatility autocorrelation
    factor = (
        df['volatility_regime'] *  # Volatility regime strength
        df['abnormal_volume'] *    # Volume-volatility interaction
        df['intraday_vol_ratio'] * # Intraday pattern
        df['volatility_autocorr'] * # Persistence signal
        np.sign(df['volatility_zscore']) * np.exp(-abs(df['volatility_zscore']))  # Mean reversion timing
    )
    
    # Clean up and return
    factor = factor.replace([np.inf, -np.inf], np.nan)
    return factor
