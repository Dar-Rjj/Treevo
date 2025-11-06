import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import linregress

def heuristics_v2(df):
    """
    Generate novel alpha factors using volatility asymmetry, volume concentration,
    microstructure momentum, and intraday patterns.
    """
    # Initialize result series
    alpha = pd.Series(index=df.index, dtype=float)
    
    # Volatility Asymmetry Factor
    upside_returns = np.where(df['high'] > df['open'], (df['high'] - df['open']) / df['open'], 0)
    downside_returns = np.where(df['low'] < df['open'], (df['low'] - df['open']) / df['open'], 0)
    
    upside_vol = pd.Series(upside_returns, index=df.index).rolling(window=20, min_periods=10).std()
    downside_vol = pd.Series(downside_returns, index=df.index).rolling(window=20, min_periods=10).std()
    volatility_asymmetry = upside_vol / downside_vol
    
    # Volume Concentration Alpha
    def gini_coefficient(x):
        """Calculate Gini coefficient for volume concentration"""
        if len(x) < 2:
            return np.nan
        sorted_vol = np.sort(x)
        n = len(sorted_vol)
        index = np.arange(1, n + 1)
        return (np.sum((2 * index - n - 1) * sorted_vol)) / (n * np.sum(sorted_vol))
    
    volume_gini = df['volume'].rolling(window=10, min_periods=5).apply(gini_coefficient, raw=True)
    
    # Volume autocorrelation decay rate (5-day lag)
    volume_autocorr = df['volume'].rolling(window=20, min_periods=10).apply(
        lambda x: x.autocorr(lag=5) if len(x) >= 15 else np.nan
    )
    
    # High-Low range for liquidity shock detection
    high_low_range = (df['high'] - df['low']) / df['close']
    volume_concentration = volume_gini * (1 - volume_autocorr) * high_low_range
    
    # Microstructure Momentum
    # Effective spread estimation using High-Low range normalized by volume
    effective_spread = (df['high'] - df['low']) / df['close'] / (df['volume'] + 1e-8)
    
    # Large trade concentration (Amount/Volume)
    large_trade_concentration = df['amount'] / (df['volume'] * df['close'] + 1e-8)
    
    # Raw momentum (5-day return)
    raw_momentum = df['close'].pct_change(periods=5)
    
    # Order flow quality weighted momentum
    order_flow_quality = 1 / (effective_spread * large_trade_concentration + 1e-8)
    microstructure_momentum = raw_momentum * order_flow_quality
    
    # Intraday Momentum Acceleration
    # Opening momentum (first observation vs previous close)
    opening_momentum = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
    
    # Closing momentum pressure (last observation pattern)
    # Using current day's high-low range as proxy for intraday pressure
    closing_pressure = (df['close'] - df['open']) / (df['high'] - df['low'] + 1e-8)
    
    # Momentum persistence (autocorrelation of 1-day returns)
    momentum_persistence = df['close'].pct_change().rolling(window=10, min_periods=5).apply(
        lambda x: x.autocorr(lag=1) if len(x) >= 6 else np.nan
    )
    
    intraday_momentum = opening_momentum * closing_pressure * momentum_persistence
    
    # Combine factors with appropriate weights
    alpha = (
        0.3 * volatility_asymmetry.fillna(0) +
        0.25 * volume_concentration.fillna(0) +
        0.25 * microstructure_momentum.fillna(0) +
        0.2 * intraday_momentum.fillna(0)
    )
    
    # Z-score normalization
    alpha = (alpha - alpha.rolling(window=60, min_periods=30).mean()) / alpha.rolling(window=60, min_periods=30).std()
    
    return alpha
