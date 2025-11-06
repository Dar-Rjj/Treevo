import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(data):
    """
    Generate a novel alpha factor combining multiple market microstructure insights
    """
    df = data.copy()
    
    # Dynamic Volatility-Adjusted Momentum
    # Calculate 10-day momentum
    momentum = df['close'].pct_change(periods=10)
    
    # Calculate rolling volatility using high-low range (20-day window)
    daily_range = (df['high'] - df['low']) / df['close']
    volatility = daily_range.rolling(window=20).std()
    volatility = volatility.replace(0, np.nan).fillna(method='ffill')
    
    # Scale momentum by volatility
    vol_adjusted_momentum = momentum / (volatility + 1e-8)
    
    # Volume-Synchronized Price Probability
    # Calculate intraday price range normalized by close
    price_range = (df['high'] - df['low']) / df['close']
    
    # Calculate volume percentiles (20-day rolling)
    volume_percentile = df['volume'].rolling(window=20).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )
    
    # Weight price range by volume distribution
    volume_sync_price = price_range * volume_percentile
    
    # Liquidity-Adjusted Return Reversal
    # Calculate 5-day return
    short_return = df['close'].pct_change(periods=5)
    
    # Calculate liquidity measure (volume-to-amount ratio)
    liquidity = df['volume'] / (df['amount'] + 1e-8)
    liquidity_score = liquidity.rolling(window=10).rank(pct=True)
    
    # Apply liquidity filter to return (reversal signal)
    liquidity_adjusted_return = short_return * liquidity_score
    
    # Regime-Dependent Trend Strength
    # Identify volatility regime (20-day rolling std of daily range)
    vol_regime = daily_range.rolling(window=20).std()
    vol_regime_class = (vol_regime > vol_regime.rolling(window=60).median()).astype(int)
    
    # Identify volume regime (20-day rolling median)
    volume_regime = df['volume'].rolling(window=20).median()
    volume_regime_class = (df['volume'] > volume_regime).astype(int)
    
    # Calculate price trend (slope of 20-day MA over 5 days)
    ma_20 = df['close'].rolling(window=20).mean()
    trend_slope = ma_20.diff(5) / ma_20
    
    # Apply regime weights
    regime_trend = trend_slope * (1 + 0.5 * vol_regime_class + 0.3 * volume_regime_class)
    
    # Pressure-Based Breakout Detection
    # Calculate buying/selling pressure (close relative to open)
    pressure = (df['close'] - df['open']) / df['close']
    
    # Calculate pressure-volume correlation (10-day rolling)
    pressure_volume_corr = df['volume'].rolling(window=10).corr(pressure.abs())
    
    # Detect breakout signals using pressure thresholds
    pressure_threshold = pressure.rolling(window=60).quantile(0.8)
    breakout_distance = (pressure.abs() - pressure_threshold) / pressure_threshold
    
    # Scale by volume confirmation strength
    breakout_factor = breakout_distance * pressure_volume_corr
    
    # Combine all factors with equal weights
    factor = (
        0.2 * vol_adjusted_momentum +
        0.2 * volume_sync_price +
        0.2 * -liquidity_adjusted_return +  # Negative for reversal
        0.2 * regime_trend +
        0.2 * breakout_factor
    )
    
    # Normalize the final factor
    factor = (factor - factor.rolling(window=60).mean()) / factor.rolling(window=60).std()
    
    return factor
