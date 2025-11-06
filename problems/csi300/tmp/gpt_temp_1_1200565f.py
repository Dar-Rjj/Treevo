import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df, n=20, vol_threshold=1.2):
    """
    Generate multiple alpha factors using OHLCV data with dynamic volatility adjustment,
    volume analysis, and price momentum techniques.
    
    Parameters:
    df: DataFrame with columns ['open', 'high', 'low', 'close', 'amount', 'volume']
    n: rolling window period for calculations
    vol_threshold: volatility threshold for regime detection
    
    Returns:
    pandas Series of combined alpha factor values
    """
    
    # Initialize result series
    result = pd.Series(index=df.index, dtype=float)
    
    # Factor 1: Dynamic Volatility-Adjusted Price Momentum
    price_momentum = df['close'] - df['close'].shift(n)
    high_low_range = df['high'] - df['low']
    rolling_vol = high_low_range.rolling(window=n, min_periods=int(n*0.8)).std()
    factor1 = price_momentum / (rolling_vol + 1e-8)
    
    # Factor 2: Volume-Implied Price Pressure
    volume_mean = df['volume'].rolling(window=n, min_periods=int(n*0.8)).mean()
    abnormal_volume = df['volume'] / (volume_mean + 1e-8)
    intraday_return = df['close'] / df['open'] - 1
    factor2 = abnormal_volume * intraday_return
    
    # Factor 3: Bid-Ask Spread Proxy Momentum
    spread_proxy = df['high'] / df['low']
    short_term_trend = df['close'] / df['close'].shift(1) - 1
    factor3 = spread_proxy * short_term_trend
    
    # Factor 4: Liquidity-Adjusted Reversal
    price_reversal = df['close'] / df['close'].shift(1) - 1
    volume_to_amount = df['volume'] / (df['amount'] + 1e-8)
    factor4 = volume_to_amount * price_reversal
    
    # Factor 5: Acceleration-Deceleration Indicator
    velocity = df['close'] / df['close'].shift(1) - 1
    acceleration = velocity - velocity.shift(1)
    factor5 = acceleration
    
    # Factor 6: Volume-Weighted Price Acceleration
    price_acceleration = (df['close']/df['close'].shift(1)) - (df['close'].shift(1)/df['close'].shift(2))
    volume_momentum = df['volume'] / df['volume'].shift(1) - 1
    factor6 = price_acceleration * volume_momentum
    
    # Factor 7: Range-Based Volatility Regime Factor
    normalized_range = (df['high'] - df['low']) / (df['close'] + 1e-8)
    range_percentile = normalized_range.rolling(window=n, min_periods=int(n*0.8)).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) >= int(n*0.8) else np.nan
    )
    regime_indicator = (normalized_range > range_percentile * vol_threshold).astype(float)
    factor7 = regime_indicator
    
    # Factor 8: Amount Efficiency Momentum
    price_per_volume = df['amount'] / (df['volume'] + 1e-8)
    efficiency_momentum = price_per_volume / price_per_volume.shift(1) - 1
    price_direction = np.sign(df['close'] / df['close'].shift(1) - 1)
    factor8 = efficiency_momentum * price_direction
    
    # Factor 9: Opening Gap Persistence Factor
    opening_gap = df['open'] / df['close'].shift(1) - 1
    high_low_diff = df['high'] - df['low']
    range_utilization = (df['close'] - df['low']) / (high_low_diff + 1e-8)
    persistence_score = opening_gap * range_utilization
    factor9 = persistence_score
    
    # Factor 10: Volatility-Clustered Mean Reversion
    rolling_high_low_vol = high_low_range.rolling(window=n, min_periods=int(n*0.8)).std()
    vol_indicator = (rolling_high_low_vol > rolling_high_low_vol.rolling(window=n*2, min_periods=int(n*1.6)).mean() * vol_threshold).astype(float)
    rolling_mean_close = df['close'].rolling(window=n, min_periods=int(n*0.8)).mean()
    deviation_from_mean = df['close'] / rolling_mean_close - 1
    factor10 = deviation_from_mean * vol_indicator
    
    # Factor 11: Volume-Volatility Confluence Factor
    volume_surprise = df['volume'] - df['volume'].rolling(window=n, min_periods=int(n*0.8)).mean()
    volatility_surprise = high_low_range - high_low_range.rolling(window=n, min_periods=int(n*0.8)).mean()
    factor11 = volume_surprise * volatility_surprise
    
    # Combine all factors with equal weighting
    factors = [factor1, factor2, factor3, factor4, factor5, factor6, 
               factor7, factor8, factor9, factor10, factor11]
    
    # Create combined factor (equal weighted average of normalized factors)
    combined_factor = pd.Series(0, index=df.index)
    valid_count = pd.Series(0, index=df.index)
    
    for factor in factors:
        # Normalize each factor by its rolling standard deviation
        factor_normalized = factor / (factor.rolling(window=n*2, min_periods=int(n*1.6)).std() + 1e-8)
        # Only add if not NaN
        mask = ~factor_normalized.isna()
        combined_factor[mask] += factor_normalized[mask]
        valid_count[mask] += 1
    
    # Average the valid factors
    result = combined_factor / (valid_count + 1e-8)
    
    return result
