import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import linregress

def heuristics_v2(data):
    """
    Generate alpha factor using multiple technical heuristics
    """
    df = data.copy()
    
    # Initialize result series
    alpha = pd.Series(index=df.index, dtype=float)
    
    # 1. Momentum Decay with Volume Confirmation
    momentum_signal = _momentum_decay_volume_confirmation(df)
    
    # 2. Volatility Regime Adjusted Return
    volatility_adjusted = _volatility_regime_adjusted_return(df)
    
    # 3. Price-Level Relative Strength
    price_relative = _price_level_relative_strength(df)
    
    # 4. Volume-Price Divergence Detector
    divergence_signal = _volume_price_divergence(df)
    
    # 5. Overnight Gap Persistence Factor
    gap_persistence = _overnight_gap_persistence(df)
    
    # 6. Liquidity-Adjusted Momentum Reversal
    liquidity_reversal = _liquidity_adjusted_momentum_reversal(df)
    
    # 7. Intraday Strength Consistency
    intraday_strength = _intraday_strength_consistency(df)
    
    # 8. Sector-Relative Breakout Detection
    sector_breakout = _sector_relative_breakout(df)
    
    # Combine all factors with equal weights
    factors = [
        momentum_signal,
        volatility_adjusted,
        price_relative,
        divergence_signal,
        gap_persistence,
        liquidity_reversal,
        intraday_strength,
        sector_breakout
    ]
    
    # Standardize and combine
    for i, factor in enumerate(factors):
        if factor is not None and len(factor) > 0:
            factors[i] = (factor - factor.mean()) / factor.std()
    
    # Equal-weighted combination
    alpha = sum(factors) / len([f for f in factors if f is not None and len(f) > 0])
    
    return alpha

def _momentum_decay_volume_confirmation(df):
    """Momentum Decay with Volume Confirmation"""
    close = df['close']
    volume = df['volume']
    
    # Calculate momentum with exponential decay
    momentum = pd.Series(index=df.index, dtype=float)
    for i in range(4, len(df)):
        weights = np.exp(-np.arange(5) / 2.5)  # Exponential decay weights
        weights = weights / weights.sum()
        momentum.iloc[i] = (close.iloc[i-4:i+1] * weights).sum()
    
    # Calculate volume trend slope
    volume_slope = pd.Series(index=df.index, dtype=float)
    for i in range(4, len(df)):
        x = np.arange(5)
        y = volume.iloc[i-4:i+1].values
        slope, _, _, _, _ = linregress(x, y)
        volume_slope.iloc[i] = slope
    
    # Combine momentum with volume signal
    combined = momentum * np.sign(volume_slope)
    
    # Apply smoothing
    result = combined.rolling(window=3, min_periods=1).mean()
    return result

def _volatility_regime_adjusted_return(df):
    """Volatility Regime Adjusted Return"""
    close = df['close']
    high = df['high']
    low = df['low']
    
    # Calculate daily ranges
    daily_range = (high - low) / close
    
    # Calculate volatility (std of daily ranges)
    volatility = daily_range.rolling(window=10, min_periods=5).std()
    
    # Calculate 5-day return
    returns = close.pct_change(periods=5)
    
    # Normalize return by volatility
    normalized_return = returns / (volatility + 1e-8)
    
    # Apply rank transformation (cross-sectional)
    result = pd.Series(index=df.index, dtype=float)
    for date in df.index:
        date_data = normalized_return.loc[date]
        if not pd.isna(date_data):
            # Simple rank (assuming single stock context)
            result.loc[date] = date_data
    
    return result

def _price_level_relative_strength(df):
    """Price-Level Relative Strength"""
    close = df['close']
    high = df['high']
    low = df['low']
    
    resistance = high.rolling(window=20, min_periods=10).max()
    support = low.rolling(window=20, min_periods=10).min()
    
    # Calculate distances
    dist_resistance = resistance - close
    dist_support = close - support
    
    # Create ratio and apply log transformation
    ratio = dist_resistance / (dist_support + 1e-8)
    result = np.log(ratio + 1)
    
    return result

def _volume_price_divergence(df):
    """Volume-Price Divergence Detector"""
    close = df['close']
    volume = df['volume']
    
    price_slope = pd.Series(index=df.index, dtype=float)
    volume_slope = pd.Series(index=df.index, dtype=float)
    
    for i in range(9, len(df)):
        # Price slope
        x = np.arange(10)
        y_price = close.iloc[i-9:i+1].values
        slope_price, _, _, _, _ = linregress(x, y_price)
        price_slope.iloc[i] = slope_price
        
        # Volume slope
        y_volume = volume.iloc[i-9:i+1].values
        slope_volume, _, _, _, _ = linregress(x, y_volume)
        volume_slope.iloc[i] = slope_volume
    
    # Detect divergence
    divergence = price_slope * volume_slope
    result = np.sign(divergence) * np.abs(price_slope)
    
    return result

def _overnight_gap_persistence(df):
    """Overnight Gap Persistence Factor"""
    open_price = df['open']
    close = df['close']
    volume = df['volume']
    
    # Calculate overnight gap
    gap = (open_price - close.shift(1)) / close.shift(1)
    
    # Count consecutive same-sign gaps
    gap_persistence = pd.Series(index=df.index, dtype=float)
    for i in range(4, len(df)):
        current_gap = gap.iloc[i]
        if pd.isna(current_gap):
            continue
            
        consecutive_count = 1
        for j in range(1, 5):
            if i - j < 0:
                break
            prev_gap = gap.iloc[i - j]
            if pd.isna(prev_gap):
                break
            if np.sign(prev_gap) == np.sign(current_gap):
                consecutive_count += 1
            else:
                break
        
        gap_persistence.iloc[i] = consecutive_count * np.abs(current_gap)
    
    # Volume percentile rank
    volume_rank = volume.rolling(window=20, min_periods=10).rank(pct=True)
    
    # Combine with mean reversion adjustment
    result = gap_persistence * volume_rank * -1  # Mean reversion
    
    return result

def _liquidity_adjusted_momentum_reversal(df):
    """Liquidity-Adjusted Momentum Reversal"""
    close = df['close']
    volume = df['volume']
    
    # Price reversal signal
    reversal = -close.pct_change(periods=3)
    
    # Dollar volume
    dollar_volume = close * volume
    avg_dollar_volume = dollar_volume.rolling(window=10, min_periods=5).mean()
    
    # Liquidity rank
    liquidity_rank = avg_dollar_volume.rolling(window=20, min_periods=10).rank(pct=True)
    
    # Volatility scaling
    volatility = close.pct_change().rolling(window=10, min_periods=5).std()
    
    # Combine signals
    result = reversal * liquidity_rank / (volatility + 1e-8)
    
    return result

def _intraday_strength_consistency(df):
    """Intraday Strength Consistency"""
    open_price = df['open']
    high = df['high']
    low = df['low']
    close = df['close']
    
    # Calculate intraday returns
    open_close_return = (close - open_price) / open_price
    high_close_return = (close - high) / high
    low_close_return = (close - low) / low
    
    # Count positive returns
    positive_count = (
        (open_close_return > 0).astype(int) +
        (high_close_return > 0).astype(int) +
        (low_close_return > 0).astype(int)
    )
    
    # Average intraday range
    intraday_range = (high - low) / close
    
    # Combine signals
    result = positive_count * intraday_range
    
    return result

def _sector_relative_breakout(df):
    """Sector-Relative Breakout Detection"""
    close = df['close']
    volume = df['volume']
    
    # Calculate stock and sector returns (assuming single stock context)
    stock_return = close.pct_change(periods=5)
    
    # For single stock, use market-relative approach
    # In practice, this would use actual sector data
    relative_performance = stock_return  # Simplified for single stock
    
    # Identify breakout conditions
    breakout = pd.Series(index=df.index, dtype=float)
    for i in range(19, len(df)):
        current_perf = relative_performance.iloc[i]
        if pd.isna(current_perf):
            continue
            
        max_prev_perf = relative_performance.iloc[i-19:i].max()
        breakout.iloc[i] = 1 if current_perf > max_prev_perf else 0
    
    # Volume surge
    volume_ratio = volume / volume.rolling(window=20, min_periods=10).mean()
    
    # Combine signals
    result = breakout * volume_ratio
    
    return result
