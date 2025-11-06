import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(data):
    """
    Generate a composite alpha factor combining multiple heuristics:
    - Volatility-Adjusted Relative Strength Momentum
    - Volume-Weighted Price Reversal with Acceleration
    - Bid-Ask Spread Impact on Momentum Persistence
    - Liquidity-Adjusted Mean Reversion with Regime Detection
    - Multi-Timeframe Price-Volume Divergence
    """
    
    # Extract price and volume data
    close = data['close']
    high = data['high']
    low = data['low']
    volume = data['volume']
    
    # Initialize factor series
    factor = pd.Series(index=close.index, dtype=float)
    
    # 1. Volatility-Adjusted Relative Strength Momentum
    # Calculate momentum components
    short_return = close.pct_change(5)
    long_return = close.pct_change(20)
    momentum = short_return - long_return
    
    # Volatility adjustment using high-low range
    hl_range = (high - low) / close
    vol_adj = hl_range.rolling(20).std()
    vol_adj_momentum = momentum / (vol_adj + 1e-8)
    
    # Cross-sectional ranking normalized to [-1,1]
    def normalize_rank(series):
        return (series.rank(pct=True) - 0.5) * 2
    
    rs_momentum = vol_adj_momentum.groupby(level=0).transform(normalize_rank)
    
    # 2. Volume-Weighted Price Reversal with Acceleration
    recent_return = close.pct_change(1)
    prev_return = close.pct_change(5)
    reversal = recent_return - prev_return
    
    # Acceleration component
    current_return = close.pct_change(1)
    prev_day_return = close.shift(1).pct_change(1)
    acceleration = current_return - prev_day_return
    
    reversal_accel = reversal + acceleration
    
    # Volume weighting
    avg_volume = volume.rolling(20).mean()
    volume_ratio = volume / (avg_volume + 1e-8)
    volume_weighted_reversal = reversal_accel * volume_ratio
    
    # 3. Bid-Ask Spread Impact on Momentum Persistence
    # Effective spread approximation
    spread_ratio = (high - low) / close
    
    # Momentum persistence
    momentum_10d = close.pct_change(10)
    
    # Calculate momentum consistency (count positive days in last 5 days)
    def count_positive_returns(series):
        returns = series.pct_change().iloc[-5:]
        return (returns > 0).sum() / 5.0
    
    persistence_ratio = close.rolling(6).apply(count_positive_returns, raw=False)
    
    spread_momentum = momentum_10d * persistence_ratio * (1 - spread_ratio)
    
    # Volume confirmation
    volume_trend = volume.rolling(5).mean() / volume.rolling(20).mean()
    volume_adjusted_momentum = spread_momentum * np.sign(volume_trend - 1)
    
    # 4. Liquidity-Adjusted Mean Reversion with Regime Detection
    # Mean reversion signal
    price_mean = close.rolling(20).mean()
    price_std = close.rolling(20).std()
    price_zscore = (close - price_mean) / (price_std + 1e-8)
    
    # Liquidity regime detection
    volume_mean = volume.rolling(20).mean()
    volume_std = volume.rolling(20).std()
    volume_shock = (volume - volume_mean) / (volume_std + 1e-8)
    
    # Modify reversion strength by regime
    regime_multiplier = np.where(np.abs(volume_shock) > 2, 1.5, 1.0)
    liquidity_reversion = -price_zscore * regime_multiplier
    
    # Volatility filter using ATR
    tr = pd.concat([high - low, 
                   abs(high - close.shift(1)), 
                   abs(low - close.shift(1))], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    hist_vol = close.pct_change().rolling(20).std()
    vol_condition = atr / (hist_vol + 1e-8)
    
    filtered_reversion = liquidity_reversion / (vol_condition + 1e-8)
    
    # 5. Multi-Timeframe Price-Volume Divergence
    # Short-term PV relationship
    short_price_return = close.pct_change(5)
    short_volume_change = volume.pct_change(5)
    short_pv_corr = short_price_return.rolling(10).corr(short_volume_change)
    
    # Medium-term PV relationship
    medium_price_return = close.pct_change(20)
    medium_volume_change = volume.pct_change(20)
    medium_pv_corr = medium_price_return.rolling(20).corr(medium_volume_change)
    
    # Divergence detection
    pv_divergence = short_pv_corr - medium_pv_corr
    
    # Combine all components with equal weights
    components = pd.DataFrame({
        'rs_momentum': rs_momentum,
        'volume_reversal': volume_weighted_reversal,
        'spread_momentum': volume_adjusted_momentum,
        'liquidity_reversion': filtered_reversion,
        'pv_divergence': pv_divergence
    })
    
    # Normalize each component and combine
    normalized_components = components.apply(lambda x: (x - x.mean()) / (x.std() + 1e-8))
    factor = normalized_components.mean(axis=1)
    
    return factor
