import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Generate alpha factor combining momentum acceleration, volume-price breakout,
    order flow imbalance, volatility-adjusted returns, and fractal market efficiency.
    """
    # Copy dataframe to avoid modifying original
    data = df.copy()
    
    # 1. Momentum Acceleration Factor
    # Price momentum with exponential decay weighting
    close_prices = data['close']
    
    # Calculate momentum with different lookback periods
    momentum_short = close_prices.pct_change(periods=5)
    momentum_medium = close_prices.pct_change(periods=10)
    momentum_long = close_prices.pct_change(periods=20)
    
    # Apply exponential decay weights (more weight to recent periods)
    weights_short = np.exp(-np.arange(5) / 2.5)  # Decay factor 2.5
    weights_medium = np.exp(-np.arange(10) / 5)  # Decay factor 5
    weights_long = np.exp(-np.arange(20) / 10)   # Decay factor 10
    
    # Normalize weights
    weights_short = weights_short / weights_short.sum()
    weights_medium = weights_medium / weights_medium.sum()
    weights_long = weights_long / weights_long.sum()
    
    # Calculate weighted momentum
    weighted_momentum_short = close_prices.rolling(window=5).apply(
        lambda x: np.sum(x.pct_change().dropna() * weights_short[:len(x)-1]) if len(x) > 1 else 0
    )
    weighted_momentum_medium = close_prices.rolling(window=10).apply(
        lambda x: np.sum(x.pct_change().dropna() * weights_medium[:len(x)-1]) if len(x) > 1 else 0
    )
    
    # Acceleration measurement (change in momentum)
    momentum_acceleration = weighted_momentum_short - weighted_momentum_medium
    
    # 2. Volume-Price Breakout Detector
    # Congestion identification - price range compression
    price_range = (data['high'] - data['low']) / data['close']
    avg_range_20 = price_range.rolling(window=20).mean()
    range_compression = (avg_range_20 - price_range) / avg_range_20
    
    # Volume confirmation
    volume_avg_20 = data['volume'].rolling(window=20).mean()
    volume_surge = (data['volume'] - volume_avg_20) / volume_avg_20
    
    # Breakout validation - price boundary penetration
    high_20 = data['high'].rolling(window=20).max()
    low_20 = data['low'].rolling(window=20).min()
    
    upper_breakout = (data['close'] - high_20.shift(1)) / high_20.shift(1)
    lower_breakout = (data['close'] - low_20.shift(1)) / low_20.shift(1)
    
    breakout_signal = np.where(upper_breakout > 0, upper_breakout * volume_surge,
                              np.where(lower_breakout < 0, lower_breakout * volume_surge, 0))
    
    # 3. Order Flow Imbalance
    # Buy-sell pressure calculation
    price_change = data['close'].diff()
    up_volume = np.where(price_change > 0, data['volume'], 0)
    down_volume = np.where(price_change < 0, data['volume'], 0)
    
    # Rolling sums for pressure calculation
    up_volume_5 = pd.Series(up_volume, index=data.index).rolling(window=5).sum()
    down_volume_5 = pd.Series(down_volume, index=data.index).rolling(window=5).sum()
    
    # Order flow imbalance ratio
    ofi_ratio = (up_volume_5 - down_volume_5) / (up_volume_5 + down_volume_5 + 1e-8)
    
    # 4. Volatility-Adjusted Returns
    # Realized volatility calculation
    returns = data['close'].pct_change()
    realized_vol_20 = returns.rolling(window=20).std()
    
    # Volatility regime classification
    vol_regime = realized_vol_20.rolling(window=60).apply(
        lambda x: 1 if x.iloc[-1] > np.percentile(x.dropna(), 70) else 
                  (-1 if x.iloc[-1] < np.percentile(x.dropna(), 30) else 0)
    )
    
    # Volatility-adjusted returns
    vol_adjusted_returns = returns / (realized_vol_20 + 1e-8)
    regime_enhanced_returns = vol_adjusted_returns * (1 + 0.2 * vol_regime)
    
    # 5. Fractal Market Efficiency
    # Price path complexity using Hurst exponent approximation
    def hurst_approximation(series):
        if len(series) < 10:
            return 0.5
        lags = range(2, min(10, len(series)))
        tau = [np.std(np.subtract(series[lag:], series[:-lag])) for lag in lags]
        poly = np.polyfit(np.log(lags), np.log(tau), 1)
        return poly[0]
    
    hurst_20 = data['close'].rolling(window=20).apply(hurst_approximation, raw=False)
    
    # Volume distribution patterns (entropy-like measure)
    def volume_entropy(volumes):
        if len(volumes) < 5:
            return 0
        normalized = volumes / (volumes.sum() + 1e-8)
        entropy = -np.sum(normalized * np.log(normalized + 1e-8))
        return entropy / np.log(len(volumes))  # Normalize by max entropy
    
    volume_entropy_10 = data['volume'].rolling(window=10).apply(volume_entropy, raw=False)
    
    # Market efficiency score
    efficiency_score = 1 - (np.abs(hurst_20 - 0.5) + (1 - volume_entropy_10)) / 2
    
    # Combine all factors with appropriate weights
    factor = (
        0.25 * momentum_acceleration +
        0.20 * breakout_signal +
        0.20 * ofi_ratio +
        0.20 * regime_enhanced_returns.rolling(window=5).mean() +
        0.15 * efficiency_score
    )
    
    # Final normalization
    factor_normalized = (factor - factor.rolling(window=60).mean()) / (factor.rolling(window=60).std() + 1e-8)
    
    return factor_normalized
