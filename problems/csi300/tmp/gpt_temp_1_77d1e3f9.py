import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # 1. Momentum Decay with Volume Confirmation
    # Compute weighted momentum using exponential decay
    decay_weights = np.exp(-np.arange(20) / 5)  # 5-day decay factor
    decay_weights = decay_weights / decay_weights.sum()
    
    # Calculate returns for momentum
    returns = data['close'].pct_change()
    
    # Weighted momentum calculation
    weighted_momentum = pd.Series(index=data.index, dtype=float)
    for i in range(20, len(data)):
        window_returns = returns.iloc[i-19:i+1]  # 20-day window
        weighted_momentum.iloc[i] = (window_returns * decay_weights).sum()
    
    # Volume ratio (current/20-day average)
    volume_avg_20 = data['volume'].rolling(window=20, min_periods=1).mean()
    volume_ratio = data['volume'] / volume_avg_20
    
    # Momentum with volume confirmation
    momentum_volume = weighted_momentum * volume_ratio
    
    # 2. Intraday Range Efficiency
    # Calculate true range
    true_range = pd.DataFrame(index=data.index)
    true_range['hl'] = data['high'] - data['low']
    true_range['hc'] = abs(data['high'] - data['close'].shift(1))
    true_range['lc'] = abs(data['low'] - data['close'].shift(1))
    true_range['tr'] = true_range[['hl', 'hc', 'lc']].max(axis=1)
    
    # Absolute price movement
    price_movement = abs(data['close'] - data['open'])
    
    # Range efficiency
    range_efficiency = price_movement / true_range['tr']
    range_efficiency = range_efficiency.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # 3. Volatility Regime Adjusted Momentum
    # 10-day return momentum
    momentum_10 = data['close'].pct_change(periods=10)
    
    # 20-day volatility (standard deviation of returns)
    vol_20 = returns.rolling(window=20, min_periods=1).std()
    
    # Volatility adjusted momentum
    vol_adjusted_momentum = momentum_10 / (vol_20 + 1e-8)  # Add small constant to avoid division by zero
    
    # 4. Volume-Price Divergence
    # 5-day price slope using linear regression
    price_slope = pd.Series(index=data.index, dtype=float)
    for i in range(5, len(data)):
        y = data['close'].iloc[i-4:i+1].values
        x = np.arange(5)
        slope = np.polyfit(x, y, 1)[0]
        price_slope.iloc[i] = slope
    
    # 5-day volume slope
    volume_slope = pd.Series(index=data.index, dtype=float)
    for i in range(5, len(data)):
        y = data['volume'].iloc[i-4:i+1].values
        x = np.arange(5)
        slope = np.polyfit(x, y, 1)[0]
        volume_slope.iloc[i] = slope
    
    # Volume-price divergence
    volume_price_divergence = price_slope * volume_slope
    
    # 5. Liquidity-Adjusted Reversal
    # 1-day reversal
    reversal = -1 * returns
    
    # Volume percentile (current/10-day)
    volume_percentile = pd.Series(index=data.index, dtype=float)
    for i in range(10, len(data)):
        current_volume = data['volume'].iloc[i]
        past_volumes = data['volume'].iloc[i-9:i+1]
        volume_percentile.iloc[i] = current_volume / past_volumes.mean()
    
    # Liquidity-adjusted reversal
    liquidity_reversal = reversal * volume_percentile
    
    # Combine all factors with equal weights
    factors = pd.DataFrame({
        'momentum_volume': momentum_volume,
        'range_efficiency': range_efficiency,
        'vol_adjusted_momentum': vol_adjusted_momentum,
        'volume_price_divergence': volume_price_divergence,
        'liquidity_reversal': liquidity_reversal
    })
    
    # Standardize each factor
    factors_standardized = (factors - factors.rolling(window=60, min_periods=1).mean()) / factors.rolling(window=60, min_periods=1).std()
    
    # Equal-weighted combination
    final_factor = factors_standardized.mean(axis=1)
    
    return final_factor
