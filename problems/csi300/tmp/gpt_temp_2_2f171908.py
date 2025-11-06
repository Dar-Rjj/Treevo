import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Cross-Asset Momentum Spillover with Liquidity-Adjusted Signal Decay
    """
    # Create copy to avoid modifying original dataframe
    data = df.copy()
    
    # 1. Cross-Asset Momentum Framework
    # Calculate rolling returns for momentum (using only past data)
    data['stock_return_5d'] = data['close'].pct_change(5)
    data['stock_return_10d'] = data['close'].pct_change(10)
    data['stock_return_20d'] = data['close'].pct_change(20)
    
    # Sector-relative momentum proxy (using market-wide momentum as sector proxy)
    data['market_return_5d'] = data['close'].pct_change(5).rolling(window=10).mean()
    data['sector_relative_strength'] = data['stock_return_5d'] - data['market_return_5d']
    
    # Momentum persistence (days of consecutive sector outperformance)
    data['outperformance_flag'] = (data['sector_relative_strength'] > 0).astype(int)
    data['momentum_persistence'] = data['outperformance_flag'].rolling(window=10, min_periods=1).apply(
        lambda x: len(x) - np.argmin(x[::-1]) if np.any(x) else 0, raw=True
    )
    
    # Momentum acceleration (change in relative strength)
    data['momentum_acceleration'] = data['sector_relative_strength'].diff(3)
    
    # 2. Liquidity Regime Detection
    # Effective spread using daily high-low range
    data['daily_range'] = (data['high'] - data['low']) / data['close']
    data['effective_spread'] = data['daily_range'].rolling(window=5).mean()
    
    # Price impact per unit volume
    data['abs_return'] = abs(data['close'].pct_change())
    data['price_impact'] = data['abs_return'] / (data['volume'].replace(0, np.nan) + 1e-10)
    data['avg_price_impact'] = data['price_impact'].rolling(window=5).mean()
    
    # Volume clustering patterns for latent liquidity
    data['volume_zscore'] = (data['volume'] - data['volume'].rolling(window=20).mean()) / data['volume'].rolling(window=20).std()
    data['volume_clustering'] = data['volume_zscore'].rolling(window=5).std()
    
    # Liquidity shocks (abnormal volume-to-range ratios)
    data['volume_range_ratio'] = data['volume'] / (data['daily_range'] + 1e-10)
    data['liquidity_shock'] = (data['volume_range_ratio'] - data['volume_range_ratio'].rolling(window=20).mean()) / data['volume_range_ratio'].rolling(window=20).std()
    
    # 3. Signal Decay Mechanism
    # Momentum half-life using autocorrelation decay (simplified)
    returns_5d = data['close'].pct_change(5)
    autocorr_1 = returns_5d.rolling(window=20, min_periods=10).apply(
        lambda x: pd.Series(x).autocorr(lag=1) if len(x) >= 10 else np.nan, raw=False
    )
    data['momentum_half_life'] = -np.log(2) / np.log(np.abs(autocorr_1 + 1e-10))
    data['momentum_half_life'] = data['momentum_half_life'].clip(1, 50)
    
    # Adjust decay rates based on liquidity conditions
    liquidity_score = (data['effective_spread'].rank() + data['avg_price_impact'].rank()) / 2
    data['adjusted_decay_rate'] = 1 / (data['momentum_half_life'] * (1 + 0.5 * liquidity_score))
    
    # 4. Adaptive Alpha Construction
    # Combined momentum signals with decay adjustments
    momentum_signal = (
        data['sector_relative_strength'].rank() * 0.4 +
        data['momentum_persistence'].rank() * 0.3 +
        data['momentum_acceleration'].rank() * 0.3
    )
    
    # Apply liquidity filters
    liquidity_filter = np.where(
        data['liquidity_shock'].abs() < 2,
        1.0,
        np.exp(-data['liquidity_shock'].abs() / 4)
    )
    
    # Dynamic signal weighting based on cross-asset momentum coherence
    momentum_coherence = data['sector_relative_strength'].rolling(window=10).std()
    coherence_weight = 1 / (1 + momentum_coherence)
    
    # Final alpha factor with decay adjustment
    raw_alpha = momentum_signal * liquidity_filter * coherence_weight
    decay_adjusted_alpha = raw_alpha * np.exp(-data['adjusted_decay_rate'])
    
    # Normalize and return
    alpha_series = pd.Series(decay_adjusted_alpha, index=data.index)
    alpha_series = (alpha_series - alpha_series.rolling(window=20).mean()) / alpha_series.rolling(window=20).std()
    
    return alpha_series
