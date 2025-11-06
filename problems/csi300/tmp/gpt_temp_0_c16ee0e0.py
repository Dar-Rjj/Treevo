import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Price-Volume Fractal Dynamics with Liquidity Regime Switching alpha factor
    """
    # Initialize result series
    result = pd.Series(index=df.index, dtype=float)
    
    # Required columns
    cols_required = ['open', 'high', 'low', 'close', 'amount', 'volume']
    if not all(col in df.columns for col in cols_required):
        return result
    
    # Calculate daily returns and price ranges
    returns = df['close'].pct_change()
    price_range = (df['high'] - df['low']) / df['close']
    
    # 1. Multi-Fractal Market Microstructure Analysis
    # Hurst exponent estimation using high-low range scaling (5-day window)
    hurst_window = 5
    hurst_values = pd.Series(index=df.index, dtype=float)
    
    for i in range(hurst_window, len(df)):
        window_data = price_range.iloc[i-hurst_window:i]
        if len(window_data) < 2:
            continue
            
        # Simple Hurst estimation using range scaling
        log_ranges = np.log(window_data + 1e-8)
        log_time = np.log(np.arange(1, len(window_data) + 1))
        if len(log_ranges) > 1:
            hurst = np.polyfit(log_time, log_ranges, 1)[0]
            hurst_values.iloc[i] = hurst
    
    # Volume-weighted price path complexity
    volume_weighted_complexity = (df['volume'] * price_range).rolling(window=3).std()
    
    # 2. Liquidity Fractal Patterns
    # Volume clustering across time scales (3-day vs 10-day)
    volume_clustering = (df['volume'].rolling(window=3).std() / 
                        (df['volume'].rolling(window=10).std() + 1e-8))
    
    # Amount volatility as liquidity depth persistence
    amount_volatility = df['amount'].rolling(window=5).std()
    
    # 3. Volume-Price Impact Asymmetry
    # Buy-volume vs sell-volume price impact estimation
    price_impact = (df['close'] - df['open']) / df['close']
    volume_impact_ratio = (df['volume'] * np.abs(price_impact)).rolling(window=5).mean()
    
    # 4. Time-Varying Market Depth
    # Price resilience after liquidity shocks (3-day recovery)
    liquidity_shock = (df['volume'] > df['volume'].rolling(window=10).mean() * 1.5)
    price_resilience = pd.Series(index=df.index, dtype=float)
    
    for i in range(3, len(df)):
        if liquidity_shock.iloc[i-3]:
            recovery = (df['close'].iloc[i] - df['close'].iloc[i-3]) / df['close'].iloc[i-3]
            price_resilience.iloc[i] = recovery
    
    # 5. Multi-Scale Price Momentum with Fractal Context
    # Short-term momentum (3-day)
    momentum_short = df['close'].pct_change(periods=3)
    
    # Medium-term momentum (8-day)
    momentum_medium = df['close'].pct_change(periods=8)
    
    # 6. Liquidity-Constrained Momentum
    # Momentum adjusted for available liquidity
    liquidity_adjusted_momentum = momentum_short * (df['amount'].rolling(window=5).mean() / 
                                                   (df['amount'].rolling(window=20).mean() + 1e-8))
    
    # 7. Volume Fractal Regime Identification
    # Multi-scale volume entropy (3-day vs 10-day volatility ratio)
    volume_regime = (df['volume'].rolling(window=3).std() / 
                    (df['volume'].rolling(window=10).std() + 1e-8))
    
    # 8. Price-Volume Fractal Coherence
    # Correlation between price range and volume
    price_volume_coherence = (price_range.rolling(window=5).corr(df['volume']))
    
    # 9. Regime-Weighted Factor Signals
    # Combine components with regime weighting
    regime_weight = 1 / (1 + np.exp(-volume_regime))  # Sigmoid weighting
    
    # Fractal dimension component
    fractal_component = hurst_values.fillna(0) * volume_weighted_complexity.fillna(0)
    
    # Momentum component adjusted for liquidity
    momentum_component = liquidity_adjusted_momentum.fillna(0) * momentum_medium.fillna(0)
    
    # Market microstructure component
    microstructure_component = (volume_impact_ratio.fillna(0) * 
                              price_resilience.fillna(0) * 
                              price_volume_coherence.fillna(0))
    
    # Final alpha factor combining all components with regime weighting
    alpha_factor = (regime_weight.fillna(0) * fractal_component +
                   (1 - regime_weight.fillna(0)) * momentum_component +
                   microstructure_component * volume_clustering.fillna(0))
    
    # Normalize the factor
    if len(alpha_factor) > 0:
        alpha_factor = (alpha_factor - alpha_factor.rolling(window=20).mean()) / (alpha_factor.rolling(window=20).std() + 1e-8)
    
    return alpha_factor.fillna(0)
