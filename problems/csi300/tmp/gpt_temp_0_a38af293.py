import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Cross-Association Momentum Divergence factor combining cross-asset momentum,
    volume-volatility dynamics, and temporal pattern analysis.
    """
    # Initialize result series
    result = pd.Series(index=df.index, dtype=float)
    
    # Required columns check
    required_cols = ['open', 'high', 'low', 'close', 'amount', 'volume']
    if not all(col in df.columns for col in required_cols):
        return result
    
    # Calculate daily returns
    returns = df['close'].pct_change()
    
    # 1. Cross-Asset Momentum Analysis
    # Stock momentum vs rolling sector proxy (using rolling market proxy)
    stock_momentum = returns.rolling(window=5).mean()
    market_proxy = returns.rolling(window=20).mean()  # Proxy for sector momentum
    
    momentum_divergence = stock_momentum - market_proxy
    momentum_strength = momentum_divergence.rolling(window=10).std()
    
    # 2. Volume-Volatility Dynamics
    # Volume spike detection vs historical percentile
    volume_percentile = df['volume'].rolling(window=20).apply(
        lambda x: (x.iloc[-1] > np.percentile(x, 80)), raw=False
    )
    
    # Volatility ratio (stock vs rolling market volatility)
    stock_volatility = returns.rolling(window=5).std()
    market_volatility = returns.rolling(window=20).std()
    volatility_ratio = stock_volatility / (market_volatility + 1e-8)
    
    # Return per unit volume efficiency
    volume_efficiency = returns / (df['volume'] + 1e-8)
    volume_efficiency_ma = volume_efficiency.rolling(window=5).mean()
    
    # 3. Temporal Pattern Analysis
    # Morning vs afternoon return asymmetry (using intraday high/low patterns)
    morning_strength = (df['high'] - df['open']) / (df['open'] + 1e-8)
    afternoon_strength = (df['close'] - df['low']) / (df['low'] + 1e-8)
    temporal_asymmetry = morning_strength - afternoon_strength
    
    # Overnight-intraday return alignment
    overnight_returns = (df['open'] - df['close'].shift(1)) / (df['close'].shift(1) + 1e-8)
    intraday_returns = (df['close'] - df['open']) / (df['open'] + 1e-8)
    return_alignment = np.sign(overnight_returns) * np.sign(intraday_returns)
    
    # Multi-day pattern recognition (3-day momentum consistency)
    pattern_consistency = returns.rolling(window=3).apply(
        lambda x: 1 if all(x > 0) or all(x < 0) else 0, raw=False
    )
    
    # 4. Signal Integration Framework
    # Cross-asset momentum divergence signals
    momentum_signal = momentum_divergence / (momentum_strength + 1e-8)
    
    # Volume-volatility anomaly detection
    volume_anomaly = volume_percentile * volatility_ratio * volume_efficiency_ma
    
    # Temporal pattern momentum scoring
    temporal_score = (temporal_asymmetry.rolling(window=5).mean() + 
                     return_alignment.rolling(window=5).mean() + 
                     pattern_consistency.rolling(window=5).mean()) / 3
    
    # 5. Composite Factor Construction
    # Regime-adaptive signal weighting based on market volatility
    regime_weight = 1 / (market_volatility + 1e-8)
    regime_weight = regime_weight / regime_weight.rolling(window=20).mean()
    
    # Signal conflict resolution logic
    momentum_volume_alignment = np.sign(momentum_signal) * np.sign(volume_anomaly)
    conflict_resolution = np.where(momentum_volume_alignment > 0, 1, 0.5)
    
    # Dynamic signal persistence rules
    signal_persistence = (momentum_signal.rolling(window=3).std() + 
                         volume_anomaly.rolling(window=3).std() + 
                         temporal_score.rolling(window=3).std()) / 3
    
    # Final composite factor
    composite_factor = (
        momentum_signal * regime_weight * 
        volume_anomaly * conflict_resolution * 
        temporal_score / (signal_persistence + 1e-8)
    )
    
    # Normalize and fill result
    result = composite_factor.rolling(window=20).apply(
        lambda x: (x.iloc[-1] - x.mean()) / (x.std() + 1e-8), raw=False
    )
    
    return result.fillna(0)
