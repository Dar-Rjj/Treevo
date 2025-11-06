import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Dynamic Price Efficiency with Microstructure Anchoring factor
    """
    data = df.copy()
    
    # Price Efficiency Measurement
    # Intraday Efficiency Ratio
    daily_range = data['high'] - data['low']
    close_to_high = (data['close'] - data['low']) / daily_range.replace(0, np.nan)
    close_to_low = (data['high'] - data['close']) / daily_range.replace(0, np.nan)
    efficiency_ratio = 1 - 2 * np.abs(close_to_high - 0.5)
    
    # Efficiency persistence (autocorrelation)
    efficiency_persistence = efficiency_ratio.rolling(window=5, min_periods=3).corr(efficiency_ratio.shift(1))
    
    # Efficiency volatility
    efficiency_volatility = efficiency_ratio.rolling(window=10, min_periods=5).std()
    
    # Efficiency regimes
    stable_trending = ((efficiency_ratio > efficiency_ratio.rolling(window=20).quantile(0.7)) & 
                      (efficiency_volatility < efficiency_volatility.rolling(window=20).quantile(0.3))).astype(float)
    
    inefficient_choppiness = ((efficiency_ratio < efficiency_ratio.rolling(window=20).quantile(0.3)) & 
                             (efficiency_volatility > efficiency_volatility.rolling(window=20).quantile(0.7))).astype(float)
    
    efficiency_improvement = (efficiency_ratio - efficiency_ratio.rolling(window=5).mean()) / efficiency_ratio.rolling(window=5).std()
    
    # Microstructure Anchoring Analysis
    # Overnight gap anchoring
    overnight_gap = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
    gap_persistence = overnight_gap.rolling(window=5).apply(lambda x: np.corrcoef(x[:-1], x[1:])[0,1] if len(x) > 1 else np.nan)
    
    # Intraday range anchoring
    high_to_open = (data['high'] - data['open']) / (data['high'] - data['low']).replace(0, np.nan)
    low_to_open = (data['open'] - data['low']) / (data['high'] - data['low']).replace(0, np.nan)
    range_anchoring = 1 - 2 * np.abs(high_to_open - 0.5)
    
    # Volume-weighted price anchoring
    vwap = (data['high'] + data['low'] + data['close']) / 3
    volume_weighted_anchoring = (data['volume'] * np.abs(data['close'] - vwap)).rolling(window=10).mean()
    volume_concentration = data['volume'].rolling(window=10).std() / data['volume'].rolling(window=10).mean()
    
    # Anchor strength
    anchor_strength = (range_anchoring.rolling(window=10).std() + 
                      volume_weighted_anchoring.rolling(window=10).std()).replace(0, np.nan)
    anchor_strength = 1 / anchor_strength
    
    # Anchoring patterns
    institutional_accumulation = ((anchor_strength > anchor_strength.rolling(window=20).quantile(0.7)) & 
                                 (efficiency_ratio > efficiency_ratio.rolling(window=20).quantile(0.7))).astype(float)
    
    retail_noise = ((anchor_strength < anchor_strength.rolling(window=20).quantile(0.3)) & 
                   (efficiency_ratio < efficiency_ratio.rolling(window=20).quantile(0.3))).astype(float)
    
    anchor_migration = anchor_strength.diff(3).rolling(window=5).mean()
    
    # Adaptive Signal Generation
    # Anchor break efficiency
    anchor_break_threshold = data['close'].rolling(window=20).std() * 0.5
    anchor_breaks = (np.abs(data['close'] - vwap) > anchor_break_threshold).astype(int)
    break_efficiency = efficiency_ratio * anchor_breaks
    
    # Efficiency-anchor convergence
    efficiency_anchor_convergence = (efficiency_ratio - range_anchoring).abs().rolling(window=5).mean()
    
    # Anchor resilience during efficiency changes
    efficiency_changes = efficiency_ratio.diff(3).abs()
    anchor_resilience = anchor_strength / (efficiency_changes.replace(0, np.nan) + 1e-6)
    
    # Anchor-induced efficiency improvements
    anchor_efficiency_interaction = (efficiency_improvement * anchor_migration).rolling(window=5).mean()
    
    # Generate Dynamic Trading Signals
    # Momentum acceleration signal
    momentum_acceleration = ((stable_trending > 0.5) & 
                            (break_efficiency > break_efficiency.rolling(window=20).quantile(0.7)) & 
                            (institutional_accumulation > 0.5)).astype(float)
    
    # Range-bound continuation signal
    range_continuation = ((inefficient_choppiness > 0.5) & 
                         (retail_noise > 0.5) & 
                         (efficiency_anchor_convergence < efficiency_anchor_convergence.rolling(window=20).quantile(0.3))).astype(float)
    
    # Trend initiation signal
    trend_initiation = ((efficiency_improvement > efficiency_improvement.rolling(window=20).quantile(0.7)) & 
                       (anchor_migration > anchor_migration.rolling(window=20).quantile(0.7)) & 
                       (anchor_efficiency_interaction > anchor_efficiency_interaction.rolling(window=20).quantile(0.7))).astype(float)
    
    # Combine signals into final factor
    factor = (momentum_acceleration * 0.4 + 
              range_continuation * (-0.3) + 
              trend_initiation * 0.6 + 
              efficiency_improvement * 0.2 + 
              anchor_resilience.rank(pct=True) * 0.1)
    
    # Normalize the factor
    factor = (factor - factor.rolling(window=60, min_periods=20).mean()) / factor.rolling(window=60, min_periods=20).std()
    
    return factor
