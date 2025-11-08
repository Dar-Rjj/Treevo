import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Multi-Scale Entropic Momentum Factors
    Combines entropy analysis, efficiency measures, and momentum with regime adaptation
    """
    data = df.copy()
    
    # Calculate returns for entropy analysis
    data['returns'] = data['close'].pct_change()
    data['abs_returns'] = data['returns'].abs()
    
    # 1. Multi-Timeframe Entropy Analysis
    # Short-term price entropy (3-day)
    def calculate_entropy(returns_series):
        abs_returns = returns_series.abs()
        total_abs = abs_returns.sum()
        if total_abs == 0:
            return 0
        probabilities = abs_returns / total_abs
        probabilities = probabilities[probabilities > 0]  # Remove zeros for log
        entropy = -np.sum(probabilities * np.log(probabilities))
        return entropy
    
    # Rolling entropy calculations
    data['entropy_3d'] = data['abs_returns'].rolling(window=3).apply(
        lambda x: calculate_entropy(x), raw=False
    )
    data['entropy_10d'] = data['abs_returns'].rolling(window=10).apply(
        lambda x: calculate_entropy(x), raw=False
    )
    
    # Entropy momentum and trends
    data['entropy_momentum_3d'] = data['entropy_3d'] - data['entropy_3d'].shift(3)
    data['entropy_trend_10d'] = (data['entropy_10d'] - data['entropy_10d'].shift(5)) - \
                               (data['entropy_10d'] - data['entropy_10d'].shift(10))
    
    # Entropy regime classification
    data['high_entropy_regime'] = (data['entropy_3d'] > data['entropy_10d']).astype(int)
    
    # Volume flow entropy assessment
    data['volume_concentration'] = data['volume'] / data['volume'].rolling(window=5).sum()
    data['volume_entropy'] = data['volume_concentration'].rolling(window=5).apply(
        lambda x: calculate_entropy(x), raw=False
    )
    
    # Volume entropy regime alignment
    data['volume_entropy_trend'] = data['volume_entropy'] - data['volume_entropy'].shift(3)
    data['entropy_regime_consistency'] = np.where(
        (data['high_entropy_regime'] == 1) & (data['volume_entropy_trend'] > 0), 1,
        np.where((data['high_entropy_regime'] == 0) & (data['volume_entropy_trend'] <= 0), 1, 0)
    )
    
    # Price-volume entropy divergence
    data['entropy_divergence'] = data['entropy_momentum_3d'] - data['volume_entropy_trend']
    
    # Multi-scale entropy integration
    data['unified_entropy_score'] = (
        data['entropy_momentum_3d'].fillna(0) * 0.4 +
        data['entropy_trend_10d'].fillna(0) * 0.3 +
        data['entropy_regime_consistency'] * 0.3
    )
    
    # 2. Fractal-Entropic Efficiency Analysis
    # Basic range efficiency
    data['range_efficiency'] = (data['close'] - data['open']).abs() / (data['high'] - data['low'])
    data['range_efficiency'] = data['range_efficiency'].replace([np.inf, -np.inf], 0)
    
    # Entropy-weighted efficiency
    data['entropy_weighted_efficiency'] = data['range_efficiency'] * (1 - data['entropy_3d'].fillna(0))
    
    # Multi-timeframe efficiency consistency
    data['efficiency_3d'] = data['range_efficiency'].rolling(window=3).mean()
    data['efficiency_10d'] = data['range_efficiency'].rolling(window=10).mean()
    data['efficiency_stability'] = 1 - (data['efficiency_3d'] - data['efficiency_10d']).abs()
    
    # Volume-entropy distribution analysis
    data['volume_per_movement'] = data['amount'] / (data['high'] - data['low'])
    data['volume_per_movement'] = data['volume_per_movement'].replace([np.inf, -np.inf], 0)
    data['volume_dist_entropy'] = data['volume_per_movement'].rolling(window=5).apply(
        lambda x: calculate_entropy(x), raw=False
    )
    
    # Efficiency-entropy correlation analysis
    data['efficiency_entropy_corr'] = -data['range_efficiency'] * data['entropy_3d'].fillna(0)
    
    # 3. Multi-Scale Momentum with Entropic Filtering
    # Short-term momentum
    data['momentum_3d'] = (data['close'] - data['close'].shift(3)) / data['close'].shift(3)
    data['entropy_adj_momentum_3d'] = data['momentum_3d'] * (1 - data['entropy_3d'].fillna(0))
    
    # Medium-term momentum
    data['momentum_10d'] = (data['close'] - data['close'].shift(10)) / data['close'].shift(10)
    data['entropy_adj_momentum_10d'] = data['momentum_10d'] * (1 - data['entropy_10d'].fillna(0))
    
    # Multi-scale momentum integration
    data['integrated_momentum'] = (
        data['entropy_adj_momentum_3d'].fillna(0) * 0.6 +
        data['entropy_adj_momentum_10d'].fillna(0) * 0.4
    ) * data['entropy_regime_consistency']
    
    # Volume momentum with entropic context
    data['volume_momentum_3d'] = (data['volume'] - data['volume'].shift(3)) / data['volume'].shift(3)
    data['entropy_adj_volume_momentum'] = data['volume_momentum_3d'] * (1 - data['volume_entropy'].fillna(0))
    
    # Volume-price entropy alignment
    data['entropy_alignment_score'] = 1 - (data['entropy_momentum_3d'] - data['volume_entropy_trend']).abs()
    
    # 4. Entropic Cross-Factor Synthesis
    # Entropy-weighted momentum signals
    data['entropy_weighted_momentum'] = data['integrated_momentum'] * (1 - data['unified_entropy_score'].fillna(0))
    
    # Efficiency-entropy balanced scores
    data['efficiency_entropy_balanced'] = (
        data['entropy_weighted_efficiency'].fillna(0) * 0.5 +
        data['efficiency_entropy_corr'].fillna(0) * 0.3 +
        data['efficiency_stability'].fillna(0) * 0.2
    )
    
    # Volume entropy confirmation
    data['volume_entropy_confirmation'] = (
        data['entropy_adj_volume_momentum'].fillna(0) * 0.4 +
        data['entropy_alignment_score'].fillna(0) * 0.3 +
        data['volume_dist_entropy'].fillna(0) * 0.3
    )
    
    # Final regime-adaptive scoring
    high_entropy_component = (
        data['entropy_weighted_momentum'].fillna(0) * 0.3 +
        data['volume_entropy_confirmation'].fillna(0) * 0.4 +
        data['entropy_divergence'].fillna(0) * 0.3
    )
    
    low_entropy_component = (
        data['efficiency_entropy_balanced'].fillna(0) * 0.5 +
        data['entropy_weighted_momentum'].fillna(0) * 0.3 +
        data['efficiency_stability'].fillna(0) * 0.2
    )
    
    # Final factor combining both regimes
    final_factor = np.where(
        data['high_entropy_regime'] == 1,
        high_entropy_component,
        low_entropy_component
    )
    
    return pd.Series(final_factor, index=data.index, name='multi_scale_entropic_momentum')
