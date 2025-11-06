import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Cross-Asset Relative Value with Microstructure Momentum factor
    Combines relative momentum divergence with order flow analysis
    """
    # Make a copy to avoid modifying original dataframe
    data = df.copy()
    
    # 1. Cross-Sectional Momentum Ranking
    # Calculate 3-day and 8-day momentum
    data['momentum_3d'] = data['close'] / data['close'].shift(3) - 1
    data['momentum_8d'] = data['close'] / data['close'].shift(8) - 1
    
    # Calculate cross-sectional percentile ranks (within each day)
    data['momentum_3d_rank'] = data.groupby(data.index)['momentum_3d'].transform(
        lambda x: x.rank(pct=True)
    )
    data['momentum_8d_rank'] = data.groupby(data.index)['momentum_8d'].transform(
        lambda x: x.rank(pct=True)
    )
    
    # 2. Relative Momentum Divergence Detection
    data['momentum_divergence'] = (
        data['momentum_3d_rank'] - data['momentum_8d_rank']
    ).abs()
    
    # 3. Sector-Relative Momentum Positioning
    # Calculate sector average momentum (using 8-day momentum as sector proxy)
    data['sector_avg_momentum'] = data.groupby(data.index)['momentum_8d'].transform('mean')
    data['sector_relative_deviation'] = (
        data['momentum_8d'] - data['sector_avg_momentum']
    ) / data['sector_avg_momentum'].abs().clip(lower=1e-8)
    
    # 4. Volume-Price Efficiency Gradient
    # Calculate daily price range efficiency
    data['daily_range'] = data['high'] - data['low']
    data['price_movement_per_volume'] = (
        (data['close'] - data['open']).abs() / data['volume'].clip(lower=1)
    )
    
    # 3-day efficiency gradient
    data['efficiency_gradient'] = (
        data['price_movement_per_volume'] - 
        data['price_movement_per_volume'].shift(3)
    ) / data['price_movement_per_volume'].shift(3).abs().clip(lower=1e-8)
    
    # 5. Intraday Order Flow Patterns
    # Opening vs closing efficiency ratio
    data['open_close_efficiency'] = (
        (data['close'] - data['open']).abs() / 
        data['daily_range'].clip(lower=1e-8)
    )
    
    # 3-day average of opening efficiency
    data['open_efficiency_3d_avg'] = data['open_close_efficiency'].rolling(
        window=3, min_periods=1
    ).mean()
    
    # 6. Microstructure Momentum Signals
    # Volume-weighted price momentum
    data['vw_momentum'] = (
        data['momentum_3d'] * data['volume'] / 
        data['volume'].rolling(window=5, min_periods=1).mean()
    )
    
    # Order flow acceleration (change in volume-weighted momentum)
    data['order_flow_acceleration'] = (
        data['vw_momentum'] - data['vw_momentum'].shift(2)
    )
    
    # 7. Bid-Ask Spread Proxy Analysis
    # Use daily range normalized by price as spread proxy
    data['spread_proxy'] = (
        data['daily_range'] / data['close'].rolling(window=5, min_periods=1).mean()
    )
    
    # 8. Price Impact Sensitivity
    data['price_impact'] = (
        (data['close'] - data['open']).abs() / 
        data['volume'].clip(lower=1).pow(0.5)
    )
    
    # 9. Microstructure Regime Classification
    # High impact vs low impact regime indicator
    data['impact_regime'] = (
        data['price_impact'] > data['price_impact'].rolling(window=10, min_periods=1).median()
    ).astype(float)
    
    # 10. Composite Alpha Factor Construction
    
    # Cross-asset relative value component
    relative_value = (
        -data['momentum_divergence'] *  # Mean reversion on divergence
        data['sector_relative_deviation'].abs() *  # Amplify extreme deviations
        np.sign(data['sector_relative_deviation'])  # Direction for mean reversion
    )
    
    # Microstructure momentum component
    microstructure_momentum = (
        data['order_flow_acceleration'] *
        data['efficiency_gradient'] *
        (1 - data['spread_proxy'])  # Penalize high spread environments
    )
    
    # Regime-adaptive weighting
    regime_weight = 0.7 + 0.3 * data['impact_regime']  # Adjust weights by regime
    
    # Final composite factor
    alpha_factor = (
        regime_weight * relative_value +
        (1 - regime_weight) * microstructure_momentum
    )
    
    # Normalize by cross-sectional z-score each day
    def normalize_cross_sectional(group):
        if len(group) > 1:
            return (group - group.mean()) / group.std()
        else:
            return group * 0  # Return zeros if only one asset
    
    alpha_factor_normalized = alpha_factor.groupby(alpha_factor.index).transform(
        normalize_cross_sectional
    )
    
    return alpha_factor_normalized
