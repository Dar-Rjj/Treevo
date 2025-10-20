import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Volume-Price Divergence with Regime-Switching Momentum factor
    """
    data = df.copy()
    
    # Calculate Price-Based Momentum Indicators
    data['price_momentum_8'] = data['close'] / data['close'].shift(8) - 1
    data['price_momentum_13'] = data['close'] / data['close'].shift(13) - 1
    data['price_momentum_21'] = data['close'] / data['close'].shift(21) - 1
    
    # Calculate Volume-Based Momentum Indicators
    data['volume_momentum_8'] = data['volume'] / data['volume'].shift(8) - 1
    data['volume_momentum_13'] = data['volume'] / data['volume'].shift(13) - 1
    data['volume_momentum_21'] = data['volume'] / data['volume'].shift(21) - 1
    
    # Quantify Divergence Patterns
    data['divergence_ratio_8'] = data['price_momentum_8'] / (data['volume_momentum_8'] + 1e-8)
    data['divergence_ratio_13'] = data['price_momentum_13'] / (data['volume_momentum_13'] + 1e-8)
    data['divergence_ratio_21'] = data['price_momentum_21'] / (data['volume_momentum_21'] + 1e-8)
    
    # Calculate divergence persistence
    data['divergence_persistence'] = (
        (data['divergence_ratio_8'].rolling(5).std() + 
         data['divergence_ratio_13'].rolling(5).std() + 
         data['divergence_ratio_21'].rolling(5).std()) / 3
    )
    
    # Detect Regime-Switching Conditions
    # Volatility regimes
    data['true_range'] = data['high'] - data['low']
    data['atr_5'] = data['true_range'].rolling(5).mean()
    data['volatility_regime'] = (data['atr_5'] > data['atr_5'].rolling(21).mean()).astype(int)
    
    # Trend transitions
    data['short_ma'] = data['close'].rolling(5).mean()
    data['medium_ma'] = data['close'].rolling(13).mean()
    data['long_ma'] = data['close'].rolling(21).mean()
    
    data['trend_strength'] = (
        (data['short_ma'] > data['medium_ma']).astype(int) + 
        (data['medium_ma'] > data['long_ma']).astype(int) + 
        (data['price_momentum_8'] > 0).astype(int)
    )
    
    # Liquidity regime changes
    data['volume_zscore'] = (
        data['volume'] - data['volume'].rolling(21).mean()
    ) / (data['volume'].rolling(21).std() + 1e-8)
    
    data['liquidity_regime'] = (data['volume_zscore'].abs() > 1).astype(int)
    
    # Build Regime Classification System
    data['regime_state'] = (
        data['volatility_regime'] * 4 + 
        (data['trend_strength'] >= 2).astype(int) * 2 + 
        data['liquidity_regime']
    )
    
    # Calculate regime persistence
    data['regime_persistence'] = (
        data['regime_state'].rolling(5).apply(lambda x: len(set(x)) == 1, raw=False)
    ).astype(int)
    
    # Develop Adaptive Momentum Framework
    # Regime-adjusted momentum
    data['regime_weighted_momentum'] = (
        data['price_momentum_8'] * (1 + data['volatility_regime'] * 0.3) * 
        (1 + (data['trend_strength'] / 3)) * 
        (1 + data['liquidity_regime'] * 0.2)
    )
    
    # Dynamic momentum thresholds
    data['momentum_threshold'] = (
        data['regime_weighted_momentum'].rolling(21).std() * 
        (1 + data['volatility_regime'] * 0.5)
    )
    
    # Momentum acceleration
    data['momentum_acceleration'] = (
        data['regime_weighted_momentum'] - data['regime_weighted_momentum'].shift(3)
    )
    
    # Generate Predictive Alpha Factor
    # Combine divergence with regime-aware momentum
    data['divergence_score'] = (
        data['divergence_ratio_8'].rank(pct=True) + 
        data['divergence_ratio_13'].rank(pct=True) + 
        data['divergence_ratio_21'].rank(pct=True)
    ) / 3
    
    data['regime_adjusted_divergence'] = (
        data['divergence_score'] * 
        (1 + (data['regime_state'] / 7)) *  # Normalize regime state
        data['regime_persistence']
    )
    
    # Composite divergence-momentum score
    data['composite_score'] = (
        data['regime_adjusted_divergence'] * 
        data['regime_weighted_momentum'] * 
        (1 + data['momentum_acceleration'].clip(-0.1, 0.1))
    )
    
    # Signal validation system
    data['price_volume_confirmation'] = (
        (data['price_momentum_8'] * data['volume_momentum_8'] > 0).astype(int) +
        (data['price_momentum_13'] * data['volume_momentum_13'] > 0).astype(int) +
        (data['price_momentum_21'] * data['volume_momentum_21'] > 0).astype(int)
    ) / 3
    
    # Final predictive factor
    data['alpha_factor'] = (
        data['composite_score'] * 
        data['price_volume_confirmation'] * 
        (1 - data['divergence_persistence'].rank(pct=True))
    )
    
    return data['alpha_factor']
