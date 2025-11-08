import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Generate a novel alpha factor based on multi-dimensional alignment analysis
    combining price momentum, volume confirmation, regime consistency, and microstructure efficiency.
    """
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate basic price features
    data['returns_1d'] = data['close'] / data['close'].shift(1) - 1
    data['returns_5d'] = data['close'] / data['close'].shift(5) - 1
    data['returns_20d'] = data['close'] / data['close'].shift(20) - 1
    
    # Volatility calculations
    data['current_vol'] = (data['high'] - data['low']) / data['close']
    data['short_vol'] = data['current_vol'].rolling(window=3).mean()
    data['medium_vol'] = data['current_vol'].rolling(window=15).mean()
    
    # Volume features
    data['volume_ma_20'] = data['volume'].rolling(window=20).mean()
    data['volume_intensity'] = data['volume'] / data['volume_ma_20'].shift(1)
    data['volume_momentum_3d'] = data['volume'] / data['volume'].shift(3) - 1
    data['amount_efficiency'] = data['amount'] / (data['volume'] * (data['high'] + data['low'] + data['close']) / 3)
    
    # Price efficiency measures
    data['intraday_efficiency'] = (data['close'] - data['open']) / (data['high'] - data['low']).replace(0, np.nan)
    data['gap_efficiency'] = abs(data['open'] / data['close'].shift(1) - 1) / (data['current_vol'].replace(0, np.nan))
    data['trend_efficiency'] = abs(data['returns_1d']) / (data['current_vol'].replace(0, np.nan))
    
    # Market quality indicators
    data['spread_proxy'] = (data['high'] - data['low']) / ((data['high'] + data['low']) / 2).replace(0, np.nan)
    data['liquidity_score'] = data['volume'] / (data['high'] - data['low']).replace(0, np.nan)
    
    # Multi-Timeframe Momentum Alignment Score
    def calculate_momentum_alignment(row):
        momentum_signs = [np.sign(row['returns_1d']), np.sign(row['returns_5d']), np.sign(row['returns_20d'])]
        directional_consistency = sum(momentum_signs) / 3.0  # Range: -1 to 1
        
        # Magnitude progression (acceleration/deceleration)
        momentum_magnitudes = [abs(row['returns_1d']), abs(row['returns_5d']/5), abs(row['returns_20d']/20)]
        magnitude_ratio = momentum_magnitudes[0] / (np.mean(momentum_magnitudes[1:]) + 1e-8)
        
        # Weighted alignment score
        weights = [0.4, 0.35, 0.25]  # Higher weight to shorter timeframes
        weighted_momentum = (weights[0] * row['returns_1d'] + 
                           weights[1] * row['returns_5d'] + 
                           weights[2] * row['returns_20d'])
        
        alignment_score = (directional_consistency * 0.4 + 
                          np.tanh(magnitude_ratio - 1) * 0.3 + 
                          weighted_momentum * 0.3)
        
        return alignment_score
    
    # Volume-Price Alignment Score
    def calculate_volume_alignment(row):
        volume_momentum_signs = [np.sign(row['volume_momentum_3d']), np.sign(row['volume_intensity'] - 1)]
        price_momentum_signs = [np.sign(row['returns_1d']), np.sign(row['returns_5d'])]
        
        # Volume confirmation
        volume_confirmation = sum([1 for v, p in zip(volume_momentum_signs, price_momentum_signs) if v == p]) / 2.0
        
        # Volume leadership (volume precedes price)
        volume_strength = abs(row['volume_momentum_3d']) / (abs(row['returns_1d']) + 1e-8)
        
        alignment_strength = (volume_confirmation * 0.6 + 
                            np.tanh(volume_strength - 1) * 0.4)
        
        return alignment_strength
    
    # Regime Consistency Score
    def calculate_regime_consistency(row):
        # Volatility regime consistency
        vol_regime_current = 'normal'
        if row['current_vol'] < 0.7 * row['medium_vol']:
            vol_regime_current = 'low'
        elif row['current_vol'] > 1.3 * row['medium_vol']:
            vol_regime_current = 'high'
        
        vol_regime_short = 'normal'
        if row['short_vol'] < 0.7 * row['medium_vol']:
            vol_regime_short = 'low'
        elif row['short_vol'] > 1.3 * row['medium_vol']:
            vol_regime_short = 'high'
        
        vol_consistency = 1.0 if vol_regime_current == vol_regime_short else 0.5
        
        # Volume regime consistency
        volume_regime = 'normal'
        if row['volume_intensity'] > 1.5:
            volume_regime = 'high'
        elif row['volume_intensity'] < 0.7:
            volume_regime = 'low'
        
        volume_trend_regime = 'normal'
        if abs(row['volume_momentum_3d']) > 0.2:
            volume_trend_regime = 'high' if row['volume_momentum_3d'] > 0 else 'low'
        
        volume_consistency = 1.0 if volume_regime == volume_trend_regime else 0.5
        
        return (vol_consistency + volume_consistency) / 2.0
    
    # Efficiency Composite Score
    def calculate_efficiency_score(row):
        # Price efficiency components
        price_efficiency = (np.tanh(row['intraday_efficiency']) * 0.4 + 
                          np.tanh(1 - row['gap_efficiency']) * 0.3 + 
                          np.tanh(1 - row['trend_efficiency']) * 0.3)
        
        # Volume efficiency
        volume_efficiency = np.tanh(1 - row['amount_efficiency']) * 0.6 + np.tanh(row['liquidity_score']) * 0.4
        
        # Market quality
        market_quality = np.tanh(1 - row['spread_proxy']) * 0.5 + np.tanh(row['liquidity_score']) * 0.5
        
        efficiency_composite = (price_efficiency * 0.5 + volume_efficiency * 0.3 + market_quality * 0.2)
        
        return efficiency_composite
    
    # Apply scoring functions
    data['momentum_alignment'] = data.apply(calculate_momentum_alignment, axis=1)
    data['volume_alignment'] = data.apply(calculate_volume_alignment, axis=1)
    data['regime_consistency'] = data.apply(calculate_regime_consistency, axis=1)
    data['efficiency_score'] = data.apply(calculate_efficiency_score, axis=1)
    
    # Final Multi-Dimensional Alignment Factor
    def calculate_final_factor(row):
        # Volatility adjustment
        vol_adjustment = 1.0 / (1.0 + abs(row['current_vol'] - row['medium_vol']))
        
        # Combine components with regime-adaptive weights
        momentum_component = row['momentum_alignment'] * 0.35
        volume_component = row['volume_alignment'] * 0.25
        regime_component = row['regime_consistency'] * 0.20
        efficiency_component = row['efficiency_score'] * 0.20
        
        base_factor = (momentum_component + volume_component + regime_component + efficiency_component)
        
        # Apply volatility adjustment
        final_factor = base_factor * vol_adjustment
        
        return final_factor
    
    # Generate final factor values
    data['alpha_factor'] = data.apply(calculate_final_factor, axis=1)
    
    # Clean and return the factor series
    factor_series = data['alpha_factor'].replace([np.inf, -np.inf], np.nan).fillna(0)
    
    return factor_series
