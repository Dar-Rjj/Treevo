import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Helper function for price-volume entropy
    def price_volume_entropy(close, volume, window=5):
        price_change = close.pct_change().abs()
        volume_change = volume.pct_change().abs()
        combined = price_change * volume_change
        rolling_entropy = combined.rolling(window=window).apply(
            lambda x: -np.sum(x * np.log(x + 1e-8)) if (x > 0).any() else 0
        )
        return rolling_entropy / rolling_entropy.rolling(window=20).max()
    
    # Helper function for volume-price entropy
    def volume_price_entropy(close, volume, window=5):
        price_vol = close.pct_change().rolling(window=3).std()
        volume_vol = volume.pct_change().rolling(window=3).std()
        combined = price_vol * volume_vol
        rolling_entropy = combined.rolling(window=window).apply(
            lambda x: -np.sum(x * np.log(x + 1e-8)) if (x > 0).any() else 0
        )
        return rolling_entropy / rolling_entropy.rolling(window=20).max()
    
    # Helper function for volume-entropy complexity
    def volume_entropy_complexity(volume, window=10):
        volume_returns = volume.pct_change()
        complexity = volume_returns.rolling(window=window).apply(
            lambda x: np.sqrt(np.mean(x**2)) / (np.mean(np.abs(x)) + 1e-8)
        )
        return complexity / complexity.rolling(window=20).max()
    
    # Helper function for volume fractal dimension
    def volume_fractal_dimension(volume, window=10):
        volume_range = volume.rolling(window=window).max() - volume.rolling(window=window).min()
        volume_std = volume.rolling(window=window).std()
        fractal = np.log(volume_range + 1e-8) / (np.log(volume_std + 1e-8) + 1e-8)
        return fractal / fractal.rolling(window=20).max()
    
    # Calculate entropy measures
    pv_entropy = price_volume_entropy(data['close'], data['volume'])
    vp_entropy = volume_price_entropy(data['close'], data['volume'])
    vol_entropy_complexity = volume_entropy_complexity(data['volume'])
    vol_fractal = volume_fractal_dimension(data['volume'])
    
    # Volatility-Entropy Momentum Decay components
    morning_momentum_vol_entropy = ((data['open'] - data['close'].shift(1)) / data['close'].shift(1)) / \
                                  (data['high'] - data['low']) * (1 - pv_entropy)
    
    intraday_accel_vol_entropy = ((data['close'] - data['open']) / data['open']) / \
                                (data['high'] - data['low']) * vp_entropy
    
    momentum_persistence_vol_entropy = ((data['close'] - data['close'].shift(2)) / data['close'].shift(2)) / \
                                      data['close'].rolling(window=5).std() * vol_fractal
    
    # Entropy-Efficiency in Momentum-Volatility components
    gap_fill_vol_entropy = ((data['high'] - data['open']) / (data['open'] - data['close'].shift(1))) / \
                          (data['high'] - data['low']) * (1 - pv_entropy)
    gap_fill_vol_entropy = gap_fill_vol_entropy.replace([np.inf, -np.inf], 0).fillna(0)
    
    range_util_entropy = (abs(data['close'] - data['close'].shift(1)) / (data['high'] - data['low'])) * \
                        ((data['close'] - data['open']) / data['open']) * vol_entropy_complexity
    
    volatility_compression_entropy = ((data['high'] - data['low']) / data['close'].shift(1)) / \
                                    abs(data['open'] - data['close'].shift(1)) * data['close'].shift(1) * vol_fractal
    volatility_compression_entropy = volatility_compression_entropy.replace([np.inf, -np.inf], 0).fillna(0)
    
    # Volume-Entropy Momentum Elasticity components
    volume_accel_entropy = ((data['volume'] / data['volume'].shift(1)) / abs(data['close'] - data['open']) / data['open']) * \
                          pv_entropy
    volume_accel_entropy = volume_accel_entropy.replace([np.inf, -np.inf], 0).fillna(0)
    
    trade_size_entropy_sensitivity = ((data['amount'] / data['volume']) / abs(data['open'] - data['close'].shift(1)) / \
                                     data['close'].shift(1)) * vol_entropy_complexity
    trade_size_entropy_sensitivity = trade_size_entropy_sensitivity.replace([np.inf, -np.inf], 0).fillna(0)
    
    volume_pressure_entropy_efficiency = (data['volume'] / data['volume'].shift(5)) * \
                                       ((data['close'] - data['close'].shift(1)) / data['close'].shift(1)) * vol_fractal
    
    # Liquidity Depth Entropy Fracture components
    support_liquidity_vol_entropy = ((data['open'] - data['low']) / (data['amount'] / data['volume'])) / \
                                   (data['high'] - data['low']) * (1 - pv_entropy)
    support_liquidity_vol_entropy = support_liquidity_vol_entropy.replace([np.inf, -np.inf], 0).fillna(0)
    
    resistance_liquidity_efficiency_entropy = ((data['high'] - data['close']) / (data['amount'] / data['volume'])) / \
                                            (data['high'] - data['low']) * vp_entropy
    resistance_liquidity_efficiency_entropy = resistance_liquidity_efficiency_entropy.replace([np.inf, -np.inf], 0).fillna(0)
    
    liquidity_imbalance_vol_entropy = ((support_liquidity_vol_entropy - resistance_liquidity_efficiency_entropy) / \
                                      (data['high'] - data['low'])) * vol_fractal
    
    # Liquidity-Entropy Cascade Dynamics components
    morning_liquidity_entropy = (data['volume'] / data['volume'].shift(1)) * \
                               ((data['open'] - data['close'].shift(1)) / data['close'].shift(1)) * pv_entropy
    
    intraday_liquidity_entropy_persistence = ((data['volume'] / data['volume'].shift(1)) / abs(data['close'] - data['open']) / \
                                             data['open']) * ((data['close'] - data['close'].shift(1)) / data['close'].shift(1)) * \
                                             vol_entropy_complexity
    intraday_liquidity_entropy_persistence = intraday_liquidity_entropy_persistence.replace([np.inf, -np.inf], 0).fillna(0)
    
    # Entropy-Elasticity Factor Synthesis
    core_entropy_elasticity = morning_momentum_vol_entropy * intraday_accel_vol_entropy
    persistence_entropy_elasticity = momentum_persistence_vol_entropy * range_util_entropy
    efficiency_entropy_elasticity = gap_fill_vol_entropy * volatility_compression_entropy
    
    volume_entropy_elasticity = volume_accel_entropy * volume_pressure_entropy_efficiency
    depth_entropy_elasticity = support_liquidity_vol_entropy * resistance_liquidity_efficiency_entropy
    cascade_entropy_elasticity = morning_liquidity_entropy * intraday_liquidity_entropy_persistence
    
    # Cross-Entropy Integration
    volatility_liquidity_entropy = core_entropy_elasticity * depth_entropy_elasticity
    momentum_liquidity_entropy_fracture = persistence_entropy_elasticity * volume_entropy_elasticity
    regime_transition_entropy = efficiency_entropy_elasticity * cascade_entropy_elasticity
    
    # Entropy-Breakout Synchronization
    volatility_entropy_breakout = core_entropy_elasticity * (1 - pv_entropy)
    liquidity_entropy_breakout_confirmation = volume_entropy_elasticity * vol_entropy_complexity
    fractal_breakout_entropy_alignment = depth_entropy_elasticity * vol_fractal
    
    regime_momentum_entropy_compression = persistence_entropy_elasticity * vol_fractal
    noise_entropy_breakout = efficiency_entropy_elasticity * (1 - pv_entropy)
    flow_entropy_breakout_quality = cascade_entropy_elasticity * vol_entropy_complexity
    
    # Composite Entropy-Volatility Alpha
    momentum_volatility_entropy_component = core_entropy_elasticity * persistence_entropy_elasticity
    liquidity_entropy_fracture_component = volume_entropy_elasticity * depth_entropy_elasticity
    cross_entropy_dynamics = volatility_liquidity_entropy * momentum_liquidity_entropy_fracture
    
    volatility_entropy_weight = morning_momentum_vol_entropy / intraday_accel_vol_entropy
    volatility_entropy_weight = volatility_entropy_weight.replace([np.inf, -np.inf], 0).fillna(0)
    
    liquidity_depth_entropy_weight = support_liquidity_vol_entropy / resistance_liquidity_efficiency_entropy
    liquidity_depth_entropy_weight = liquidity_depth_entropy_weight.replace([np.inf, -np.inf], 0).fillna(0)
    
    entropy_transition_sensitivity = liquidity_imbalance_vol_entropy * regime_transition_entropy
    
    volatility_entropy_breakout_enhanced = volatility_entropy_breakout * regime_momentum_entropy_compression
    liquidity_entropy_breakout_enhanced = liquidity_entropy_breakout_confirmation * flow_entropy_breakout_quality
    fractal_entropy_breakout_enhanced = fractal_breakout_entropy_alignment * noise_entropy_breakout
    
    # Final Alpha Output
    primary_signal = (momentum_volatility_entropy_component * 0.3 + 
                     liquidity_entropy_fracture_component * 0.3 + 
                     cross_entropy_dynamics * 0.4)
    
    regime_weights = (volatility_entropy_weight * 0.4 + 
                     liquidity_depth_entropy_weight * 0.3 + 
                     entropy_transition_sensitivity * 0.3)
    
    breakout_enhancement = (volatility_entropy_breakout_enhanced * 0.4 + 
                           liquidity_entropy_breakout_enhanced * 0.3 + 
                           fractal_entropy_breakout_enhanced * 0.3)
    
    # Final composite alpha
    alpha = (primary_signal * 0.5 + 
            regime_weights * 0.3 + 
            breakout_enhancement * 0.2)
    
    # Normalize and clean
    alpha = alpha.replace([np.inf, -np.inf], 0).fillna(0)
    alpha = (alpha - alpha.rolling(window=20).mean()) / (alpha.rolling(window=20).std() + 1e-8)
    
    return alpha
