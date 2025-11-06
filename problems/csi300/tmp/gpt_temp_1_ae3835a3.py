import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Multi-Scale Momentum Fracture Analysis
    # Short-term Momentum Fracture (2-4 days)
    short_term_price_fracture = (df['close'] / df['close'].shift(2) - 1) - (df['close'] / df['close'].shift(4) - 1)
    short_term_volume_fracture = (df['volume'] / df['volume'].shift(2) - 1) - (df['volume'] / df['volume'].shift(4) - 1)
    
    # Medium-term Momentum Fracture (4-8 days)
    medium_term_price_fracture = (df['close'] / df['close'].shift(8) - 1) - (df['close'] / df['close'].shift(4) - 1)
    medium_term_volume_fracture = (df['volume'] / df['volume'].shift(8) - 1) - (df['volume'] / df['volume'].shift(4) - 1)
    
    # Fracture Divergence Detection
    price_fracture_divergence = short_term_price_fracture - medium_term_price_fracture
    volume_fracture_divergence = short_term_volume_fracture - medium_term_volume_fracture
    combined_fracture_divergence = price_fracture_divergence * volume_fracture_divergence
    
    # Bid-Ask Imbalance Microstructure Integration
    # Imbalance-Based Fracture Patterns
    spread = (df['high'] - df['low']) / ((df['high'] + df['low']) / 2)
    short_term_imbalance_fracture = (spread / spread.shift(2) - 1) - (spread / spread.shift(5) - 1)
    imbalance_volatility = spread.pct_change().rolling(window=4, min_periods=1).std()
    
    # Volume-Price Efficiency Analysis
    volume_efficiency = df['volume'] / ((df['high'] - df['low']) / ((df['high'] + df['low']) / 2))
    
    # Calculate price efficiency using rolling sum of absolute returns
    price_returns = df['close'].pct_change()
    price_efficiency = (df['close'] / df['close'].shift(2) - 1) / price_returns.rolling(window=2, min_periods=1).apply(lambda x: np.sum(np.abs(x)), raw=True)
    efficiency_divergence = (volume_efficiency - price_efficiency) * np.sign(volume_efficiency)
    
    # Imbalance-Volume Confirmation
    imbalance_volume_alignment = (np.sign(short_term_imbalance_fracture) == np.sign(volume_fracture_divergence)).astype(float)
    imbalance_volume_strength = np.abs(short_term_imbalance_fracture) * np.abs(volume_fracture_divergence)
    confirmation_score = imbalance_volume_alignment * imbalance_volume_strength
    
    # Regime-Transition Context
    # Transition-Based Regime Signals
    short_term_transition = price_returns.rolling(window=4, min_periods=1).std()
    medium_term_transition = price_returns.rolling(window=8, min_periods=1).std()
    transition_regime = short_term_transition / medium_term_transition
    
    # Momentum Efficiency Analysis
    momentum_efficiency = (df['close'] / df['close'].shift(4) - 1) / price_returns.rolling(window=4, min_periods=1).apply(lambda x: np.sum(np.abs(x)), raw=True)
    
    # Calculate efficiency consistency (count of days with same direction)
    momentum_efficiency_sign = np.sign(momentum_efficiency)
    efficiency_consistency = momentum_efficiency_sign.rolling(window=4, min_periods=1).apply(lambda x: np.sum(x == x.iloc[-1]) if len(x) > 0 else 0, raw=False)
    
    # Transition Adjustment
    transition_multiplier = 1 + (transition_regime - 1) * 0.4
    efficiency_weight = momentum_efficiency * efficiency_consistency
    
    # Signal Integration and Enhancement
    # Core Fracture Component
    base_fracture_score = combined_fracture_divergence * confirmation_score
    
    # Regime-Adaptive Weighting
    transition_weighted_fracture = base_fracture_score * transition_multiplier
    efficiency_weighted_confirmation = transition_weighted_fracture * efficiency_weight
    regime_optimized_signal = efficiency_weighted_confirmation * (1 + np.abs(transition_regime - 1))
    
    # Opening Validation
    opening_gap = df['open'] / df['close'].shift(1) - 1
    opening_range = (df['high'] - df['low']) / df['open']
    opening_efficiency = np.abs(opening_gap) / opening_range
    
    direction_confirmation = (np.sign(regime_optimized_signal) == np.sign(opening_gap)).astype(float)
    strength_confirmation = (opening_efficiency > 0.6).astype(float)
    combined_opening_score = direction_confirmation * strength_confirmation
    
    opening_enhanced_signal = regime_optimized_signal * (1 + combined_opening_score * 0.25)
    
    # Final Alpha Factor Generation
    # Fracture Persistence Validation
    momentum_persistence = np.sign(short_term_price_fracture).rolling(window=4, min_periods=1).apply(lambda x: np.sum(x == x.iloc[-1]) if len(x) > 0 else 0, raw=False)
    imbalance_persistence = np.sign(short_term_imbalance_fracture).rolling(window=4, min_periods=1).apply(lambda x: np.sum(x == x.iloc[-1]) if len(x) > 0 else 0, raw=False)
    combined_persistence = (momentum_persistence + imbalance_persistence) / 8
    
    # Enhanced Alpha Factor
    alpha_factor = opening_enhanced_signal * (1 + combined_persistence)
    
    return alpha_factor
