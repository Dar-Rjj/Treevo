import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Momentum-Decay Asymmetry Analysis alpha factor
    Combines asymmetric momentum decay patterns with volume-weighted momentum structure
    and price-level dynamics for enhanced return prediction
    """
    
    # Calculate returns for different timeframes
    returns_1d = df['close'].pct_change()
    returns_3d = df['close'].pct_change(3)
    returns_5d = df['close'].pct_change(5)
    returns_10d = df['close'].pct_change(10)
    
    # Volume metrics
    volume_ma_5 = df['volume'].rolling(window=5).mean()
    volume_ma_20 = df['volume'].rolling(window=20).mean()
    high_volume_threshold = volume_ma_20 * 1.5
    low_volume_threshold = volume_ma_20 * 0.7
    
    # Price level metrics
    high_20 = df['high'].rolling(window=20).max()
    low_20 = df['low'].rolling(window=20).min()
    current_range_position = (df['close'] - low_20) / (high_20 - low_20)
    
    # 1. Asymmetric Momentum Decay Patterns
    # Calculate momentum decay rates for positive vs negative momentum
    positive_momentum = returns_5d.where(returns_5d > 0)
    negative_momentum = returns_5d.where(returns_5d < 0)
    
    # Momentum persistence using autocorrelation decay
    momentum_persistence_10 = returns_1d.rolling(window=10).apply(
        lambda x: np.corrcoef(x[:-1], x[1:])[0,1] if len(x) > 1 and not np.all(x == x[0]) else 0, 
        raw=False
    )
    
    # 2. Volume-Weighted Momentum Structure
    # Volume-adjusted momentum
    volume_persistence_ratio = volume_ma_5 / volume_ma_20
    volume_adjusted_momentum = returns_5d * volume_persistence_ratio
    
    # High-volume momentum acceleration
    high_volume_mask = df['volume'] > high_volume_threshold
    high_volume_momentum_accel = returns_1d.where(high_volume_mask, 0) * returns_5d
    
    # Low-volume momentum stability
    low_volume_mask = df['volume'] < low_volume_threshold
    low_volume_momentum_stability = returns_5d.rolling(window=5).std().where(low_volume_mask, 1)
    low_volume_momentum = returns_5d / low_volume_momentum_stability
    
    # 3. Price-Level Momentum Dynamics
    # Support/resistance momentum effects
    near_high = (current_range_position > 0.8).astype(int)
    near_low = (current_range_position < 0.2).astype(int)
    
    support_momentum = returns_5d.where(near_low == 1, 0)
    resistance_momentum = returns_5d.where(near_high == 1, 0)
    
    # Breakout momentum acceleration (price breaking recent range)
    range_breakout_up = (df['close'] > high_20.shift(1)).astype(int)
    range_breakout_down = (df['close'] < low_20.shift(1)).astype(int)
    breakout_momentum = returns_1d * (range_breakout_up - range_breakout_down)
    
    # 4. Core momentum components
    # Asymmetric decay component
    positive_decay_rate = positive_momentum.rolling(window=5).apply(
        lambda x: np.exp(-1/len(x)) if len(x) > 0 else 1, raw=False
    )
    negative_decay_rate = negative_momentum.rolling(window=5).apply(
        lambda x: np.exp(-1/len(x)) if len(x) > 0 else 1, raw=False
    )
    decay_asymmetry = positive_decay_rate - negative_decay_rate
    
    # Core Momentum = Asymmetric Decay × Volume-Adjusted Momentum
    core_momentum = decay_asymmetry * volume_adjusted_momentum
    
    # 5. Enhanced Momentum = Core × Breakout Acceleration × Regime Adaptation
    # Regime adaptation using volatility
    volatility_20 = returns_1d.rolling(window=20).std()
    regime_adaptation = 1 / (1 + volatility_20)
    
    enhanced_momentum = core_momentum * breakout_momentum * regime_adaptation
    
    # 6. Final Alpha construction
    # Decay-convergence alignment (agreement across timeframes)
    momentum_3d = returns_3d.rolling(window=5).mean()
    momentum_10d = returns_10d.rolling(window=5).mean()
    decay_convergence = np.sign(momentum_3d) * np.sign(momentum_10d) * np.abs(momentum_3d - momentum_10d)
    
    # Volume-price momentum coherence
    price_momentum = returns_5d
    volume_momentum = df['volume'].pct_change(5)
    momentum_coherence = np.corrcoef(price_momentum.rolling(window=10).mean(), 
                                   volume_momentum.rolling(window=10).mean())[0,1]
    
    # Final alpha combination
    final_alpha = enhanced_momentum * decay_convergence * (1 + momentum_coherence)
    
    # Clean and normalize
    final_alpha = final_alpha.replace([np.inf, -np.inf], np.nan)
    final_alpha = (final_alpha - final_alpha.rolling(window=20).mean()) / final_alpha.rolling(window=20).std()
    
    return final_alpha
