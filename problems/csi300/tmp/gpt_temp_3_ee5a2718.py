import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Cross-Asset Momentum Diffusion with Volume Signature Analysis
    Generates alpha factor combining momentum propagation patterns with volume confirmation signals
    """
    
    # Initialize result series
    result = pd.Series(index=df.index, dtype=float)
    
    # Required columns
    required_cols = ['open', 'high', 'low', 'close', 'amount', 'volume']
    if not all(col in df.columns for col in required_cols):
        return result
    
    # Calculate momentum signals
    close = df['close']
    
    # Short-term momentum (5-day)
    momentum_5d = (close - close.shift(5)) / close.shift(5)
    
    # Medium-term momentum (10-day) for timeframe convergence
    momentum_10d = (close - close.shift(10)) / close.shift(10)
    
    # Volume analysis
    volume = df['volume']
    amount = df['amount']
    
    # Volume momentum and acceleration
    volume_momentum = (volume - volume.shift(1)) / volume.shift(1)
    
    # Volume concentration (large trade dominance)
    volume_concentration = amount / volume
    
    # Normalize volume concentration using rolling statistics
    vol_conc_ma = volume_concentration.rolling(window=20, min_periods=10).mean()
    vol_conc_std = volume_concentration.rolling(window=20, min_periods=10).std()
    normalized_vol_conc = (volume_concentration - vol_conc_ma) / vol_conc_std
    
    # Volume regime detection
    volume_ma_20 = volume.rolling(window=20, min_periods=10).mean()
    volume_std_20 = volume.rolling(window=20, min_periods=10).std()
    volume_zscore = (volume - volume_ma_20) / volume_std_20
    
    # High volume regime indicator
    high_volume_regime = (volume_zscore > 1).astype(float)
    low_volume_regime = (volume_zscore < -0.5).astype(float)
    
    # Momentum strength and persistence
    momentum_strength = momentum_5d.rolling(window=5, min_periods=3).std()
    momentum_persistence = momentum_5d.rolling(window=5, min_periods=3).apply(
        lambda x: np.corrcoef(range(len(x)), x)[0,1] if len(x) > 1 and not np.isnan(x).any() else 0
    )
    
    # Cross-timeframe momentum convergence
    timeframe_convergence = momentum_5d * momentum_10d
    
    # Volume-confirmed momentum
    volume_confirmed_momentum = momentum_5d * np.sign(volume_momentum)
    
    # Volume divergence detection
    price_volume_divergence = np.sign(momentum_5d) != np.sign(volume_momentum)
    
    # Construct composite factors
    for i in range(len(df)):
        if i < 20:  # Ensure sufficient history
            result.iloc[i] = 0
            continue
            
        # Momentum-Volume Interaction Components
        current_momentum = momentum_5d.iloc[i]
        current_volume_momentum = volume_momentum.iloc[i]
        current_vol_conc = normalized_vol_conc.iloc[i]
        
        # High volume, strong momentum: trend acceleration
        if high_volume_regime.iloc[i] and abs(current_momentum) > 0.02:
            volume_amplified_momentum = current_momentum * (1 + min(current_volume_momentum, 2.0))
        # High volume, weak momentum: exhaustion signals
        elif high_volume_regime.iloc[i] and abs(current_momentum) < 0.005:
            volume_amplified_momentum = -current_momentum * abs(current_volume_momentum)
        # Low volume, strong momentum: fragile trends
        elif low_volume_regime.iloc[i] and abs(current_momentum) > 0.02:
            volume_amplified_momentum = current_momentum * 0.5
        # Low volume, weak momentum: consolidation phases
        elif low_volume_regime.iloc[i] and abs(current_momentum) < 0.005:
            volume_amplified_momentum = 0
        else:
            volume_amplified_momentum = current_momentum
        
        # Volume concentration effect
        if not np.isnan(current_vol_conc):
            if abs(current_vol_conc) > 1:  # Significant concentration
                concentration_multiplier = 1 + (current_vol_conc * 0.1)
                volume_amplified_momentum *= concentration_multiplier
        
        # Timeframe convergence component
        current_timeframe_conv = timeframe_convergence.iloc[i] if not np.isnan(timeframe_convergence.iloc[i]) else 0
        
        # Momentum persistence adjustment
        current_persistence = momentum_persistence.iloc[i] if not np.isnan(momentum_persistence.iloc[i]) else 0
        persistence_multiplier = 1 + (current_persistence * 0.5)
        
        # Price-volume divergence penalty
        if price_volume_divergence.iloc[i]:
            divergence_penalty = 0.7
        else:
            divergence_penalty = 1.0
        
        # Composite alpha factor
        composite_alpha = (
            volume_amplified_momentum * 0.4 +
            current_timeframe_conv * 0.3 +
            volume_confirmed_momentum.iloc[i] * 0.2 +
            current_momentum * momentum_strength.iloc[i] * 0.1
        ) * persistence_multiplier * divergence_penalty
        
        result.iloc[i] = composite_alpha
    
    # Final normalization
    result_ma = result.rolling(window=20, min_periods=10).mean()
    result_std = result.rolling(window=20, min_periods=10).std()
    normalized_result = (result - result_ma) / result_std
    
    return normalized_result.fillna(0)
