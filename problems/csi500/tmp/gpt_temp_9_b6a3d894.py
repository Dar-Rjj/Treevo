import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Regime-Adaptive Multi-Timeframe Momentum Divergence Alpha Factor
    """
    # Initialize result series
    result = pd.Series(index=df.index, dtype=float)
    
    # Calculate price momentum components
    df['price_mom_2d'] = df['close'] / df['close'].shift(2) - 1
    df['price_mom_5d'] = df['close'] / df['close'].shift(5) - 1
    df['price_mom_10d'] = df['close'] / df['close'].shift(10) - 1
    df['price_mom_20d'] = df['close'] / df['close'].shift(20) - 1
    
    # Calculate volume momentum components
    df['volume_mom_2d'] = df['volume'] / df['volume'].shift(2) - 1
    df['volume_mom_5d'] = df['volume'] / df['volume'].shift(5) - 1
    df['volume_mom_10d'] = df['volume'] / df['volume'].shift(10) - 1
    df['volume_mom_20d'] = df['volume'] / df['volume'].shift(20) - 1
    
    # Calculate regime detection components
    # Market activity regime
    df['amount_rolling_mean'] = df['amount'].rolling(window=10).mean()
    df['volume_rolling_mean'] = df['volume'].rolling(window=10).mean()
    df['amount_intensity'] = df['amount'] / df['amount_rolling_mean']
    df['volume_intensity'] = df['volume'] / df['volume_rolling_mean']
    
    # Price volatility regime
    df['prev_close'] = df['close'].shift(1)
    df['true_range'] = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            abs(df['high'] - df['prev_close']),
            abs(df['low'] - df['prev_close'])
        )
    )
    df['range_rolling_mean'] = df['true_range'].rolling(window=10).mean()
    df['range_intensity'] = df['true_range'] / df['range_rolling_mean']
    
    # Calculate divergence signals
    df['divergence_2d'] = df['price_mom_2d'] - df['volume_mom_2d']
    df['divergence_5d'] = df['price_mom_5d'] - df['volume_mom_5d']
    df['divergence_10d'] = df['price_mom_10d'] - df['volume_mom_10d']
    df['divergence_20d'] = df['price_mom_20d'] - df['volume_mom_20d']
    
    # Calculate divergence quality metrics
    divergences = ['divergence_2d', 'divergence_5d', 'divergence_10d', 'divergence_20d']
    df['direction_consistency'] = df[divergences].gt(0).sum(axis=1)
    
    # Calculate magnitude progression (avoiding division by zero)
    df['magnitude_progression'] = np.where(
        (df['divergence_2d'] != 0) & (df['divergence_5d'] != 0),
        df['divergence_5d'] / df['divergence_2d'],
        0
    )
    
    # Volume confirmation
    volume_moments = ['volume_mom_2d', 'volume_mom_5d', 'volume_mom_10d', 'volume_mom_20d']
    df['volume_confirmation'] = np.sign(df[volume_moments].mean(axis=1)) * df[volume_moments].abs().mean(axis=1)
    
    # Price momentum strength
    price_moments = ['price_mom_2d', 'price_mom_5d', 'price_mom_10d', 'price_mom_20d']
    df['price_momentum_strength'] = df[price_moments].abs().mean(axis=1)
    
    # Process each row to calculate regime-adaptive weights
    for idx in df.index:
        if pd.isna(df.loc[idx, 'divergence_20d']):
            continue
            
        # Determine activity regime
        amount_intensity = df.loc[idx, 'amount_intensity']
        volume_intensity = df.loc[idx, 'volume_intensity']
        
        if amount_intensity > 1.2 and volume_intensity > 1.1:
            activity_weights = [0.4, 0.3, 0.2, 0.1]  # High activity
        elif amount_intensity >= 0.8 and amount_intensity <= 1.2 and volume_intensity >= 0.9 and volume_intensity <= 1.1:
            activity_weights = [0.3, 0.3, 0.2, 0.2]  # Normal activity
        elif amount_intensity < 0.8 and volume_intensity < 0.9:
            activity_weights = [0.2, 0.3, 0.3, 0.2]  # Low activity
        else:
            activity_weights = [0.25, 0.3, 0.25, 0.2]  # Default
        
        # Determine volatility regime
        range_intensity = df.loc[idx, 'range_intensity']
        
        if range_intensity > 1.3:
            volatility_weights = [0.3, 0.3, 0.2, 0.2]  # High volatility
        elif range_intensity >= 0.7 and range_intensity <= 1.3:
            volatility_weights = [0.25, 0.3, 0.25, 0.2]  # Normal volatility
        else:
            volatility_weights = [0.2, 0.25, 0.3, 0.25]  # Low volatility
        
        # Combine weights
        final_weights = [(a + v) / 2 for a, v in zip(activity_weights, volatility_weights)]
        
        # Calculate weighted divergence score
        divergence_values = [
            df.loc[idx, 'divergence_2d'],
            df.loc[idx, 'divergence_5d'],
            df.loc[idx, 'divergence_10d'],
            df.loc[idx, 'divergence_20d']
        ]
        
        base_divergence = sum(w * d for w, d in zip(final_weights, divergence_values))
        
        # Quality multiplier
        quality_multiplier = df.loc[idx, 'direction_consistency'] / 4
        
        # Volume adjustment
        volume_adjustment = df.loc[idx, 'volume_confirmation'] * 0.3
        
        # Regime intensity and signal amplification
        regime_intensity = abs(amount_intensity - 1) + abs(range_intensity - 1)
        signal_amplification = 1 + (regime_intensity * 0.15)
        
        # Final alpha calculation
        final_alpha = base_divergence * quality_multiplier * signal_amplification + volume_adjustment
        
        result.loc[idx] = final_alpha
    
    return result
