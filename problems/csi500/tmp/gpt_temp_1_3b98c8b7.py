import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Initialize result series
    result = pd.Series(index=df.index, dtype=float)
    
    # Calculate basic components
    df['high_low_range'] = df['high'] - df['low']
    df['close_change'] = df['close'].diff()
    
    # Multi-Scale Volatility Regime Classification
    # Short-Term Volatility Dynamics
    df['vol_momentum_3d'] = df['high_low_range'] / df['high_low_range'].shift(3) - 1
    df['vol_persistence_intra'] = df['high_low_range'] / df['high_low_range'].shift(1) - 1
    df['vol_acceleration'] = df['vol_momentum_3d'] - df['vol_persistence_intra']
    
    # Medium-Term Volatility Structure
    df['vol_ratio_10d'] = df['high_low_range'] / df['high_low_range'].rolling(window=10).mean()
    df['vol_fractal_dim'] = np.log(df['high_low_range']) / np.log(df['high_low_range'].rolling(window=5).mean())
    df['vol_regime_strength'] = df['vol_ratio_10d'] * df['vol_fractal_dim']
    
    # Volatility Regime Classification
    conditions = [
        (df['vol_acceleration'] > 0.2) & (df['vol_regime_strength'] > 1.2),
        (df['vol_acceleration'] < -0.1) & (df['vol_regime_strength'] < 0.8)
    ]
    choices = ['high', 'low']
    df['vol_regime'] = np.select(conditions, choices, default='transition')
    
    # Price-Volume Fractal Coherence Analysis
    # Multi-Timeframe Price Fractals
    df['price_fractal_3d'] = (df['close'] - df['close'].shift(3)) / df['high_low_range'].shift(3)
    df['price_fractal_5d'] = (df['close'] - df['close'].shift(5)) / df['high_low_range'].rolling(window=5).mean()
    df['price_fractal_momentum'] = df['price_fractal_3d'] / df['price_fractal_5d'] - 1
    
    # Volume Fractal Dynamics
    df['volume_fractal_ratio'] = df['volume'] / df['volume'].rolling(window=5).mean()
    df['volume_momentum_fractal'] = df['volume'] / df['volume'].shift(3) - 1
    df['volume_accel_fractal'] = (df['volume'] / df['volume'].shift(1) - 1) - (df['volume'].shift(1) / df['volume'].shift(2) - 1)
    
    # Price-Volume Coherence Signals
    df['fractal_coherence'] = np.sign(df['price_fractal_momentum']) * np.sign(df['volume_momentum_fractal'])
    df['coherence_strength'] = np.abs(df['price_fractal_momentum']) * np.abs(df['volume_momentum_fractal'])
    df['acceleration_alignment'] = np.sign(df['vol_acceleration']) * np.sign(df['volume_accel_fractal'])
    
    # Amount-Volume Divergence Detection
    # Amount Fractal Patterns
    df['amount_momentum_div'] = (df['amount'] / df['amount'].shift(1) - 1) - (df['volume'] / df['volume'].shift(1) - 1)
    df['amount_volume_ratio_fractal'] = df['amount'] / df['volume']
    
    # Calculate 3-day amount persistence
    amount_sign_change = np.sign(df['amount'] / df['amount'].shift(1) - 1)
    df['amount_persistence_3d'] = (
        (amount_sign_change == amount_sign_change.shift(1)) & 
        (amount_sign_change == amount_sign_change.shift(2))
    ).astype(int)
    
    # Volume Efficiency Analysis
    df['volume_price_efficiency'] = np.abs(df['close_change']) / df['volume']
    df['amount_price_efficiency'] = np.abs(df['close_change']) / df['amount']
    df['efficiency_divergence'] = df['volume_price_efficiency'] - df['amount_price_efficiency']
    
    # Divergence Confirmation Signals
    df['strong_divergence'] = df['amount_momentum_div'] * df['efficiency_divergence']
    df['persistence_weighted_div'] = df['strong_divergence'] * (1 + 0.15 * df['amount_persistence_3d'])
    df['volume_confirmed_div'] = df['persistence_weighted_div'] * df['volume_fractal_ratio']
    
    # Regime-Adaptive Signal Synthesis
    # Base Coherence-Divergence Signal
    df['coherence_enhanced_signal'] = df['fractal_coherence'] * df['coherence_strength']
    df['divergence_adjusted_signal'] = df['coherence_enhanced_signal'] * df['volume_confirmed_div']
    df['acceleration_aligned_signal'] = df['divergence_adjusted_signal'] * df['acceleration_alignment']
    
    # Volatility Regime Weighting
    conditions_weight = [
        df['vol_regime'] == 'high',
        df['vol_regime'] == 'low'
    ]
    choices_weight = [
        0.8 * df['acceleration_aligned_signal'] + 0.2 * df['amount_volume_ratio_fractal'],
        0.3 * df['acceleration_aligned_signal'] + 0.7 * df['amount_volume_ratio_fractal']
    ]
    df['weighted_signal'] = np.select(conditions_weight, choices_weight, 
                                     default=0.5 * df['acceleration_aligned_signal'] + 0.5 * df['amount_volume_ratio_fractal'])
    
    # Fractal Persistence Adjustment
    # Calculate coherence persistence
    coherence_sign = np.sign(df['fractal_coherence'])
    df['coherence_persistence'] = (
        (coherence_sign == coherence_sign.shift(1)) & 
        (coherence_sign == coherence_sign.shift(2))
    ).astype(int)
    
    # Calculate regime persistence
    regime_persistence = []
    for i in range(len(df)):
        if i < 2:
            regime_persistence.append(0)
        else:
            current_regime = df['vol_regime'].iloc[i]
            count = 0
            for j in range(1, 3):
                if df['vol_regime'].iloc[i-j] == current_regime:
                    count += 1
            regime_persistence.append(count)
    df['regime_persistence'] = regime_persistence
    
    df['persistence_weighted_signal'] = df['weighted_signal'] * (1 + 0.1 * (df['coherence_persistence'] + df['regime_persistence']))
    
    # Dynamic Multi-Scale Smoothing
    # Determine smoothing windows based on regime and persistence
    base_windows = {
        'high': 2,
        'low': 5,
        'transition': 3
    }
    
    df['smoothing_window'] = df['vol_regime'].map(base_windows)
    
    # Adjust windows based on regime persistence
    df.loc[df['regime_persistence'] >= 2, 'smoothing_window'] = df['smoothing_window'] - 1
    df.loc[df['regime_persistence'] == 0, 'smoothing_window'] = df['smoothing_window'] + 1
    
    # Apply smoothing
    df['smoothed_signal'] = np.nan
    for i in range(len(df)):
        window = int(df['smoothing_window'].iloc[i])
        if i >= window - 1:
            df['smoothed_signal'].iloc[i] = df['persistence_weighted_signal'].iloc[i-window+1:i+1].mean()
    
    # Final Alpha Construction
    # Multi-Dimensional Volatility Scaling
    df['price_vol_scale'] = np.sqrt((df['high_low_range'] ** 2).rolling(window=15).mean())
    
    # Calculate volume volatility
    volume_vol = (df['volume'] / df['volume'].shift(1) - 1) ** 2
    df['volume_vol_scale'] = np.sqrt(volume_vol.rolling(window=15).mean())
    
    df['combined_vol_scale'] = df['price_vol_scale'] * df['volume_vol_scale']
    
    # Volatility Regime Adaptive Alpha
    df['scaled_signal'] = df['smoothed_signal'] / df['combined_vol_scale']
    df['final_alpha'] = df['scaled_signal'] * df['volume_fractal_ratio']
    
    return df['final_alpha']
