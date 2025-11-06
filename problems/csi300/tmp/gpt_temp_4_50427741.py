import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Initialize result series
    result = pd.Series(index=df.index, dtype=float)
    
    # Pre-calculate necessary columns
    df = df.copy()
    
    # Basic price changes and ratios
    df['prev_close'] = df['close'].shift(1)
    df['gap_fracture'] = (df['open'] - df['prev_close']) / df['prev_close']
    df['intraday_fracture'] = (df['high'] - df['low']) / df['prev_close']
    df['close_fracture'] = abs(df['close'] - df['prev_close']) / df['prev_close']
    
    # Volume-Price Entropy
    df['avg_price'] = (df['high'] + df['low'] + df['close']) / 3
    df['price_dev_sq'] = (df['close'] - df['avg_price']) ** 2
    df['volume_price_entropy'] = -(df['price_dev_sq'] * df['volume']) / df['volume']
    
    # Pressure Differential (5-day window)
    pressure_up = pd.Series(index=df.index, dtype=float)
    pressure_down = pd.Series(index=df.index, dtype=float)
    
    for i in range(len(df)):
        if i < 5:
            pressure_up.iloc[i] = 1.0
            pressure_down.iloc[i] = 1.0
            continue
            
        up_vol = 0
        down_vol = 0
        for j in range(5):
            if df['close'].iloc[i-j] > df['close'].iloc[i-j-1]:
                up_vol += df['volume'].iloc[i-j]
            elif df['close'].iloc[i-j] < df['close'].iloc[i-j-1]:
                down_vol += df['volume'].iloc[i-j]
        
        pressure_up.iloc[i] = up_vol if up_vol > 0 else 1.0
        pressure_down.iloc[i] = down_vol if down_vol > 0 else 1.0
    
    df['pressure_differential'] = pressure_up / pressure_down
    
    # Simplified Fractal Dimension Change (using rolling variance ratios)
    df['hurst_short'] = df['close'].rolling(window=10).std() / df['close'].rolling(window=5).std()
    df['hurst_medium'] = df['close'].rolling(window=20).std() / df['close'].rolling(window=10).std()
    df['fractal_dimension_change'] = df['hurst_medium'] - df['hurst_short']
    
    # Fracture-Entropy Integration
    df['gap_entropy_alignment'] = df['gap_fracture'] * df['volume_price_entropy']
    df['intraday_pressure_coherence'] = df['intraday_fracture'] * df['pressure_differential']
    df['fracture_entropy_cascade'] = df['close_fracture'] * df['fractal_dimension_change']
    
    # Volume Timing Components
    df['prev_volume'] = df['volume'].shift(1)
    df['volume_spike'] = df['volume'] / df['prev_volume']
    df['volume_4d_avg'] = df['volume'].rolling(window=5).mean()
    df['volume_consistency'] = df['volume'] / df['volume_4d_avg']
    df['volume_timing_score'] = df['volume_spike'] * df['volume_consistency']
    
    # Range Efficiency Components
    df['intraday_range_efficiency'] = abs(df['close'] - df['open']) / (df['high'] - df['low']).replace(0, 1e-10)
    true_range = np.maximum(df['high'] - df['low'], 
                           np.maximum(abs(df['high'] - df['prev_close']), 
                                     abs(df['low'] - df['prev_close'])))
    df['true_range_efficiency'] = (df['high'] - df['low']) / true_range.replace(0, 1e-10)
    df['volume_efficiency_score'] = df['volume_timing_score'] * df['true_range_efficiency']
    
    # Volume-Entropy Synchronization
    df['volume_entropy_alignment'] = df['volume_timing_score'] * df['volume_price_entropy']
    df['volume_pressure_coherence'] = df['volume_efficiency_score'] * df['pressure_differential']
    df['volume_fracture_efficiency'] = df['volume_entropy_alignment'] * df['volume_pressure_coherence']
    
    # Multi-Frequency Persistence
    df['close_t_minus_1'] = df['close'].shift(1)
    df['close_t_minus_2'] = df['close'].shift(2)
    df['close_t_minus_5'] = df['close'].shift(5)
    df['close_t_minus_10'] = df['close'].shift(10)
    
    df['short_term_persistence'] = (df['close'] - df['close_t_minus_1']) * (df['close_t_minus_1'] - df['close_t_minus_2'])
    df['medium_term_persistence'] = (df['close'] - df['close_t_minus_5']) * (df['close_t_minus_5'] - df['close_t_minus_10'])
    df['persistence_ratio'] = df['short_term_persistence'] / df['medium_term_persistence'].replace(0, 1e-10)
    
    # Entropy-Enhanced Momentum
    df['entropy_momentum_transmission'] = df['volume_price_entropy'] * df['short_term_persistence']
    df['pressure_momentum_amplification'] = df['pressure_differential'] * df['medium_term_persistence']
    df['fractal_momentum_cascade'] = df['fractal_dimension_change'] * df['persistence_ratio']
    
    # Fracture-Enhanced Persistence
    df['gap_persistence_interaction'] = df['gap_fracture'] * df['short_term_persistence']
    df['fracture_entropy_persistence'] = df['close_fracture'] * df['volume_price_entropy']
    
    # Volatility-Pressure Regime
    df['high_low_14d_avg'] = (df['high'] - df['low']).rolling(window=14).mean()
    df['high_volatility_detection'] = (df['high'] - df['low']) / df['high_low_14d_avg'].replace(0, 1e-10)
    df['pressure_regime'] = (df['pressure_differential'] > 1.2).astype(float)
    df['volatility_pressure_alignment'] = df['high_volatility_detection'] * df['pressure_regime'] * df['volume_efficiency_score']
    
    # Entropy-Compression Regime
    df['volume_price_entropy_4d_avg'] = df['volume_price_entropy'].rolling(window=5).mean()
    df['low_entropy'] = (df['volume_price_entropy'] < df['volume_price_entropy_4d_avg']).astype(float)
    df['compression_detection'] = ((df['high'] - df['low']) / df['prev_close'] < 0.02).astype(float)
    df['entropy_compression_breakout'] = df['low_entropy'] * df['compression_detection'] * df['volume_spike']
    
    # Regime Synchronization
    df['regime_weighted_fracture'] = df['fracture_entropy_cascade'] * df['volatility_pressure_alignment']
    df['volume_entropy_alignment_regime'] = df['volume_price_entropy'] * df['volume_efficiency_score']
    df['regime_synchronization'] = df['regime_weighted_fracture'] * df['volume_entropy_alignment_regime']
    
    # Core Signal Integration
    df['primary_signal'] = df['fracture_entropy_cascade'] * df['volume_fracture_efficiency']
    df['secondary_signal'] = df['pressure_momentum_amplification'] * df['volume_efficiency_score']
    df['tertiary_signal'] = df['regime_synchronization'] * df['fractal_momentum_cascade']
    
    # Hierarchical Confirmation
    df['micro_confirmation'] = df['gap_persistence_interaction'] * df['volume_spike']
    df['meso_confirmation'] = df['intraday_pressure_coherence'] * df['volume_consistency']
    df['macro_confirmation'] = df['fracture_entropy_persistence'] * df['volume_efficiency_score']
    
    # Multi-Timeframe Synthesis
    df['signal_persistence'] = df['primary_signal'].rolling(window=3).mean()
    df['confirmation_strength'] = df['secondary_signal'].rolling(window=2).mean()
    
    # Final Alpha
    df['final_alpha'] = df['signal_persistence'] * df['confirmation_strength'] * df['tertiary_signal']
    
    # Fill NaN values and return
    result = df['final_alpha'].fillna(0)
    return result
