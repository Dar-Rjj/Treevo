import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(data):
    """
    Multi-Timeframe Acceleration with Volume-Price Convergence Alpha Factor
    """
    df = data.copy()
    
    # Multi-Timeframe Momentum Calculation
    df['momentum_1d'] = (df['close'] - df['close'].shift(1)) / df['close'].shift(1)
    df['momentum_3d'] = (df['close'] - df['close'].shift(3)) / df['close'].shift(3)
    df['momentum_5d'] = (df['close'] - df['close'].shift(5)) / df['close'].shift(5)
    df['momentum_10d'] = (df['close'] - df['close'].shift(10)) / df['close'].shift(10)
    
    # Acceleration Gradient Analysis
    df['primary_acceleration'] = df['momentum_3d'] - df['momentum_1d']
    df['secondary_acceleration'] = df['momentum_5d'] - df['momentum_3d']
    df['tertiary_acceleration'] = df['momentum_10d'] - df['momentum_5d']
    
    # Acceleration Persistence Scoring
    def calculate_persistence(series):
        persistence = pd.Series(index=series.index, dtype=float)
        current_streak = 0
        for i in range(len(series)):
            if i == 0 or np.sign(series.iloc[i]) == np.sign(series.iloc[i-1]):
                current_streak += 1
            else:
                current_streak = 1
            persistence.iloc[i] = sum(0.85 ** j for j in range(current_streak))
        return persistence
    
    df['acceleration_persistence'] = calculate_persistence(df['primary_acceleration'])
    
    # Multi-Timeframe Volume Analysis
    df['volume_momentum_3d'] = (df['volume'] - df['volume'].shift(3)) / (df['volume'].shift(3) + 1e-8)
    df['volume_acceleration'] = (df['volume'] / (df['volume'].shift(1) + 1e-8)) - \
                               (df['volume'].shift(1) / (df['volume'].shift(2) + 1e-8))
    
    # Volume Persistence
    def volume_persistence_calc(volume_series):
        persistence = pd.Series(index=volume_series.index, dtype=float)
        current_streak = 0
        for i in range(len(volume_series)):
            if i == 0 or volume_series.iloc[i] > volume_series.iloc[i-1]:
                current_streak += 1
            else:
                current_streak = 1
            persistence.iloc[i] = current_streak
        return persistence
    
    df['volume_persistence'] = volume_persistence_calc(df['volume'])
    df['volume_trend_strength'] = df['volume_momentum_3d'] * df['volume_persistence']
    
    # Price-Volume Convergence Scoring
    df['direction_convergence'] = np.sign(df['momentum_3d']) * np.sign(df['volume_momentum_3d'])
    df['acceleration_convergence'] = np.sign(df['primary_acceleration']) * np.sign(df['volume_acceleration'])
    df['strength_convergence'] = np.minimum(np.abs(df['momentum_3d']), np.abs(df['volume_momentum_3d']))
    
    # Combined convergence score
    df['convergence_score'] = df['direction_convergence'] * df['acceleration_convergence'] * df['strength_convergence']
    
    # Convergence Confidence Assessment
    def convergence_confidence(row):
        direction_match = row['direction_convergence'] > 0
        acceleration_match = row['acceleration_convergence'] > 0
        strength_threshold = np.abs(row['strength_convergence']) > np.percentile(np.abs(df['strength_convergence'].dropna()), 60)
        
        if direction_match and acceleration_match and strength_threshold:
            return 1.2  # High confidence
        elif (direction_match and acceleration_match) or (direction_match and strength_threshold):
            return 0.8  # Medium confidence
        else:
            return 0.4  # Low confidence
    
    df['convergence_confidence'] = df.apply(convergence_confidence, axis=1)
    
    # Multi-Dimensional Volatility Measurement
    df['daily_range_volatility'] = (df['high'] - df['low']) / (df['close'] + 1e-8)
    df['close_to_close_volatility'] = np.abs(df['close'] - df['close'].shift(1)) / (df['close'].shift(1) + 1e-8)
    
    # Volatility Persistence (5-day window standard deviation)
    df['volatility_persistence'] = df['daily_range_volatility'].rolling(window=5).std()
    df['volatility_stability_score'] = 1 / (df['volatility_persistence'] + 0.0001)
    
    # Volatility-Adaptive Scaling
    df['volatility_scaling_factor'] = 1 / (df['daily_range_volatility'] + 0.0001)
    
    # Stability-Weighted Confidence
    df['stability_confidence'] = df['volatility_stability_score'] * 0.6
    
    # Multi-Factor Integration Framework
    # Core Acceleration Component
    df['primary_acceleration_signal'] = df['primary_acceleration'] * df['acceleration_persistence']
    df['multi_timeframe_validation'] = df['secondary_acceleration'] * df['tertiary_acceleration']
    df['enhanced_acceleration'] = df['primary_acceleration_signal'] * (1 + df['multi_timeframe_validation'])
    
    # Convergence-Enhanced Signal
    df['convergence_aligned_signal'] = df['enhanced_acceleration'] * df['convergence_confidence']
    
    # Volatility-Adapted Final Factor
    df['final_alpha'] = df['convergence_aligned_signal'] * df['volatility_scaling_factor']
    df['final_alpha'] = df['final_alpha'] * df['stability_confidence']
    
    # Clean up and return
    alpha_factor = df['final_alpha'].replace([np.inf, -np.inf], np.nan).fillna(0)
    
    return alpha_factor
