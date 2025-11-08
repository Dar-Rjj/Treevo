import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(data):
    """
    Dynamic Regime-Adaptive Momentum-Volume Convergence Alpha Factor
    
    Parameters:
    data: DataFrame with columns ['open', 'high', 'low', 'close', 'amount', 'volume']
    
    Returns:
    Series: Alpha factor values indexed by date
    """
    
    df = data.copy()
    
    # Helper function for EMA calculation
    def ema(series, window, alpha=0.3):
        return series.ewm(alpha=alpha, adjust=False).mean()
    
    # Signal Smoothing and Acceleration
    # Price Momentum Components
    df['price_momentum_5'] = ema(df['close'], 5) / ema(df['close'].shift(5), 5) - 1
    df['price_momentum_10'] = ema(df['close'], 10) / ema(df['close'].shift(10), 10) - 1
    df['price_momentum_20'] = ema(df['close'], 20) / ema(df['close'].shift(20), 20) - 1
    
    # Volume Momentum Components
    df['volume_momentum_5'] = ema(df['volume'], 5) / ema(df['volume'].shift(5), 5) - 1
    df['volume_momentum_10'] = ema(df['volume'], 10) / ema(df['volume'].shift(10), 10) - 1
    df['volume_momentum_20'] = ema(df['volume'], 20) / ema(df['volume'].shift(20), 20) - 1
    
    # Momentum Acceleration Calculation
    df['price_acceleration'] = df['price_momentum_5'] - df['price_momentum_10']
    df['volume_acceleration'] = df['volume_momentum_5'] - df['volume_momentum_10']
    
    # Dynamic Regime Detection
    # Amount-Based Participation Regime
    df['amount_acceleration'] = (df['amount'] / df['amount'].shift(5)) - (df['amount'].shift(5) / df['amount'].shift(10))
    df['high_participation'] = (df['amount_acceleration'] > 0).astype(int)
    
    # Volatility Regime Assessment
    df['price_range_vol'] = (df['high'] - df['low']) / df['close']
    df['volatility_trend'] = df['price_range_vol'] / df['price_range_vol'].rolling(window=5).mean()
    df['high_volatility'] = (df['volatility_trend'] > 1.2).astype(int)
    
    # Volume-Weighted Momentum Convergence
    # Momentum-Volume Divergence Signals
    df['divergence_5'] = df['price_momentum_5'] - df['volume_momentum_5']
    df['divergence_10'] = df['price_momentum_10'] - df['volume_momentum_10']
    df['divergence_20'] = df['price_momentum_20'] - df['volume_momentum_20']
    
    # Volume Confirmation Strength
    df['volume_acceleration_alignment'] = np.sign(df['price_acceleration']) == np.sign(df['volume_acceleration'])
    df['volume_persistence'] = (df['volume'] > df['volume'].shift(1)).rolling(window=3).sum()
    df['volume_breakout'] = (df['volume'] > df['volume'].rolling(window=5).mean()).astype(int)
    
    # Regime-Dependent Weighting
    df['volume_confirmation_weight'] = np.where(
        df['high_volatility'] == 1, 0.7, 0.3
    )
    df['momentum_persistence_weight'] = np.where(
        df['high_participation'] == 1, 0.2, 0.0
    )
    
    # Adaptive Factor Construction
    # Multi-timeframe Signal Combination
    df['weighted_divergence'] = (
        0.5 * df['divergence_5'] + 
        0.3 * df['divergence_10'] + 
        0.2 * df['divergence_20']
    )
    
    # Acceleration premium
    df['acceleration_premium'] = 0.3 * df['price_acceleration'] + 0.2 * df['volume_acceleration']
    
    # Volume confirmation multiplier
    df['volume_confirmation_multiplier'] = (
        df['volume_confirmation_weight'] * 
        (df['volume_acceleration_alignment'] * 0.4 + 
         df['volume_persistence'] * 0.3 + 
         df['volume_breakout'] * 0.3)
    )
    
    # Regime Transition Smoothing
    df['amount_accel_magnitude'] = df['amount_acceleration'].abs()
    df['regime_smooth_factor'] = df['amount_accel_magnitude'].rolling(window=5).mean()
    df['volatility_persistence'] = df['high_volatility'].rolling(window=3).mean()
    
    # Dynamic weight adjustment
    df['regime_stability'] = 1 - (df['high_volatility'].diff().abs().rolling(window=5).mean())
    df['dynamic_weight'] = 0.5 + 0.3 * df['regime_stability']
    
    # Cross-Sectional Enhancement
    df['relative_momentum'] = df['price_momentum_5'].rolling(window=20).apply(
        lambda x: (x.iloc[-1] - x.mean()) / x.std() if x.std() > 0 else 0
    )
    df['volume_intensity'] = df['volume_breakout'].rolling(window=10).sum()
    df['amount_participation'] = df['amount_acceleration'].rolling(window=10).mean()
    
    # Final Alpha Construction
    df['base_factor'] = (
        df['weighted_divergence'] * df['dynamic_weight'] +
        df['acceleration_premium'] * (1 - df['dynamic_weight'])
    )
    
    df['regime_enhanced_factor'] = (
        df['base_factor'] * 
        (1 + df['volume_confirmation_multiplier']) * 
        (1 + df['momentum_persistence_weight'])
    )
    
    df['cross_sectional_enhancement'] = (
        0.4 * df['relative_momentum'] +
        0.3 * df['volume_intensity'] +
        0.3 * df['amount_participation']
    )
    
    # Final Alpha Output
    alpha = (
        0.7 * df['regime_enhanced_factor'] +
        0.3 * df['cross_sectional_enhancement']
    )
    
    # Smooth the final output
    alpha_smoothed = alpha.rolling(window=3, min_periods=1).mean()
    
    return alpha_smoothed
