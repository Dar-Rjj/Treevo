import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(data):
    """
    Multi-Timeframe Momentum Acceleration with Regime-Adaptive Volume Confirmation
    """
    df = data.copy()
    
    # Price Momentum Calculation
    df['price_momentum_5d'] = df['close'] / df['close'].shift(5) - 1
    df['price_momentum_10d'] = df['close'] / df['close'].shift(10) - 1
    df['price_momentum_20d'] = df['close'] / df['close'].shift(20) - 1
    
    # Volume Analysis
    df['volume_momentum'] = df['volume'] / df['volume'].shift(5) - 1
    df['volume_price_divergence'] = df['volume_momentum'] - df['price_momentum_5d']
    df['volume_acceleration'] = df['volume_momentum'] - (df['volume'].shift(5) / df['volume'].shift(10) - 1)
    
    # Amount-Based Regime Detection
    df['amount_momentum'] = df['amount'] / df['amount'].shift(5) - 1
    df['amount_acceleration'] = df['amount_momentum'] - (df['amount'].shift(5) / df['amount'].shift(10) - 1)
    df['regime_intensity'] = np.abs(df['amount_acceleration']) * np.abs(df['amount_momentum'])
    
    # Exponential Smoothing (alpha=0.3)
    alpha = 0.3
    
    # Momentum Component Smoothing
    df['ema_price_momentum_5d'] = df['price_momentum_5d'].ewm(alpha=alpha, adjust=False).mean()
    df['ema_price_momentum_10d'] = df['price_momentum_10d'].ewm(alpha=alpha, adjust=False).mean()
    df['ema_price_momentum_20d'] = df['price_momentum_20d'].ewm(alpha=alpha, adjust=False).mean()
    
    # Volume Component Smoothing
    df['ema_volume_momentum'] = df['volume_momentum'].ewm(alpha=alpha, adjust=False).mean()
    df['ema_volume_price_divergence'] = df['volume_price_divergence'].ewm(alpha=alpha, adjust=False).mean()
    df['ema_volume_acceleration'] = df['volume_acceleration'].ewm(alpha=alpha, adjust=False).mean()
    
    # Regime Component Smoothing
    df['ema_amount_momentum'] = df['amount_momentum'].ewm(alpha=alpha, adjust=False).mean()
    df['ema_amount_acceleration'] = df['amount_acceleration'].ewm(alpha=alpha, adjust=False).mean()
    df['ema_regime_intensity'] = df['regime_intensity'].ewm(alpha=alpha, adjust=False).mean()
    
    # Momentum Acceleration Framework
    df['short_term_acceleration'] = df['ema_price_momentum_5d'] - df['ema_price_momentum_10d']
    df['medium_term_acceleration'] = df['ema_price_momentum_10d'] - df['ema_price_momentum_20d']
    df['cross_timeframe_acceleration'] = (df['ema_price_momentum_5d'] + df['ema_price_momentum_10d']) - 2 * df['ema_price_momentum_20d']
    
    # Acceleration Persistence
    df['acceleration_change_rate'] = df['cross_timeframe_acceleration'] - df['cross_timeframe_acceleration'].shift(1)
    df['momentum_quality'] = np.abs(df['cross_timeframe_acceleration']) * (1 - np.abs(df['acceleration_change_rate']))
    df['trend_strength'] = df['ema_price_momentum_5d'] * df['momentum_quality']
    
    # Volatility Normalization
    df['daily_range_volatility'] = (df['high'] - df['low']) / df['close']
    df['rolling_volatility_20d'] = df['close'].rolling(window=20).std()
    df['volatility_regime'] = df['daily_range_volatility'] / df['rolling_volatility_20d']
    
    # Volatility-Adjusted Components
    df['volatility_scaled_momentum'] = df['cross_timeframe_acceleration'] / df['rolling_volatility_20d']
    df['volatility_scaled_volume_divergence'] = df['ema_volume_price_divergence'] / df['rolling_volatility_20d']
    df['risk_adjusted_trend'] = df['trend_strength'] / df['rolling_volatility_20d']
    
    # Regime Classification
    df['high_participation_regime'] = (df['ema_amount_momentum'] > 0) & (df['ema_regime_intensity'] > 0)
    df['low_participation_regime'] = (df['ema_amount_momentum'] <= 0) | (df['ema_regime_intensity'] <= 0)
    df['regime_transition'] = np.abs(df['ema_amount_acceleration']) > df['ema_amount_acceleration'].rolling(window=20).std()
    
    # Cross-Sectional Ranking System
    def cross_sectional_rank(series):
        return (series.rank() - 1) / (len(series) - 1)
    
    # Calculate ranks for each component
    df['rank_momentum'] = df.groupby(df.index)['volatility_scaled_momentum'].transform(cross_sectional_rank)
    df['rank_volume_divergence'] = df.groupby(df.index)['volatility_scaled_volume_divergence'].transform(cross_sectional_rank)
    df['rank_trend'] = df.groupby(df.index)['risk_adjusted_trend'].transform(cross_sectional_rank)
    
    # Directional signals
    df['directional_signal_momentum'] = 2 * df['rank_momentum'] - 1
    df['directional_signal_volume'] = 2 * df['rank_volume_divergence'] - 1
    df['directional_signal_trend'] = 2 * df['rank_trend'] - 1
    
    # Regime-Adaptive Signal Components
    df['momentum_acceleration_component'] = df['volatility_scaled_momentum'] * df['directional_signal_momentum']
    df['volume_confirmation_component'] = df['volatility_scaled_volume_divergence'] * df['directional_signal_volume']
    df['regime_intensity_component'] = df['ema_regime_intensity'] * df['directional_signal_trend']
    
    # Adaptive Weighting with Transition Smoothing
    def get_adaptive_weights(row):
        if row['high_participation_regime']:
            return 0.4, 0.5, 0.1  # momentum, volume, regime
        elif row['low_participation_regime']:
            return 0.6, 0.3, 0.1  # momentum, volume, regime
        else:  # transition
            return 0.5, 0.4, 0.1  # balanced weights
    
    # Apply adaptive weights
    weights = df.apply(get_adaptive_weights, axis=1, result_type='expand')
    weights.columns = ['weight_momentum', 'weight_volume', 'weight_regime']
    df = pd.concat([df, weights], axis=1)
    
    # Blended Factor Value
    df['blended_value'] = (
        df['weight_momentum'] * df['momentum_acceleration_component'] +
        df['weight_volume'] * df['volume_confirmation_component'] +
        df['weight_regime'] * df['regime_intensity_component']
    )
    
    # Final volatility normalization and cross-sectional ranking
    df['final_factor'] = df['blended_value'] / df['rolling_volatility_20d']
    df['alpha_factor'] = df.groupby(df.index)['final_factor'].transform(cross_sectional_rank)
    
    return df['alpha_factor']
