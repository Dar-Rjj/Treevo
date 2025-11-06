import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(data):
    """
    Momentum Persistence Breakout Factor
    Combines multi-timeframe momentum analysis, breakout detection, and volume confirmation
    to generate regime-aware trading signals.
    """
    df = data.copy()
    
    # Multi-Timeframe Momentum Analysis
    # Short-term Momentum (3-day)
    df['short_price_change'] = (df['close'] - df['close'].shift(3)) / df['close'].shift(3)
    df['volume_trend'] = df['volume'] / df['volume'].rolling(window=3).mean()
    
    # Medium-term Momentum (10-day)
    df['medium_price_change'] = (df['close'] - df['close'].shift(10)) / df['close'].shift(10)
    high_10 = df['high'].rolling(window=10).max()
    low_10 = df['low'].rolling(window=10).min()
    df['volatility_adjustment'] = df['medium_price_change'] / (high_10 - low_10)
    
    # Momentum Alignment
    df['momentum_alignment'] = np.sign(df['short_price_change']) == np.sign(df['medium_price_change'])
    df['strength_ratio'] = np.abs(df['short_price_change']) / np.abs(df['medium_price_change'])
    df['strength_ratio'] = df['strength_ratio'].replace([np.inf, -np.inf], 0).fillna(0)
    
    # Breakout Detection with Adaptive Thresholds
    # Volatility-Regime Breakout Bands
    df['recent_volatility'] = df['high'].rolling(window=5).max() - df['low'].rolling(window=5).min()
    df['historical_volatility'] = df['high'].rolling(window=20).max() - df['low'].rolling(window=20).min()
    df['volatility_ratio'] = df['recent_volatility'] / df['historical_volatility']
    df['volatility_ratio'] = df['volatility_ratio'].replace([np.inf, -np.inf], 1).fillna(1)
    
    # Adaptive Breakout Levels
    upper_breakout_level = df['high'].rolling(window=10).max() * (1 + df['volatility_ratio'] * 0.1)
    lower_breakout_level = df['low'].rolling(window=10).min() * (1 - df['volatility_ratio'] * 0.1)
    
    df['breakout_status'] = 0
    df.loc[df['high'] > upper_breakout_level, 'breakout_status'] = 1
    df.loc[df['low'] < lower_breakout_level, 'breakout_status'] = -1
    
    # Breakout Quality Assessment
    df['breakout_magnitude'] = 0
    df.loc[df['breakout_status'] == 1, 'breakout_magnitude'] = (
        (df['high'] - upper_breakout_level) / upper_breakout_level
    )
    df.loc[df['breakout_status'] == -1, 'breakout_magnitude'] = (
        (lower_breakout_level - df['low']) / lower_breakout_level
    )
    
    # Breakout Duration
    df['breakout_duration'] = df['breakout_status'].rolling(window=3).apply(
        lambda x: (x != 0).sum(), raw=True
    )
    
    # Volume-Confirmed Momentum Persistence
    # Volume Confirmation Signals
    df['volume_surge'] = df['volume'] > (df['volume'].rolling(window=10).mean() * 1.8)
    df['volume_persistence'] = df['volume'].rolling(window=3).apply(
        lambda x: (x > df['volume'].rolling(window=5).mean().iloc[-1]).sum() / 3, raw=False
    )
    df['volume_momentum_correlation'] = (
        np.sign(df['close'] - df['close'].shift(1)) == 
        np.sign(df['volume'] - df['volume'].shift(1))
    )
    
    # Momentum Persistence Metrics
    df['intraday_momentum_persistence'] = (
        np.sign(df['close'] - df['open']) == 
        np.sign(df['open'] - df['close'].shift(1))
    )
    
    # Multi-day Momentum Streak
    def momentum_streak(x):
        if len(x) < 3:
            return 0
        streak = 0
        for i in range(1, len(x)):
            if np.sign(x[i] - x[i-1]) == np.sign(x[i-1] - x[i-2]):
                streak += 1
            else:
                break
        return streak
    
    df['multi_day_momentum_streak'] = df['close'].rolling(window=5).apply(
        momentum_streak, raw=True
    )
    
    # Momentum Acceleration
    df['momentum_acceleration'] = (
        (df['close'] - df['close'].shift(3)) - 
        (df['close'].shift(3) - df['close'].shift(6))
    )
    
    # Regime-Aware Factor Combination
    # Volatility Regime Classification
    df['volatility_regime'] = 'normal'
    df.loc[df['recent_volatility'] > df['historical_volatility'] * 1.2, 'volatility_regime'] = 'high'
    df.loc[df['recent_volatility'] < df['historical_volatility'] * 0.8, 'volatility_regime'] = 'low'
    
    # Regime-Adaptive Weights
    df['regime_weight'] = 0.5
    df.loc[df['volatility_regime'] == 'high', 'regime_weight'] = 0.3
    df.loc[df['volatility_regime'] == 'low', 'regime_weight'] = 0.7
    
    # Factor Components
    df['momentum_component'] = (
        df['momentum_alignment'].astype(float) * 
        df['momentum_acceleration'] * 
        df['multi_day_momentum_streak']
    )
    
    df['breakout_component'] = (
        df['breakout_status'] * 
        df['breakout_magnitude'] * 
        df['breakout_duration']
    )
    
    df['volume_component'] = (
        df['volume_surge'].astype(float) * 
        df['volume_persistence'] * 
        df['volume_momentum_correlation'].astype(float)
    )
    
    # Final Factor Generation
    # Raw Factor Combination
    df['base_factor'] = (
        df['momentum_component'] * 
        df['breakout_component'] * 
        df['volume_component']
    )
    df['regime_adjustment'] = df['base_factor'] * df['regime_weight']
    
    # Signal Smoothing
    df['short_term_smoothing'] = df['base_factor'].rolling(window=3).mean()
    df['medium_term_trend'] = df['base_factor'].rolling(window=10).mean()
    df['smoothing_ratio'] = df['short_term_smoothing'] / df['medium_term_trend']
    df['smoothing_ratio'] = df['smoothing_ratio'].replace([np.inf, -np.inf], 1).fillna(1)
    
    # Final Factor Output
    df['volatility_scaled_factor'] = df['regime_adjustment'] / df['recent_volatility']
    df['volatility_scaled_factor'] = df['volatility_scaled_factor'].replace([np.inf, -np.inf], 0).fillna(0)
    
    df['trend_confirmed_factor'] = df['volatility_scaled_factor'] * df['smoothing_ratio']
    
    # Final factor with momentum alignment sign
    final_factor = df['trend_confirmed_factor'] * np.sign(df['momentum_alignment'].astype(float) - 0.5)
    
    return final_factor
