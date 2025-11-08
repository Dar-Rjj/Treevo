import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Momentum Persistence with Volume-Regime Alignment alpha factor
    """
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Initialize result series
    result = pd.Series(index=df.index, dtype=float)
    
    # Calculate basic price returns
    df['price_return'] = df['close'] - df['close'].shift(1)
    df['intraday_momentum'] = df['close'] - df['open']
    
    # Multi-Timeframe Momentum
    # Ultra-Short (1-day)
    df['ultra_short_return'] = df['price_return']
    
    # Short-Term (3-day)
    df['short_term_return'] = df['close'] - df['close'].shift(2)
    
    # Calculate daily returns for consistency
    df['daily_return'] = df['close'].pct_change()
    df['return_consistency'] = (
        (df['daily_return'].shift(2) > 0).astype(int) +
        (df['daily_return'].shift(1) > 0).astype(int) +
        (df['daily_return'] > 0).astype(int)
    )
    
    # Medium-Term (5-day)
    df['medium_term_return'] = df['close'] - df['close'].shift(4)
    
    # Persistence Score
    df['persistence_score'] = (
        np.sign(df['daily_return'].shift(4)) +
        np.sign(df['daily_return'].shift(3)) +
        np.sign(df['daily_return'].shift(2)) +
        np.sign(df['daily_return'].shift(1)) +
        np.sign(df['daily_return'])
    )
    
    # Momentum Quality Metrics
    # Acceleration Signals
    df['short_term_accel'] = (df['close'] - df['close'].shift(1)) - (df['close'].shift(1) - df['close'].shift(2))
    df['medium_term_accel'] = (df['close'] - df['close'].shift(2)) - (df['close'].shift(2) - df['close'].shift(4))
    
    # Stability Measures
    df['drawdown_resistance'] = np.minimum(0, np.minimum(
        df['close']/df['close'].shift(1) - 1,
        np.minimum(df['close']/df['close'].shift(2) - 1,
                  np.minimum(df['close']/df['close'].shift(3) - 1,
                            df['close']/df['close'].shift(4) - 1))
    ))
    
    # Calculate rolling min and max for recovery strength
    df['min_close_5d'] = df['close'].rolling(window=5, min_periods=5).min()
    df['recovery_strength'] = (df['close'] - df['min_close_5d']) / (df['high'] - df['low'])
    
    # Volatility-Adjusted Momentum
    df['short_term_volatility'] = (
        (df['high'] - df['low']).shift(2) +
        (df['high'] - df['low']).shift(1) +
        (df['high'] - df['low'])
    )
    df['medium_term_volatility'] = (
        (df['high'] - df['low']).shift(4) +
        (df['high'] - df['low']).shift(3) +
        (df['high'] - df['low']).shift(2) +
        (df['high'] - df['low']).shift(1) +
        (df['high'] - df['low'])
    )
    
    df['short_term_mar'] = df['short_term_return'] / (df['short_term_volatility'] + 1)
    df['medium_term_mar'] = df['medium_term_return'] / (df['medium_term_volatility'] + 1)
    
    # Volume Persistence Analysis
    df['volume_change'] = df['volume'] - df['volume'].shift(1)
    df['volume_direction'] = np.sign(df['volume_change'])
    
    # Calculate volume direction streak
    df['volume_direction_streak'] = 0
    for i in range(1, len(df)):
        if df['volume_direction'].iloc[i] == df['volume_direction'].iloc[i-1]:
            df['volume_direction_streak'].iloc[i] = df['volume_direction_streak'].iloc[i-1] + 1
        else:
            df['volume_direction_streak'].iloc[i] = 1
    
    df['volume_change_magnitude'] = np.abs(df['volume_change'])
    df['volume_persistence_strength'] = df['volume_direction_streak'] * df['volume_change_magnitude']
    
    # Volume Regime Classification
    df['short_term_avg_volume'] = (df['volume'] + df['volume'].shift(1) + df['volume'].shift(2)) / 3
    df['medium_term_avg_volume'] = df['volume'].rolling(window=5, min_periods=5).mean()
    df['volume_regime_ratio'] = df['short_term_avg_volume'] / df['medium_term_avg_volume']
    
    # Volume-Momentum Alignment
    df['alignment_signal'] = np.sign(df['price_return']) * np.sign(df['volume_change'])
    
    # Calculate alignment streak
    df['alignment_streak'] = 0
    for i in range(1, len(df)):
        if df['alignment_signal'].iloc[i] > 0:
            df['alignment_streak'].iloc[i] = df['alignment_streak'].iloc[i-1] + 1
        else:
            df['alignment_streak'].iloc[i] = 0
    
    df['alignment_confidence'] = df['alignment_streak'] * np.abs(df['price_return'])
    df['volume_responsive_momentum'] = df['price_return'] * np.log(df['volume'] + 1)
    df['momentum_efficiency'] = np.abs(df['price_return']) / (np.abs(df['volume_change']) + 1)
    df['coordination_quality'] = df['alignment_confidence'] * df['momentum_efficiency']
    
    # Regime Detection
    df['volatility_ratio'] = df['short_term_volatility'] / df['medium_term_volatility']
    df['volatility_regime'] = np.select(
        [df['volatility_ratio'] > 1.1, df['volatility_ratio'] < 0.9],
        ['high', 'low'],
        default='normal'
    )
    
    df['volume_regime'] = np.select(
        [df['volume_regime_ratio'] > 1.05, df['volume_regime_ratio'] < 0.95],
        ['high', 'low'],
        default='normal'
    )
    
    df['momentum_regime'] = np.select(
        [(df['persistence_score'] >= 3) & (df['short_term_accel'] > 0),
         (df['persistence_score'] <= -3) & (df['short_term_accel'] < 0)],
        ['strong', 'weak'],
        default='transitional'
    )
    
    # Base Momentum Score
    df['weighted_momentum'] = (
        4 * df['ultra_short_return'] + 
        3 * df['short_term_return'] + 
        2 * df['medium_term_return']
    ) / 9
    
    df['base_momentum'] = df['weighted_momentum'] * (1 + df['return_consistency'] / 5)
    
    # Volume Alignment Integration
    df['volume_confirmed_momentum'] = df['base_momentum'] * (1 + df['alignment_confidence'] / 50)
    df['persistence_boosted'] = df['volume_confirmed_momentum'] * (1 + df['volume_persistence_strength'] / 1000)
    
    # Risk Adjustment
    df['volatility_scaled'] = df['persistence_boosted'] / (df['medium_term_volatility'] + 1)
    df['drawdown_protected'] = df['volatility_scaled'] * (1 + df['drawdown_resistance'])
    
    # Regime-Specific Enhancements
    # Volatility regime weights
    volatility_multiplier = np.select(
        [df['volatility_regime'] == 'high', df['volatility_regime'] == 'low'],
        [0.8, 1.2],
        default=1.0
    )
    
    # Volume regime multipliers
    volume_multiplier = np.select(
        [df['volume_regime'] == 'high', df['volume_regime'] == 'low'],
        [1.15, 0.85],
        default=1.0
    )
    
    # Momentum regime focus
    momentum_multiplier = np.select(
        [df['momentum_regime'] == 'strong', df['momentum_regime'] == 'weak'],
        [1.1, 0.9],
        default=1.0
    )
    
    # Apply regime adjustments
    df['regime_adjusted'] = (
        df['drawdown_protected'] * 
        volatility_multiplier * 
        volume_multiplier * 
        momentum_multiplier
    )
    
    # Final Signal Refinement
    # Persistence Filter
    alignment_filter = (df['alignment_streak'] >= 2).astype(float)
    momentum_filter = (df['short_term_accel'] > 0).astype(float)
    volume_regime_filter = (df['volume_regime'] != 'low').astype(float)
    
    # Combine filters
    persistence_filter = (alignment_filter + momentum_filter + volume_regime_filter) / 3
    
    # Regime consistency check
    volatility_volume_aligned = (
        ((df['volatility_regime'] == 'high') & (df['volume_regime'] == 'high')) |
        ((df['volatility_regime'] == 'normal') & (df['volume_regime'] == 'normal')) |
        ((df['volatility_regime'] == 'low') & (df['volume_regime'] == 'low'))
    ).astype(float)
    
    momentum_regime_fit = (
        ((df['momentum_regime'] == 'strong') & (df['regime_adjusted'] > 0)) |
        ((df['momentum_regime'] == 'weak') & (df['regime_adjusted'] < 0)) |
        (df['momentum_regime'] == 'transitional')
    ).astype(float)
    
    regime_consistency = (volatility_volume_aligned + momentum_regime_fit) / 2
    
    # Final composite factor
    df['composite_factor'] = (
        df['regime_adjusted'] * 
        persistence_filter * 
        regime_consistency
    )
    
    # Normalize and return
    result = df['composite_factor']
    result = (result - result.mean()) / result.std()
    
    return result
