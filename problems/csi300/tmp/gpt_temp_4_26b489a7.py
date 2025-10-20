import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Price-Volume Efficiency Momentum Divergence factor
    Combines price efficiency, volume efficiency, and their divergence patterns
    to generate predictive signals for future returns
    """
    # Price efficiency calculations
    df['price_range'] = df['high'] - df['low']
    df['daily_return'] = df['close'].pct_change()
    
    # 5-day close-to-close efficiency (return per unit range)
    df['price_efficiency_5d'] = (df['close'] / df['close'].shift(5) - 1) / (
        df['price_range'].rolling(window=5, min_periods=3).mean() / df['close'].shift(5)
    )
    
    # 10-day close-to-close efficiency momentum
    df['price_efficiency_10d'] = (df['close'] / df['close'].shift(10) - 1) / (
        df['price_range'].rolling(window=10, min_periods=5).mean() / df['close'].shift(10)
    )
    df['price_efficiency_momentum'] = df['price_efficiency_5d'] - df['price_efficiency_10d'].shift(5)
    
    # Efficiency acceleration (5-day vs 21-day)
    df['price_efficiency_21d'] = (df['close'] / df['close'].shift(21) - 1) / (
        df['price_range'].rolling(window=21, min_periods=10).mean() / df['close'].shift(21)
    )
    df['efficiency_acceleration'] = df['price_efficiency_5d'] - df['price_efficiency_21d']
    
    # Volume efficiency calculations
    df['volume_efficiency'] = df['volume'] / (df['price_range'] + 1e-8)
    
    # 5-day volume efficiency momentum
    df['volume_efficiency_5d'] = df['volume_efficiency'].rolling(window=5, min_periods=3).mean()
    df['volume_efficiency_10d'] = df['volume_efficiency'].rolling(window=10, min_periods=5).mean()
    df['volume_efficiency_momentum'] = df['volume_efficiency_5d'] - df['volume_efficiency_10d'].shift(5)
    
    # Volume efficiency acceleration (5-day vs 21-day)
    df['volume_efficiency_21d'] = df['volume_efficiency'].rolling(window=21, min_periods=10).mean()
    df['volume_efficiency_acceleration'] = df['volume_efficiency_5d'] - df['volume_efficiency_21d']
    
    # Range momentum integration
    df['range_change_5d'] = df['price_range'].pct_change(periods=5)
    df['range_volume_interaction'] = df['range_change_5d'] * df['volume_efficiency_momentum']
    
    # Range-adjusted efficiency momentum
    df['range_adjusted_efficiency'] = df['price_efficiency_momentum'] / (df['price_range'].rolling(window=5, min_periods=3).std() + 1e-8)
    
    # Price-volume efficiency divergence
    df['price_efficiency_trend'] = df['price_efficiency_5d'].rolling(window=5, min_periods=3).apply(
        lambda x: 1 if (x.iloc[-1] > x.iloc[0]) else (-1 if x.iloc[-1] < x.iloc[0] else 0), raw=False
    )
    df['volume_efficiency_trend'] = df['volume_efficiency_5d'].rolling(window=5, min_periods=3).apply(
        lambda x: 1 if (x.iloc[-1] > x.iloc[0]) else (-1 if x.iloc[-1] < x.iloc[0] else 0), raw=False
    )
    
    df['efficiency_divergence'] = 0
    # Positive divergence: price efficiency up, volume efficiency down
    df.loc[(df['price_efficiency_trend'] == 1) & (df['volume_efficiency_trend'] == -1), 'efficiency_divergence'] = 1
    # Negative divergence: price efficiency down, volume efficiency up
    df.loc[(df['price_efficiency_trend'] == -1) & (df['volume_efficiency_trend'] == 1), 'efficiency_divergence'] = -1
    
    # Divergence strength assessment
    df['divergence_strength'] = (
        np.abs(df['price_efficiency_momentum']) * np.abs(df['volume_efficiency_momentum']) * 
        df['efficiency_divergence']
    )
    
    # Multi-timeframe divergence
    df['short_term_efficiency'] = df['price_efficiency_5d'].rolling(window=3, min_periods=2).mean()
    df['long_term_efficiency'] = df['price_efficiency_21d'].rolling(window=5, min_periods=3).mean()
    df['cross_timeframe_divergence'] = (
        (df['short_term_efficiency'] - df['long_term_efficiency']) * 
        np.sign(df['short_term_efficiency'].diff())
    )
    
    # Efficiency regime classification
    df['efficiency_20d_avg'] = df['price_efficiency_5d'].rolling(window=20, min_periods=10).mean()
    df['efficiency_regime'] = np.where(
        df['price_efficiency_5d'] > df['efficiency_20d_avg'], 1, 
        np.where(df['price_efficiency_5d'] < df['efficiency_20d_avg'], -1, 0)
    )
    
    # Efficiency breakout detection
    df['efficiency_breakout'] = (
        (df['price_efficiency_5d'] > df['efficiency_20d_avg'].shift(1)) & 
        (df['efficiency_regime'].shift(1) == -1)
    ).astype(int) - (
        (df['price_efficiency_5d'] < df['efficiency_20d_avg'].shift(1)) & 
        (df['efficiency_regime'].shift(1) == 1)
    ).astype(int)
    
    # Persistence analysis
    df['efficiency_improvement_streak'] = 0
    current_streak = 0
    for i in range(1, len(df)):
        if df['price_efficiency_5d'].iloc[i] > df['price_efficiency_5d'].iloc[i-1]:
            current_streak = max(current_streak + 1, 1)
        elif df['price_efficiency_5d'].iloc[i] < df['price_efficiency_5d'].iloc[i-1]:
            current_streak = min(current_streak - 1, -1)
        else:
            current_streak = 0
        df.iloc[i, df.columns.get_loc('efficiency_improvement_streak')] = current_streak
    
    # Volume efficiency persistence
    df['volume_efficiency_trend_streak'] = 0
    current_vol_streak = 0
    for i in range(1, len(df)):
        if df['volume_efficiency_5d'].iloc[i] > df['volume_efficiency_5d'].iloc[i-1]:
            current_vol_streak = max(current_vol_streak + 1, 1)
        elif df['volume_efficiency_5d'].iloc[i] < df['volume_efficiency_5d'].iloc[i-1]:
            current_vol_streak = min(current_vol_streak - 1, -1)
        else:
            current_vol_streak = 0
        df.iloc[i, df.columns.get_loc('volume_efficiency_trend_streak')] = current_vol_streak
    
    # Cross-efficiency persistence alignment
    df['persistence_alignment'] = np.sign(df['efficiency_improvement_streak']) * np.sign(df['volume_efficiency_trend_streak'])
    
    # Adaptive factor generation
    # Strong efficiency momentum signal
    df['strong_efficiency_signal'] = (
        (df['price_efficiency_momentum'] > df['price_efficiency_momentum'].rolling(window=10, min_periods=5).quantile(0.7)) &
        (df['volume_efficiency_momentum'] > 0) &
        (df['persistence_alignment'] > 0)
    ).astype(int)
    
    # Efficiency divergence reversal signal
    df['divergence_reversal_signal'] = (
        (df['efficiency_divergence'] != 0) &
        (df['cross_timeframe_divergence'].abs() > df['cross_timeframe_divergence'].rolling(window=10, min_periods=5).std()) &
        (df['persistence_alignment'] == 0)
    ).astype(int) * np.sign(df['efficiency_divergence'])
    
    # Composite efficiency divergence factor
    df['composite_efficiency_divergence'] = (
        df['divergence_strength'] * 0.4 +
        df['strong_efficiency_signal'] * 0.3 +
        df['divergence_reversal_signal'] * 0.2 +
        df['efficiency_breakout'] * 0.1
    )
    
    # Regime transition analysis
    df['efficiency_regime_change'] = df['efficiency_regime'].diff()
    df['regime_transition_momentum'] = df['efficiency_regime_change'] * df['price_efficiency_momentum']
    
    # Cross-regime efficiency divergence
    df['cross_regime_divergence'] = (
        (df['efficiency_regime'] != df['efficiency_regime'].shift(1)) * 
        df['cross_timeframe_divergence']
    )
    
    # Final factor calculation
    factor = (
        df['composite_efficiency_divergence'] * 0.5 +
        df['regime_transition_momentum'] * 0.3 +
        df['cross_regime_divergence'] * 0.2
    )
    
    # Clean up and return
    factor = factor.replace([np.inf, -np.inf], np.nan)
    factor = factor.fillna(method='ffill').fillna(0)
    
    return factor
