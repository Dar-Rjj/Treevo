import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Compute Momentum Divergence
    # Short-Term Momentum (5-day)
    short_term_momentum = (data['close'] - data['close'].shift(5)) / data['close'].shift(5)
    
    # Medium-Term Momentum (20-day)
    medium_term_momentum = (data['close'] - data['close'].shift(20)) / data['close'].shift(20)
    
    # Compute Divergence Signal
    momentum_divergence = np.abs(short_term_momentum - medium_term_momentum)
    
    # Calculate Volatility Components
    # Daily Range
    daily_range = (data['high'] - data['low']) / data['close']
    
    # Volatility Regime
    avg_range_20 = daily_range.rolling(window=20, min_periods=10).mean()
    avg_range_60 = daily_range.rolling(window=60, min_periods=30).mean()
    
    # Regime classification: 1 for low volatility, -1 for high volatility
    volatility_regime = np.where(avg_range_20 < avg_range_60, 1.0, -1.0)
    
    # Assess Asymmetric Volatility
    daily_returns = data['close'].pct_change()
    
    # Separate up and down days
    up_days = daily_returns > 0
    down_days = daily_returns < 0
    
    # Calculate average range for positive vs negative return days (using 20-day window)
    up_day_avg_range = daily_range.rolling(window=20).apply(
        lambda x: x[up_days.loc[x.index].values].mean() if up_days.loc[x.index].sum() > 0 else np.nan, 
        raw=False
    )
    down_day_avg_range = daily_range.rolling(window=20).apply(
        lambda x: x[down_days.loc[x.index].values].mean() if down_days.loc[x.index].sum() > 0 else np.nan, 
        raw=False
    )
    
    # Asymmetric volatility ratio
    asymmetric_ratio = down_day_avg_range / up_day_avg_range
    asymmetric_ratio = asymmetric_ratio.fillna(1.0)  # Handle division by zero
    
    # Apply Combined Adjustment
    # Volatility Regime Weighting (higher weight in low volatility regimes)
    regime_weight = np.where(volatility_regime == 1, 1.2, 0.8)
    
    # Asymmetric Volatility Adjustment
    # Emphasize divergence signals during low volatility down days
    asymmetric_adjustment = np.where(
        (volatility_regime == 1) & (daily_returns < 0), 
        1.5,  # Strong emphasis on divergence during low volatility down days
        np.where(
            (volatility_regime == -1) & (daily_returns > 0),
            0.7,  # Reduced emphasis during high volatility up days
            1.0   # Neutral adjustment otherwise
        )
    )
    
    # Combine Signals
    # Multiply divergence by regime weight and apply asymmetric adjustment
    final_factor = momentum_divergence * regime_weight * asymmetric_adjustment
    
    return final_factor
