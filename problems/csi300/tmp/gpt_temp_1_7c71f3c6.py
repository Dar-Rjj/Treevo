import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Regime-Adaptive Momentum-Efficiency Divergence Factor
    Combines asymmetric momentum, multi-timeframe efficiency analysis, 
    volume acceleration patterns, and volatility regime classification
    """
    data = df.copy()
    
    # Calculate returns
    data['returns'] = data['close'].pct_change()
    data['prev_close'] = data['close'].shift(1)
    
    # 1. Asymmetric Momentum Component
    # Upside momentum: 5-day average of positive close returns
    positive_returns = data['returns'].where(data['returns'] > 0, 0)
    data['upside_momentum'] = positive_returns.rolling(window=5, min_periods=3).mean()
    
    # Downside momentum: 5-day average of negative close returns
    negative_returns = data['returns'].where(data['returns'] < 0, 0)
    data['downside_momentum'] = negative_returns.rolling(window=5, min_periods=3).mean()
    
    # Asymmetry ratio and momentum divergence
    data['asymmetry_ratio'] = data['upside_momentum'] / (abs(data['downside_momentum']) + 1e-8)
    data['momentum_divergence'] = data['asymmetry_ratio'] * data['close'].pct_change(periods=5)
    
    # 2. Multi-Timeframe Efficiency Analysis
    # Daily range calculation
    data['daily_range'] = (data['high'] - data['low']) / data['prev_close']
    
    # Short-term efficiency (5-day)
    data['return_5d'] = data['close'].pct_change(periods=5)
    data['avg_range_5d'] = data['daily_range'].rolling(window=5, min_periods=3).mean()
    data['efficiency_5d'] = data['return_5d'] / (data['avg_range_5d'] + 1e-8)
    
    # Long-term efficiency (20-day)
    data['return_20d'] = data['close'].pct_change(periods=20)
    data['avg_range_20d'] = data['daily_range'].rolling(window=20, min_periods=10).mean()
    data['efficiency_20d'] = data['return_20d'] / (data['avg_range_20d'] + 1e-8)
    
    # Efficiency divergence signal
    data['efficiency_gap'] = data['efficiency_5d'] - data['efficiency_20d']
    data['efficiency_divergence'] = data['efficiency_gap'] * data['return_5d']
    
    # 3. Volume Acceleration Patterns
    data['volume_change_3d'] = data['volume'].pct_change(periods=3)
    data['volume_change_8d'] = data['volume'].pct_change(periods=8)
    data['volume_acceleration'] = data['volume_change_3d'] - data['volume_change_8d']
    
    data['volume_change_5d'] = data['volume'].pct_change(periods=5)
    data['price_volume_divergence'] = data['return_5d'] * data['volume_change_5d']
    data['volume_momentum'] = data['volume_acceleration'] * data['price_volume_divergence']
    
    # 4. Volatility Regime Classification
    # Rolling volatility using daily range
    data['volatility_20d'] = data['daily_range'].rolling(window=20, min_periods=10).std()
    
    # Regime classification using historical percentiles
    volatility_percentile = data['volatility_20d'].rolling(window=60, min_periods=30).apply(
        lambda x: (x.iloc[-1] > np.percentile(x.dropna(), 70)) * 1 + 
                  (x.iloc[-1] < np.percentile(x.dropna(), 30)) * -1, 
        raw=False
    )
    
    # Regime stability (5-day window for regime changes)
    regime_stability = volatility_percentile.rolling(window=5, min_periods=3).std()
    data['regime_stability_score'] = 1 / (1 + regime_stability)
    
    # 5. Regime-Specific Signal Construction
    # Base combined signals
    data['momentum_efficiency_signal'] = data['momentum_divergence'] * data['efficiency_divergence']
    data['triple_divergence'] = data['momentum_efficiency_signal'] * data['volume_momentum']
    
    # Regime-adaptive weighting
    low_vol_weight = (volatility_percentile == -1).astype(float)
    high_vol_weight = (volatility_percentile == 1).astype(float)
    transition_weight = (volatility_percentile == 0).astype(float)
    
    # Regime-specific adjustments
    data['regime_adjusted_signal'] = (
        low_vol_weight * data['triple_divergence'] * 1.2 +  # Emphasize in low vol
        high_vol_weight * data['triple_divergence'] * 0.8 +  # De-emphasize in high vol
        transition_weight * data['triple_divergence'] * data['regime_stability_score']  # Smooth in transitions
    )
    
    # 6. Final Alpha Generation
    # Volatility-scaled output with risk management
    data['daily_range_efficiency'] = data['return_5d'] / (data['daily_range'] + 1e-8)
    data['final_alpha'] = data['regime_adjusted_signal'] * data['daily_range_efficiency']
    
    # Apply regime-appropriate filtering
    # Reduce signal strength during high volatility transitions
    volatility_filter = 1 - (high_vol_weight * transition_weight * 0.3)
    data['final_alpha'] = data['final_alpha'] * volatility_filter
    
    # Clean and return the alpha factor
    alpha = data['final_alpha'].replace([np.inf, -np.inf], np.nan)
    
    return alpha
