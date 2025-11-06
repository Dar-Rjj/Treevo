import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Price Gap Analysis
    data['overnight_gap'] = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
    data['intraday_realization'] = (data['close'] - data['open']) / data['open']
    
    # Volume Confirmation Signals
    data['volume_5d_avg'] = data['volume'].rolling(window=5, min_periods=1).mean()
    data['volume_surge'] = data['volume'] / data['volume_5d_avg']
    
    # Volume Consistency (count of days with volume above 5-day average in last 5 days)
    data['volume_above_avg'] = (data['volume'] > data['volume_5d_avg']).astype(int)
    data['volume_consistency'] = data['volume_above_avg'].rolling(window=5, min_periods=1).sum() / 5
    
    # Gap Persistence Assessment
    data['overnight_gap_sign'] = np.sign(data['overnight_gap'])
    
    # Gap Direction Persistence (excluding current day)
    def gap_persistence_func(x):
        if len(x) < 5:
            return 0.0
        current_sign = x.iloc[-1]
        past_signs = x.iloc[-5:-1]  # t-4 to t-1
        return (past_signs == current_sign).sum() / 4
    
    data['gap_direction_persistence'] = (
        data['overnight_gap_sign']
        .expanding(min_periods=5)
        .apply(gap_persistence_func, raw=False)
    )
    
    # Realization Efficiency
    data['realization_efficiency'] = (
        np.abs(data['intraday_realization']) / 
        (np.abs(data['overnight_gap']) + 0.0001)
    )
    
    # Market Regime Context
    data['recent_volatility'] = data['close'].rolling(window=5, min_periods=1).std()
    data['price_trend'] = (
        np.sign(data['close'].pct_change(periods=5)) * 
        (1 - (data['high'].rolling(window=5, min_periods=1).max() - 
              data['low'].rolling(window=5, min_periods=1).min()) / data['close'])
    )
    
    # Composite Alpha Factor
    # Base component
    base_factor = data['overnight_gap'] * data['intraday_realization']
    
    # Volume Multiplier
    volume_multiplier = np.where(
        data['volume_surge'] > 1.5, 1.3,
        np.where(data['volume_surge'] < 0.7, 0.8, 1.0)
    )
    
    # Persistence Multiplier
    persistence_multiplier = np.where(
        data['gap_direction_persistence'] > 0.75, 1.2,
        np.where(data['gap_direction_persistence'] < 0.5, 0.9, 1.0)
    )
    
    # Realization Multiplier
    realization_multiplier = np.where(
        data['realization_efficiency'] > 1.0, 1.4,
        np.where(data['realization_efficiency'] < 0.5, 0.6, 1.0)
    )
    
    # Regime Adjustment
    regime_adjustment = np.where(
        data['recent_volatility'] > 0.02, 0.7,
        np.where(data['price_trend'] > 0.1, 1.1, 1.0)
    )
    
    # Final composite factor
    composite_factor = (
        base_factor * 
        volume_multiplier * 
        persistence_multiplier * 
        realization_multiplier * 
        regime_adjustment
    )
    
    return pd.Series(composite_factor, index=data.index)
