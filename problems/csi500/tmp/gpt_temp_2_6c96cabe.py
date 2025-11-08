import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate returns
    data['return_1d'] = data['close'].pct_change(1)
    data['return_3d'] = data['close'].pct_change(3)
    data['return_5d'] = data['close'].pct_change(5)
    data['return_10d'] = data['close'].pct_change(10)
    data['return_20d'] = data['close'].pct_change(20)
    
    # Momentum Component
    # Exponential decay weights
    momentum_weights = {
        1: np.exp(-0.1 * 1),
        3: np.exp(-0.1 * 3),
        5: np.exp(-0.1 * 5),
        10: np.exp(-0.1 * 10)
    }
    
    # Weighted momentum
    data['momentum'] = (
        momentum_weights[1] * data['return_1d'] +
        momentum_weights[3] * data['return_3d'] +
        momentum_weights[5] * data['return_5d'] +
        momentum_weights[10] * data['return_10d']
    )
    
    # Volume Component
    # Volume Trend Strength
    data['volume_roc_5d'] = (data['volume'] - data['volume'].shift(5)) / data['volume'].shift(5)
    data['volume_acceleration'] = (
        (data['volume'] / data['volume'].shift(1)) / 
        (data['volume'].shift(5) / data['volume'].shift(6))
    )
    
    # Price-Volume Efficiency
    # Calculate rolling volume-weighted return
    rolling_window = 5
    volume_weighted_returns = []
    for i in range(len(data)):
        if i >= rolling_window - 1:
            window_data = data.iloc[i-rolling_window+1:i+1]
            weighted_return = (
                (window_data['volume'] * window_data['return_1d']).sum() / 
                window_data['volume'].sum()
            )
            volume_weighted_returns.append(weighted_return)
        else:
            volume_weighted_returns.append(np.nan)
    
    data['volume_weighted_return'] = volume_weighted_returns
    data['efficiency_ratio'] = data['volume_weighted_return'] / data['return_5d']
    
    # Combine volume components
    data['volume_factor'] = 0.5 * data['volume_roc_5d'] + 0.3 * data['volume_acceleration'] + 0.2 * data['efficiency_ratio']
    
    # Volatility Component
    # Volatility Regime Identification
    data['volatility_20d'] = data['return_1d'].rolling(window=20).std()
    data['volatility_5d'] = data['return_1d'].rolling(window=5).std()
    data['volatility_ratio'] = data['volatility_5d'] / data['volatility_20d']
    
    # Range-Based Volatility
    data['daily_range'] = (data['high'] - data['low']) / data['close']
    
    # Range persistence (5-day correlation of ranges)
    range_persistence = []
    for i in range(len(data)):
        if i >= 9:
            recent_range = data['daily_range'].iloc[i-4:i+1].values
            previous_range = data['daily_range'].iloc[i-9:i-4].values
            if len(recent_range) == 5 and len(previous_range) == 5:
                corr = np.corrcoef(recent_range, previous_range)[0, 1]
                range_persistence.append(corr if not np.isnan(corr) else 0)
            else:
                range_persistence.append(0)
        else:
            range_persistence.append(0)
    
    data['range_persistence'] = range_persistence
    
    # Combine volatility components
    data['volatility_factor'] = 0.6 * data['volatility_ratio'] + 0.4 * data['range_persistence']
    
    # Regime Classification
    # Calculate rolling median of 20-day volatility (60-day window)
    data['volatility_20d_median'] = data['volatility_20d'].rolling(window=60).median()
    
    # Regime flags
    data['high_vol_regime'] = data['volatility_20d'] > data['volatility_20d_median']
    data['low_vol_regime'] = data['volatility_20d'] < data['volatility_20d_median']
    data['trending_market'] = abs(data['return_20d']) > (2 * data['volatility_20d'])
    data['mean_reverting_market'] = abs(data['return_20d']) < (0.5 * data['volatility_20d'])
    
    # Regime-Aware Combination
    alpha_signal = []
    
    for i in range(len(data)):
        if pd.isna(data['momentum'].iloc[i]) or pd.isna(data['volume_factor'].iloc[i]) or pd.isna(data['volatility_factor'].iloc[i]):
            alpha_signal.append(np.nan)
            continue
            
        # Get regime flags
        high_vol = data['high_vol_regime'].iloc[i]
        low_vol = data['low_vol_regime'].iloc[i]
        trending = data['trending_market'].iloc[i]
        mean_rev = data['mean_reverting_market'].iloc[i]
        
        # Default weights (moderate volatility regime)
        w_momentum, w_volume, w_volatility = 0.4, 0.35, 0.25
        
        # Adjust weights based on regimes
        if high_vol:
            w_momentum, w_volume, w_volatility = 0.3, 0.4, 0.3
        elif low_vol:
            w_momentum, w_volume, w_volatility = 0.5, 0.3, 0.2
        
        if trending:
            w_momentum, w_volume, w_volatility = 0.6, 0.3, 0.1
        elif mean_rev:
            w_momentum, w_volume, w_volatility = 0.2, 0.4, 0.4
        
        # Calculate regime-adaptive alpha
        alpha_value = (
            w_momentum * data['momentum'].iloc[i] +
            w_volume * data['volume_factor'].iloc[i] +
            w_volatility * data['volatility_factor'].iloc[i]
        )
        
        alpha_signal.append(alpha_value)
    
    # Create output series
    result = pd.Series(alpha_signal, index=data.index, name='regime_aware_alpha')
    
    return result
