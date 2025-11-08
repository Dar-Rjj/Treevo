import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Volatility Regime Classification
    data['daily_range'] = (data['high'] - data['low']) / data['close']
    data['avg_5day_range'] = data['daily_range'].rolling(window=5, min_periods=5).mean()
    data['high_vol_regime'] = data['daily_range'] > data['avg_5day_range']
    
    # Price Efficiency Analysis
    data['daily_efficiency'] = (data['close'] - data['open']) / (data['high'] - data['low'])
    data['daily_efficiency'] = data['daily_efficiency'].replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Efficiency trend - count consecutive days with same efficiency direction
    data['efficiency_sign'] = np.sign(data['daily_efficiency'])
    data['efficiency_trend_count'] = 0
    for i in range(1, len(data)):
        if data['efficiency_sign'].iloc[i] == data['efficiency_sign'].iloc[i-1]:
            data['efficiency_trend_count'].iloc[i] = data['efficiency_trend_count'].iloc[i-1] + 1
        else:
            data['efficiency_trend_count'].iloc[i] = 1
    
    # Overnight gap efficiency
    data['overnight_gap'] = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
    
    # Momentum Convergence Components
    # Price Momentum
    data['price_momentum_5d'] = data['close'] / data['close'].shift(5) - 1
    data['price_momentum_20d'] = data['close'] / data['close'].shift(20) - 1
    
    # Volume Momentum
    data['volume_momentum_5d'] = data['volume'] / data['volume'].shift(5) - 1
    data['volume_momentum_20d'] = data['volume'] / data['volume'].shift(20) - 1
    
    # Efficiency Momentum
    data['efficiency_momentum_5d'] = data['daily_efficiency'] / data['daily_efficiency'].shift(5) - 1
    data['efficiency_momentum_20d'] = data['daily_efficiency'] / data['daily_efficiency'].shift(20) - 1
    
    # Replace infinite values in efficiency momentum
    data['efficiency_momentum_5d'] = data['efficiency_momentum_5d'].replace([np.inf, -np.inf], np.nan).fillna(0)
    data['efficiency_momentum_20d'] = data['efficiency_momentum_20d'].replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Convergence Signal Generation
    # Raw convergence values
    data['price_convergence'] = data['price_momentum_5d'] * data['price_momentum_20d']
    data['volume_convergence'] = data['volume_momentum_5d'] * data['volume_momentum_20d']
    data['efficiency_convergence'] = data['efficiency_momentum_5d'] * data['efficiency_momentum_20d']
    
    # Direction alignment analysis
    data['price_converging'] = np.sign(data['price_momentum_5d']) == np.sign(data['price_momentum_20d'])
    data['volume_converging'] = np.sign(data['volume_momentum_5d']) == np.sign(data['volume_momentum_20d'])
    data['efficiency_converging'] = np.sign(data['efficiency_momentum_5d']) == np.sign(data['efficiency_momentum_20d'])
    
    # Count converging components
    data['converging_count'] = (data['price_converging'].astype(int) + 
                               data['volume_converging'].astype(int) + 
                               data['efficiency_converging'].astype(int))
    
    # Base Convergence Factor
    data['base_convergence'] = (data['price_convergence'] * 
                               data['volume_convergence'] * 
                               data['efficiency_convergence'])
    
    # Apply direction alignment multiplier
    conditions = [
        data['converging_count'] == 3,
        data['converging_count'] == 2,
        data['converging_count'] == 1,
        data['converging_count'] == 0
    ]
    choices = [3.0, 1.5, 0.7, 0.2]
    data['alignment_multiplier'] = np.select(conditions, choices, default=1.0)
    data['base_factor'] = data['base_convergence'] * data['alignment_multiplier']
    
    # Volatility Regime Enhancement
    data['final_factor'] = data['base_factor'].copy()
    
    # High volatility regime
    high_vol_mask = data['high_vol_regime'] == True
    data.loc[high_vol_mask, 'final_factor'] = (
        data.loc[high_vol_mask, 'base_factor'] * 
        (data.loc[high_vol_mask, 'volume'] / data.loc[high_vol_mask, 'volume'].shift(1)) * 
        (1 + abs(data.loc[high_vol_mask, 'daily_efficiency']))
    )
    data.loc[high_vol_mask, 'final_factor'] = (
        data.loc[high_vol_mask, 'final_factor'] * 
        (1 - abs(data.loc[high_vol_mask, 'price_convergence'])) * 1.2
    )
    
    # Low volatility regime
    low_vol_mask = data['high_vol_regime'] == False
    data.loc[low_vol_mask, 'final_factor'] = (
        data.loc[low_vol_mask, 'base_factor'] * 
        (1 + abs(data.loc[low_vol_mask, 'price_convergence'])) * 
        (1 + abs(data.loc[low_vol_mask, 'daily_efficiency']))
    )
    data.loc[low_vol_mask, 'final_factor'] = (
        data.loc[low_vol_mask, 'final_factor'] * 
        (1 + data.loc[low_vol_mask, 'efficiency_trend_count'] / 10) * 0.9
    )
    
    # Clean and return the factor
    factor = data['final_factor'].replace([np.inf, -np.inf], np.nan).fillna(0)
    return factor
