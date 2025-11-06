import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    data = df.copy()
    
    # Gap Momentum Persistence
    data['overnight_return'] = data['open'] / data['close'].shift(1) - 1
    data['gap_direction'] = np.sign(data['overnight_return'])
    
    # Track consecutive gap direction (max 5 days)
    gap_persistence = []
    current_streak = 0
    current_direction = 0
    
    for i in range(len(data)):
        if i == 0 or np.isnan(data['gap_direction'].iloc[i]):
            gap_persistence.append(0)
            current_streak = 0
            current_direction = 0
        elif data['gap_direction'].iloc[i] == current_direction:
            current_streak = min(current_streak + 1, 5)
            gap_persistence.append(current_streak)
        else:
            current_streak = 1
            current_direction = data['gap_direction'].iloc[i]
            gap_persistence.append(current_streak)
    
    data['gap_persistence_count'] = gap_persistence
    data['gap_momentum'] = data['overnight_return'] * data['gap_persistence_count']
    
    # Intraday Efficiency with Momentum Divergence
    data['intraday_range'] = data['high'] - data['low']
    data['open_close_diff'] = np.abs(data['open'] - data['close'])
    data['efficiency_ratio'] = np.where(data['open_close_diff'] > 0, 
                                       data['intraday_range'] / data['open_close_diff'], 1.0)
    
    data['momentum_5d'] = data['close'].pct_change(5)
    data['momentum_20d'] = data['close'].pct_change(20)
    data['momentum_divergence'] = data['momentum_5d'] - data['momentum_20d']
    data['efficiency_momentum'] = data['efficiency_ratio'] * data['momentum_divergence']
    
    # Volume-Weighted Acceleration
    data['return_t'] = data['close'].pct_change()
    data['return_t_1'] = data['close'].pct_change().shift(1)
    data['price_acceleration'] = data['return_t'] - data['return_t_1']
    
    data['volume_3d_avg'] = data['volume'].rolling(window=3, min_periods=1).mean()
    data['volume_weight'] = data['volume'] / data['volume_3d_avg']
    data['volume_acceleration'] = data['price_acceleration'] * data['volume_weight']
    
    # Track acceleration persistence (max 4 days)
    acc_persistence = []
    current_acc_streak = 0
    current_acc_direction = 0
    
    for i in range(len(data)):
        if i == 0 or np.isnan(data['price_acceleration'].iloc[i]):
            acc_persistence.append(0)
            current_acc_streak = 0
            current_acc_direction = 0
        elif np.sign(data['price_acceleration'].iloc[i]) == current_acc_direction:
            current_acc_streak = min(current_acc_streak + 1, 4)
            acc_persistence.append(current_acc_streak)
        else:
            current_acc_streak = 1
            current_acc_direction = np.sign(data['price_acceleration'].iloc[i])
            acc_persistence.append(current_acc_streak)
    
    data['acc_persistence_count'] = acc_persistence
    data['volume_weighted_acc'] = data['volume_acceleration'] * data['acc_persistence_count']
    
    # Volatility Adjustment & Signal Combination
    data['volatility_5d'] = (data['high'] - data['low']).rolling(window=5, min_periods=1).mean()
    data['volatility_scaling'] = 1.0 / (1.0 + data['volatility_5d'])
    
    # Apply volatility scaling to all components
    data['gap_momentum_adj'] = data['gap_momentum'] * data['volatility_scaling']
    data['efficiency_momentum_adj'] = data['efficiency_momentum'] * data['volatility_scaling']
    data['volume_weighted_acc_adj'] = data['volume_weighted_acc'] * data['volatility_scaling']
    
    # Multiply all three volatility-adjusted components
    data['combined_signal'] = (data['gap_momentum_adj'] * 
                              data['efficiency_momentum_adj'] * 
                              data['volume_weighted_acc_adj'])
    
    # Apply rolling percentiles for dynamic thresholds
    data['final_factor'] = data['combined_signal'].rolling(
        window=10, min_periods=1
    ).apply(lambda x: (x.iloc[-1] - np.percentile(x, 25)) / (np.percentile(x, 75) - np.percentile(x, 25)) 
            if (np.percentile(x, 75) - np.percentile(x, 25)) > 0 else 0.0, raw=False)
    
    return data['final_factor']
