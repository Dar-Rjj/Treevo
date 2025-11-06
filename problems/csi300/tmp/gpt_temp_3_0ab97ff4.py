import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original dataframe
    data = df.copy()
    
    # Calculate daily returns
    data['returns'] = data['close'].pct_change()
    
    # Volatility-Normalized Momentum Signal
    # 5-day raw price momentum
    data['momentum_5d'] = data['close'] / data['close'].shift(5) - 1
    
    # 20-day historical volatility using daily returns
    data['volatility_20d'] = data['returns'].rolling(window=20).std()
    
    # Risk-adjusted momentum (avoid division by zero)
    data['vol_adj_momentum_5d'] = data['momentum_5d'] / (data['volatility_20d'] + 1e-8)
    
    # Medium-term volatility-adjusted momentum (20-day)
    data['momentum_20d'] = data['close'] / data['close'].shift(20) - 1
    data['vol_adj_momentum_20d'] = data['momentum_20d'] / (data['volatility_20d'] + 1e-8)
    
    # Volume Confirmation Framework
    # Volume percentile rank over 50-day lookback
    data['volume_rank'] = data['volume'].rolling(window=50).apply(
        lambda x: (x[-1] > np.percentile(x[:-1], 70)) if len(x) > 1 else 0
    )
    
    # Directional volume flow
    data['price_change'] = data['close'] - data['close'].shift(1)
    data['directional_volume'] = np.sign(data['price_change']) * data['volume']
    
    # Volume signal strength
    conditions = [
        (data['volume_rank'] == 1) & (data['price_change'] > 0),  # Strong bullish
        (data['volume_rank'] == 1) & (data['price_change'] < 0),  # Strong bearish
        (data['volume_rank'] == 0)  # Weak signal
    ]
    choices = [1.5, 1.5, 0.5]  # Multipliers for volume confirmation
    data['volume_multiplier'] = np.select(conditions, choices, default=1.0)
    
    # Multi-Timeframe Convergence Alpha
    # Momentum convergence as product of short and medium-term signals
    data['momentum_convergence'] = data['vol_adj_momentum_5d'] * data['vol_adj_momentum_20d']
    
    # Trend persistence using momentum duration
    # Calculate momentum direction (1 for positive, -1 for negative, 0 for neutral)
    data['momentum_direction'] = np.sign(data['vol_adj_momentum_5d'])
    
    # Count consecutive days with same momentum direction
    data['consecutive_days'] = 0
    current_direction = 0
    consecutive_count = 0
    
    for i in range(len(data)):
        if i == 0:
            data.iloc[i, data.columns.get_loc('consecutive_days')] = 0
            current_direction = data['momentum_direction'].iloc[i]
            consecutive_count = 1
        else:
            if data['momentum_direction'].iloc[i] == current_direction and data['momentum_direction'].iloc[i] != 0:
                consecutive_count += 1
            else:
                consecutive_count = 1
                current_direction = data['momentum_direction'].iloc[i]
            
            data.iloc[i, data.columns.get_loc('consecutive_days')] = consecutive_count
    
    # Persistence multiplier: sqrt(consecutive_days) for trend strength
    data['persistence_multiplier'] = np.sqrt(data['consecutive_days'])
    
    # Final alpha calculation
    data['alpha'] = (data['momentum_convergence'] * 
                    data['volume_multiplier'] * 
                    data['persistence_multiplier'])
    
    # Return the alpha series with proper indexing
    return data['alpha']
