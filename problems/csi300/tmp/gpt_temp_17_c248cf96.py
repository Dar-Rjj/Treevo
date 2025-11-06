import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate 10-day fractal efficiency
    total_movement_10 = data['high'].rolling(window=10).apply(lambda x: np.sum(np.abs(x - data.loc[x.index, 'low'])), raw=False)
    net_movement_10 = np.abs(data['close'] - data['close'].shift(10))
    efficiency_10 = net_movement_10 / total_movement_10
    
    # Calculate 20-day fractal efficiency
    total_movement_20 = data['high'].rolling(window=20).apply(lambda x: np.sum(np.abs(x - data.loc[x.index, 'low'])), raw=False)
    net_movement_20 = np.abs(data['close'] - data['close'].shift(20))
    efficiency_20 = net_movement_20 / total_movement_20
    
    # Calculate volume acceleration
    volume_momentum_5 = data['volume'] / data['volume'].shift(5) - 1
    volume_momentum_10 = data['volume'] / data['volume'].shift(10) - 1
    volume_acceleration = volume_momentum_5 - volume_momentum_10
    
    # Calculate price momentum acceleration
    price_momentum_5 = data['close'] / data['close'].shift(5) - 1
    price_momentum_10 = data['close'] / data['close'].shift(10) - 1
    momentum_acceleration = price_momentum_5 - price_momentum_10
    
    # Combine components
    efficiency_product = efficiency_10 * efficiency_20
    volume_adjusted = efficiency_product * (1 + volume_acceleration)
    final_factor = volume_adjusted * momentum_acceleration
    
    return final_factor
