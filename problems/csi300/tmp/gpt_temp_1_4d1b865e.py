import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying the original dataframe
    data = df.copy()
    
    # Calculate overnight gap
    data['overnight_gap'] = data['open'] / data['close'].shift(1) - 1
    
    # Calculate True Range
    data['tr1'] = data['high'] - data['low']
    data['tr2'] = abs(data['high'] - data['close'].shift(1))
    data['tr3'] = abs(data['low'] - data['close'].shift(1))
    data['true_range'] = data[['tr1', 'tr2', 'tr3']].max(axis=1)
    
    # Calculate 5-day Average True Range
    data['atr_5'] = data['true_range'].rolling(window=5, min_periods=5).mean()
    
    # Calculate gap magnitude relative to ATR
    data['gap_magnitude'] = abs(data['overnight_gap']) / data['atr_5']
    
    # Calculate volume-to-amount ratio
    data['volume_amount_ratio'] = data['volume'] / data['amount']
    
    # Calculate liquidity momentum (change in volume-amount ratio)
    data['liquidity_momentum'] = data['volume_amount_ratio'] - data['volume_amount_ratio'].shift(1)
    
    # Calculate historical percentile of liquidity momentum (20-day window)
    data['liquidity_percentile'] = data['liquidity_momentum'].rolling(window=20, min_periods=20).apply(
        lambda x: (x.iloc[-1] - x.mean()) / x.std() if x.std() > 0 else 0
    )
    
    # Generate reversal signal with hyperbolic tangent scaling
    data['reversal_signal'] = -np.sign(data['overnight_gap']) * np.tanh(data['gap_magnitude'])
    
    # Calculate normalized liquidity score with logistic transformation
    data['liquidity_score'] = 1 / (1 + np.exp(-data['liquidity_percentile']))
    
    # Combine reversal signal with liquidity filter
    data['alpha'] = data['reversal_signal'] * data['liquidity_score']
    
    # Return the alpha series
    return data['alpha']
