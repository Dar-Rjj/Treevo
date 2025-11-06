import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Generate alpha factor combining multiple market microstructure signals
    """
    result = pd.Series(index=df.index, dtype=float)
    
    for i in range(3, len(df)):
        if i < 3:
            result.iloc[i] = 0
            continue
            
        # Price-Volume Divergence
        price_momentum = df['close'].iloc[i] / df['close'].iloc[i-3] - 1
        volume_momentum = df['volume'].iloc[i] / df['volume'].iloc[i-3]
        pv_divergence = (price_momentum - volume_momentum) * price_momentum
        
        # Range Efficiency Momentum
        daily_efficiency = abs(df['close'].iloc[i] - df['close'].iloc[i-1]) / (df['high'].iloc[i] - df['low'].iloc[i])
        three_day_momentum = df['close'].iloc[i] / df['close'].iloc[i-3] - 1
        efficiency_momentum = daily_efficiency * three_day_momentum
        
        # Volume-Confirmed Reversals
        extreme_move = abs(df['close'].iloc[i] / df['close'].iloc[i-1] - 1)
        volume_ratio = df['volume'].iloc[i] / df['volume'].iloc[i-1]
        reversal_signal = -np.sign(df['close'].iloc[i] - df['close'].iloc[i-1]) * extreme_move / volume_ratio
        
        # Amount Flow Direction
        daily_price_change = df['close'].iloc[i] - df['close'].iloc[i-1]
        amount_flow_t = df['amount'].iloc[i] * np.sign(daily_price_change)
        
        if i >= 5:
            amount_flow_sum = sum(df['amount'].iloc[j] * np.sign(df['close'].iloc[j] - df['close'].iloc[j-1]) 
                                for j in range(i-2, i+1))
            amount_sum = sum(df['amount'].iloc[j] for j in range(i-2, i+1))
            flow_persistence = amount_flow_sum / amount_sum if amount_sum != 0 else 0
        else:
            flow_persistence = 0
        
        # Volatility-Regime Signals
        daily_volatility = (df['high'].iloc[i] - df['low'].iloc[i]) / df['close'].iloc[i]
        
        if i >= 5:
            five_day_volatility = sum((df['high'].iloc[j] - df['low'].iloc[j]) / df['close'].iloc[j] 
                                    for j in range(i-4, i+1)) / 5
        else:
            five_day_volatility = daily_volatility
            
        regime_signal = (daily_volatility - five_day_volatility) * (df['close'].iloc[i] - df['close'].iloc[i-1])
        
        # Combine all signals with equal weights
        combined_signal = (pv_divergence + efficiency_momentum + reversal_signal + 
                          flow_persistence + regime_signal) / 5
        
        result.iloc[i] = combined_signal
    
    # Fill initial values
    result = result.fillna(0)
    
    return result
