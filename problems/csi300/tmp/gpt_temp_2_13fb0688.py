import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Generate novel alpha factors based on liquidity-volatility interaction, 
    price-volume efficiency, and dynamic factor integration.
    """
    # Create a copy to avoid modifying original dataframe
    data = df.copy()
    
    # Initialize factor series
    factor = pd.Series(index=data.index, dtype=float)
    
    # Calculate required components with proper shifting to avoid lookahead bias
    for i in range(2, len(data)):
        if i < 2:
            factor.iloc[i] = 0
            continue
            
        # Current day values
        open_t = data['open'].iloc[i]
        high_t = data['high'].iloc[i]
        low_t = data['low'].iloc[i]
        close_t = data['close'].iloc[i]
        amount_t = data['amount'].iloc[i]
        volume_t = data['volume'].iloc[i]
        
        # Previous day values (t-1)
        open_t1 = data['open'].iloc[i-1]
        high_t1 = data['high'].iloc[i-1]
        low_t1 = data['low'].iloc[i-1]
        close_t1 = data['close'].iloc[i-1]
        amount_t1 = data['amount'].iloc[i-1]
        volume_t1 = data['volume'].iloc[i-1]
        
        # Two days ago values (t-2)
        high_t2 = data['high'].iloc[i-2]
        low_t2 = data['low'].iloc[i-2]
        close_t2 = data['close'].iloc[i-2]
        volume_t2 = data['volume'].iloc[i-2]
        
        # Liquidity-Volatility Interaction factors
        # Micro-Liquidity Stress
        if amount_t != 0:
            micro_liquidity = ((high_t - low_t) / amount_t) * np.sign(close_t - open_t)
        else:
            micro_liquidity = 0
            
        # Volatility Regime Switch
        if close_t != 0 and close_t2 != 0 and volume_t2 != 0:
            vol_regime = np.sign(
                ((high_t - low_t) / close_t) - ((high_t2 - low_t2) / close_t2)
            ) * (volume_t / volume_t2)
        else:
            vol_regime = 0
        
        # Price-Volume Efficiency factors
        # Hidden Pressure
        if (high_t - low_t) != 0 and amount_t != 0 and (high_t1 - low_t1) != 0 and amount_t1 != 0:
            hidden_pressure = (
                ((close_t - low_t) / (high_t - low_t)) * (volume_t / amount_t) - 
                ((close_t1 - low_t1) / (high_t1 - low_t1)) * (volume_t1 / amount_t1)
            )
        else:
            hidden_pressure = 0
            
        # Volume-Weighted Efficiency
        if volume_t != 0 and amount_t != 0 and (high_t - low_t) != 0:
            vol_weighted_eff = ((close_t - open_t) / volume_t) * ((high_t - low_t) / amount_t)
        else:
            vol_weighted_eff = 0
        
        # Dynamic Factor Integration factors
        # Regime-Adaptive Momentum
        if close_t2 != 0 and volume_t2 != 0 and close_t != 0 and close_t2 != 0:
            regime_momentum = (
                (close_t / close_t2 - 1) * 
                (volume_t / volume_t2) * 
                np.sign(((high_t - low_t) / close_t) - ((high_t2 - low_t2) / close_t2))
            )
        else:
            regime_momentum = 0
            
        # Liquidity Convergence
        if (amount_t1 / volume_t1) != 0 and (high_t - low_t) != 0:
            liquidity_conv = (
                ((amount_t / volume_t) / (amount_t1 / volume_t1)) * 
                ((close_t - open_t) / (high_t - low_t))
            )
        else:
            liquidity_conv = 0
        
        # Combine all factors with equal weighting
        combined_factor = (
            micro_liquidity + 
            vol_regime + 
            hidden_pressure + 
            vol_weighted_eff + 
            regime_momentum + 
            liquidity_conv
        )
        
        factor.iloc[i] = combined_factor
    
    # Fill initial values with 0
    factor = factor.fillna(0)
    
    return factor
