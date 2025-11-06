import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Generate novel alpha factors using bid-ask imbalance dynamics, regime-switching volatility capture,
    volume-price divergence momentum, and microstructure informed reversal patterns.
    """
    result = pd.Series(index=df.index, dtype=float)
    
    for i in range(len(df)):
        if i < 20:  # Need enough data for rolling calculations
            result.iloc[i] = 0
            continue
            
        # Current values
        open_ = df['open'].iloc[i]
        high = df['high'].iloc[i]
        low = df['low'].iloc[i]
        close = df['close'].iloc[i]
        volume = df['volume'].iloc[i]
        
        # Historical values
        close_prev = df['close'].iloc[i-1]
        volume_prev = df['volume'].iloc[i-1]
        close_prev3 = df['close'].iloc[i-3] if i >= 3 else close
        volume_prev3 = df['volume'].iloc[i-3] if i >= 3 else volume
        
        # Bid-Ask Imbalance Dynamics
        if (close - low) != 0:
            bid_ask_1 = ((high - close) / (close - low)) * np.sign(volume - volume_prev)
        else:
            bid_ask_1 = 0
            
        if (high - low) != 0:
            bid_ask_2 = ((close - open_) / (high - low)) * ((volume / volume_prev) - 1)
        else:
            bid_ask_2 = 0
            
        # Regime-Switching Volatility Capture
        rolling_median_10 = df['close'].iloc[i-9:i+1].median()
        rolling_std_20 = df['close'].iloc[i-19:i+1].std()
        rolling_mean_5 = df['close'].iloc[i-4:i+1].mean()
        
        if (high - low) != 0:
            regime_1 = (close - rolling_median_10) * abs(close - close_prev) / (high - low)
        else:
            regime_1 = 0
            
        if rolling_std_20 != 0:
            regime_2 = ((high - low) / rolling_std_20) * np.sign(close - rolling_mean_5)
        else:
            regime_2 = 0
            
        # Volume-Price Divergence Momentum
        if close_prev3 != 0:
            volume_momentum_1 = ((close - close_prev3) / close_prev3) * ((volume / volume_prev3) - 1)
        else:
            volume_momentum_1 = 0
            
        if volume_prev != 0:
            volume_momentum_2 = ((close / close_prev) - (volume / volume_prev)) * abs(close - open_)
        else:
            volume_momentum_2 = 0
            
        # Microstructure Informed Reversal
        rolling_median_volume_10 = df['volume'].iloc[i-9:i+1].median()
        rolling_mean_close_3 = df['close'].iloc[i-2:i+1].mean()
        
        if (high - low) != 0:
            reversal_1 = ((open_ - close_prev) / (high - low)) * np.sign(volume - rolling_median_volume_10)
        else:
            reversal_1 = 0
            
        if abs(close - open_) != 0:
            reversal_2 = (close - rolling_mean_close_3) * (high - low) / abs(close - open_)
        else:
            reversal_2 = 0
            
        # Combine factors with equal weighting
        combined_factor = (
            bid_ask_1 + bid_ask_2 + 
            regime_1 + regime_2 + 
            volume_momentum_1 + volume_momentum_2 + 
            reversal_1 + reversal_2
        )
        
        result.iloc[i] = combined_factor
    
    return result
