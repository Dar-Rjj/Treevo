import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Multi-Timeframe Momentum with Range Volatility and Volume Confirmation alpha factor
    """
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Multi-Timeframe Momentum Calculation
    data['momentum_2d'] = data['close'] / data['close'].shift(2) - 1
    data['momentum_5d'] = data['close'] / data['close'].shift(5) - 1
    data['momentum_15d'] = data['close'] / data['close'].shift(15) - 1
    
    # Range-Based Volatility Adjustment
    # Ultra-short volatility (2-day)
    data['vol_2d_range'] = ((data['high'] - data['low']) / data['close'] + 
                           (data['high'].shift(1) - data['low'].shift(1)) / data['close'].shift(1)) / 2
    
    # Short-term volatility (5-day)
    vol_5d = []
    for i in range(len(data)):
        if i >= 4:
            vol_sum = 0
            for j in range(5):
                vol_sum += (data['high'].iloc[i-j] - data['low'].iloc[i-j]) / data['close'].iloc[i-j]
            vol_5d.append(vol_sum / 5)
        else:
            vol_5d.append(np.nan)
    data['vol_5d_range'] = vol_5d
    
    # Medium-term volatility (15-day)
    vol_15d = []
    for i in range(len(data)):
        if i >= 14:
            vol_sum = 0
            for j in range(15):
                vol_sum += (data['high'].iloc[i-j] - data['low'].iloc[i-j]) / data['close'].iloc[i-j]
            vol_15d.append(vol_sum / 15)
        else:
            vol_15d.append(np.nan)
    data['vol_15d_range'] = vol_15d
    
    # Volatility-adjusted momentum
    data['vol_adj_momentum_2d'] = data['momentum_2d'] / (data['vol_2d_range'] + 0.0001)
    data['vol_adj_momentum_5d'] = data['momentum_5d'] / (data['vol_5d_range'] + 0.0001)
    data['vol_adj_momentum_15d'] = data['momentum_15d'] / (data['vol_15d_range'] + 0.0001)
    
    # Dynamic Volume Confirmation
    # Volume momentum analysis
    data['volume_momentum_2d'] = data['volume'] / data['volume'].shift(2)
    data['volume_momentum_5d'] = data['volume'] / data['volume'].shift(5)
    data['volume_acceleration'] = data['volume_momentum_2d'] / data['volume_momentum_5d']
    
    # Volume regime classification
    volume_confirmation = []
    for i in range(len(data)):
        vol_mom_2d = data['volume_momentum_2d'].iloc[i]
        vol_mom_5d = data['volume_momentum_5d'].iloc[i]
        vol_acc = data['volume_acceleration'].iloc[i]
        momentum_2d = data['momentum_2d'].iloc[i]
        
        if pd.notna(vol_mom_2d) and pd.notna(vol_acc):
            if vol_mom_2d > 1.4 and vol_acc > 1.2:
                volume_confirmation.append(2.0)  # Strong breakout
            elif vol_mom_2d > 1.3 and vol_acc > 1.1:
                volume_confirmation.append(1.6)  # Breakout
            elif vol_mom_2d > 1.2 and vol_mom_5d > 1.1:
                volume_confirmation.append(1.3)  # Confirmation
            elif momentum_2d > 0 and vol_mom_2d < 0.85:
                volume_confirmation.append(0.7)  # Divergence
            elif vol_mom_2d < 0.7:
                volume_confirmation.append(0.5)  # Weakness
            else:
                volume_confirmation.append(1.0)  # Base regime
        else:
            volume_confirmation.append(1.0)
    
    data['volume_confirmation_multiplier'] = volume_confirmation
    
    # Momentum Quality Assessment
    # Momentum consistency
    consistency_scores = []
    for i in range(len(data)):
        mom_2d = data['momentum_2d'].iloc[i]
        mom_5d = data['momentum_5d'].iloc[i]
        mom_15d = data['momentum_15d'].iloc[i]
        
        if pd.notna(mom_2d) and pd.notna(mom_5d) and pd.notna(mom_15d):
            positive_count = sum([mom_2d > 0, mom_5d > 0, mom_15d > 0])
            
            if positive_count == 3:
                consistency_scores.append(1.4)  # All positive
            elif positive_count == 2:
                consistency_scores.append(1.2)  # Two positive
            elif positive_count == 1:
                consistency_scores.append(1.0)  # Mixed signs
            elif positive_count == 0:
                consistency_scores.append(0.6)  # All negative
            else:
                consistency_scores.append(1.0)
        else:
            consistency_scores.append(1.0)
    
    data['consistency_score'] = consistency_scores
    
    # Momentum acceleration
    acceleration_bonuses = []
    for i in range(len(data)):
        mom_2d = data['momentum_2d'].iloc[i]
        mom_5d = data['momentum_5d'].iloc[i]
        mom_15d = data['momentum_15d'].iloc[i]
        
        if pd.notna(mom_2d) and pd.notna(mom_5d) and pd.notna(mom_15d) and mom_5d != 0 and mom_15d != 0:
            acc_short = mom_2d / mom_5d if abs(mom_5d) > 0.001 else 1.0
            acc_medium = mom_5d / mom_15d if abs(mom_15d) > 0.001 else 1.0
            
            if acc_short > 1.15 and acc_medium > 1.15:
                acceleration_bonuses.append(1.3)  # Strong acceleration
            elif acc_short > 1.08 or acc_medium > 1.08:
                acceleration_bonuses.append(1.15)  # Moderate acceleration
            else:
                acceleration_bonuses.append(1.0)  # Base
        else:
            acceleration_bonuses.append(1.0)
    
    data['acceleration_bonus'] = acceleration_bonuses
    
    # Alpha Factor Construction
    # Combined volatility-adjusted momentum
    data['combined_momentum'] = (0.6 * data['vol_adj_momentum_2d'] + 
                                0.3 * data['vol_adj_momentum_5d'] + 
                                0.1 * data['vol_adj_momentum_15d'])
    
    # Apply volume confirmation and momentum quality
    data['alpha_factor'] = (data['combined_momentum'] * 
                           data['volume_confirmation_multiplier'] * 
                           data['consistency_score'] * 
                           data['acceleration_bonus'])
    
    # Liquidity scaling
    data['alpha_factor'] = data['alpha_factor'] * np.log(data['amount'] + 1)
    
    return data['alpha_factor']
