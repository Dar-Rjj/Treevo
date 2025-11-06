import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Momentum-Volume Asymmetry and Microstructure Regime Detection factor
    """
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Initialize factor series
    factor = pd.Series(index=data.index, dtype=float)
    
    # Calculate rolling statistics for lookback periods
    for i in range(max(10, len(data))):
        if i < 5:  # Need sufficient history
            factor.iloc[i] = 0
            continue
            
        current_data = data.iloc[:i+1]
        
        # 1. Asymmetric Momentum Dynamics
        # Directional momentum persistence
        if i >= 3:
            # Up-momentum sustainability
            momentum_3d = current_data['close'].iloc[i] - current_data['close'].iloc[i-3]
            vol_3d = sum(abs(current_data['close'].iloc[j] - current_data['close'].iloc[j-1]) 
                        for j in range(i-2, i+1))
            up_momentum_sustain = momentum_3d / vol_3d if vol_3d != 0 else 0
            
            # Down-momentum acceleration
            down_momentum_accel = (current_data['close'].iloc[i-3] - current_data['close'].iloc[i]) / \
                                (current_data['high'].iloc[i] - current_data['low'].iloc[i]) \
                                if (current_data['high'].iloc[i] - current_data['low'].iloc[i]) != 0 else 0
            
            # Momentum reversal resistance
            uptrend_count = 0
            for j in range(max(0, i-5), i+1):
                if j > 0 and current_data['close'].iloc[j] > current_data['close'].iloc[j-1]:
                    uptrend_count += 1
            reversal_resistance = uptrend_count / 6.0
        else:
            up_momentum_sustain = down_momentum_accel = reversal_resistance = 0
        
        # Intraday momentum quality
        intraday_range = current_data['high'].iloc[i] - current_data['low'].iloc[i]
        if intraday_range != 0:
            close_high_strength = (current_data['close'].iloc[i] - current_data['open'].iloc[i]) / intraday_range
            gap_persistence = (current_data['open'].iloc[i] - current_data['close'].iloc[i-1]) / intraday_range
            eod_momentum = (current_data['close'].iloc[i] - 
                           current_data['close'].iloc[max(0, i-4):i].mean()) / intraday_range
        else:
            close_high_strength = gap_persistence = eod_momentum = 0
        
        # 2. Volume Asymmetry Patterns
        # Buy/sell pressure imbalance
        if i >= 10:
            up_volume = down_volume = 0
            for j in range(i-9, i+1):
                if j > 0 and current_data['close'].iloc[j] > current_data['close'].iloc[j-1]:
                    up_volume += current_data['volume'].iloc[j]
                elif j > 0 and current_data['close'].iloc[j] < current_data['close'].iloc[j-1]:
                    down_volume += current_data['volume'].iloc[j]
            volume_concentration = up_volume / down_volume if down_volume != 0 else 1
            
            # Volume persistence asymmetry
            if current_data['close'].iloc[i] > current_data['close'].iloc[i-1]:
                volume_persistence = current_data['volume'].iloc[i] / current_data['volume'].iloc[i-1] \
                                   if current_data['volume'].iloc[i-1] != 0 else 1
            else:
                volume_persistence = 1 - (current_data['volume'].iloc[i] / current_data['volume'].iloc[i-1] \
                                        if current_data['volume'].iloc[i-1] != 0 else 0)
            
            # High-volume reversal frequency
            avg_volume = current_data['volume'].iloc[max(0, i-9):i+1].mean()
            reversal_count = 0
            for j in range(max(1, i-9), i+1):
                if (current_data['volume'].iloc[j] > 1.5 * avg_volume and 
                    ((current_data['close'].iloc[j] > current_data['close'].iloc[j-1] and 
                      current_data['close'].iloc[j-1] < current_data['close'].iloc[max(0, j-2)]) or
                     (current_data['close'].iloc[j] < current_data['close'].iloc[j-1] and 
                      current_data['close'].iloc[j-1] > current_data['close'].iloc[max(0, j-2)]))):
                    reversal_count += 1
            high_volume_reversal = reversal_count / 10.0
        else:
            volume_concentration = volume_persistence = high_volume_reversal = 0
        
        # Micro-volume clustering
        if i >= 5:
            # Consecutive high-volume bursts
            max_consecutive = 0
            current_streak = 0
            for j in range(max(1, i-4), i+1):
                if current_data['volume'].iloc[j] > 1.3 * current_data['volume'].iloc[j-1]:
                    current_streak += 1
                    max_consecutive = max(max_consecutive, current_streak)
                else:
                    current_streak = 0
            volume_bursts = max_consecutive / 5.0
            
            # Volume dry-up patterns
            recent_avg_volume = current_data['volume'].iloc[max(0, i-4):i+1].mean()
            dry_up_count = sum(1 for j in range(max(0, i-4), i+1) 
                             if current_data['volume'].iloc[j] < 0.7 * recent_avg_volume)
            dry_up_patterns = dry_up_count / 5.0
        else:
            volume_bursts = dry_up_patterns = 0
        
        # 3. Microstructure Regime Identification
        # Liquidity regime detection
        mid_price = 0.5 * (current_data['high'].iloc[i] + current_data['low'].iloc[i])
        if mid_price != 0:
            spread_proxy = (current_data['high'].iloc[i] - current_data['low'].iloc[i]) / mid_price
        else:
            spread_proxy = 0
        
        if (current_data['high'].iloc[i] - current_data['low'].iloc[i]) != 0:
            market_depth = current_data['volume'].iloc[i] / (current_data['high'].iloc[i] - current_data['low'].iloc[i])
            slippage_prob = abs(current_data['close'].iloc[i] - current_data['open'].iloc[i]) / \
                          (current_data['high'].iloc[i] - current_data['low'].iloc[i])
        else:
            market_depth = slippage_prob = 0
        
        # Order flow patterns
        if i >= 1:
            prev_range = current_data['high'].iloc[i-1] - current_data['low'].iloc[i-1]
            if prev_range != 0:
                opening_auction = (current_data['open'].iloc[i] - current_data['close'].iloc[i-1]) / prev_range
            else:
                opening_auction = 0
        else:
            opening_auction = 0
        
        current_range = current_data['high'].iloc[i] - current_data['low'].iloc[i]
        if current_range != 0:
            closing_pressure = (current_data['close'].iloc[i] - 
                              0.5 * (current_data['high'].iloc[i] + current_data['low'].iloc[i])) / current_range
            midday_consolidation = abs((current_data['high'].iloc[i] + current_data['low'].iloc[i])/2 - 
                                     (current_data['open'].iloc[i] + current_data['close'].iloc[i])/2) / current_range
        else:
            closing_pressure = midday_consolidation = 0
        
        # 4. Momentum-Volume Divergence
        if i >= 5:
            price_momentum = current_data['close'].iloc[i] - current_data['close'].iloc[i-5]
            avg_volume_recent = current_data['volume'].iloc[max(0, i-4):i+1].mean()
            
            if avg_volume_recent != 0:
                high_momentum_low_volume = price_momentum / avg_volume_recent
            else:
                high_momentum_low_volume = 0
            
            if abs(price_momentum) != 0:
                low_momentum_high_volume = avg_volume_recent / abs(price_momentum)
            else:
                low_momentum_high_volume = 0
        else:
            high_momentum_low_volume = low_momentum_high_volume = 0
        
        # Combine all components with appropriate weights
        momentum_component = (up_momentum_sustain - down_momentum_accel + reversal_resistance + 
                            close_high_strength + gap_persistence + eod_momentum) / 6.0
        
        volume_component = (volume_concentration + volume_persistence + high_volume_reversal + 
                          volume_bursts - dry_up_patterns) / 5.0
        
        microstructure_component = (-spread_proxy + market_depth - slippage_prob + 
                                  opening_auction + closing_pressure - midday_consolidation) / 6.0
        
        divergence_component = (high_momentum_low_volume - low_momentum_high_volume) / 2.0
        
        # Final factor value
        factor.iloc[i] = (0.3 * momentum_component + 
                         0.3 * volume_component + 
                         0.25 * microstructure_component + 
                         0.15 * divergence_component)
    
    return factor
