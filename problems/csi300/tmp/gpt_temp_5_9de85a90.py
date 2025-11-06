import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Initialize result series
    alpha = pd.Series(index=df.index, dtype=float)
    
    for i in range(20, len(df)):
        current_data = df.iloc[:i+1].copy()
        
        # Intraday Price Structure Analysis
        # Opening Momentum
        if i >= 1:
            gap_magnitude = (current_data['open'].iloc[i] - current_data['close'].iloc[i-1]) / current_data['close'].iloc[i-1]
            gap_direction = np.sign(current_data['open'].iloc[i] - current_data['close'].iloc[i-1])
        else:
            gap_magnitude = 0
            gap_direction = 0
        
        # Session Momentum Components
        morning_momentum = (current_data['high'].iloc[i] - current_data['open'].iloc[i]) / current_data['open'].iloc[i]
        afternoon_momentum = (current_data['close'].iloc[i] - current_data['low'].iloc[i]) / current_data['low'].iloc[i]
        
        intraday_range = current_data['high'].iloc[i] - current_data['low'].iloc[i]
        if intraday_range > 0:
            intraday_range_efficiency = abs(current_data['close'].iloc[i] - current_data['open'].iloc[i]) / intraday_range
        else:
            intraday_range_efficiency = 0
        
        # Rejection Strength Evaluation
        if intraday_range > 0:
            upper_rejection = (current_data['high'].iloc[i] - current_data['close'].iloc[i]) / intraday_range
            lower_rejection = (current_data['close'].iloc[i] - current_data['low'].iloc[i]) / intraday_range
        else:
            upper_rejection = 0
            lower_rejection = 0
        
        # Rejection classification
        if upper_rejection > 0.7:
            rejection_class = -1.5  # Strong upper rejection
        elif lower_rejection > 0.7:
            rejection_class = 1.5   # Strong lower rejection
        else:
            rejection_class = 0.5 * np.sign(current_data['close'].iloc[i] - current_data['open'].iloc[i])
        
        # Multi-Timeframe Momentum Acceleration
        if i >= 3:
            short_term_momentum = current_data['close'].iloc[i] / current_data['close'].iloc[i-3] - 1
        else:
            short_term_momentum = 0
            
        if i >= 8:
            medium_term_momentum = current_data['close'].iloc[i] / current_data['close'].iloc[i-8] - 1
        else:
            medium_term_momentum = 0
            
        if i >= 20:
            long_term_momentum = current_data['close'].iloc[i] / current_data['close'].iloc[i-20] - 1
        else:
            long_term_momentum = 0
        
        # Acceleration-Divergence Score
        if abs(medium_term_momentum) > 1e-6:
            acceleration = (short_term_momentum - medium_term_momentum) / abs(medium_term_momentum)
        else:
            acceleration = 0
            
        divergence = abs(short_term_momentum - long_term_momentum)
        acceleration_divergence_score = acceleration * divergence
        
        # Volume-Price Efficiency & Alignment
        if i >= 1:
            volume_momentum_ratio = current_data['volume'].iloc[i] / current_data['volume'].iloc[i-1]
        else:
            volume_momentum_ratio = 1
        
        # Volume Asymmetry (5-day ratio of up-volume/down-volume)
        if i >= 5:
            recent_data = current_data.iloc[i-4:i+1]
            up_volume = recent_data[recent_data['close'] > recent_data['close'].shift(1)]['volume'].sum()
            down_volume = recent_data[recent_data['close'] < recent_data['close'].shift(1)]['volume'].sum()
            if down_volume > 0:
                volume_asymmetry = up_volume / down_volume
            else:
                volume_asymmetry = 1
        else:
            volume_asymmetry = 1
        
        # Volume-Price Alignment (5-day volume-weighted return / 5-day raw return)
        if i >= 5:
            recent_data = current_data.iloc[i-4:i+1]
            volume_weighted_return = (recent_data['close'] * recent_data['volume']).sum() / recent_data['volume'].sum()
            raw_return = recent_data['close'].iloc[-1] / recent_data['close'].iloc[0] - 1
            if abs(raw_return) > 1e-6:
                volume_price_alignment = (volume_weighted_return / recent_data['close'].iloc[0] - 1) / raw_return
            else:
                volume_price_alignment = 1
        else:
            volume_price_alignment = 1
        
        # Range Breakout & Volatility Analysis
        if i >= 1:
            current_range = current_data['high'].iloc[i] - current_data['low'].iloc[i]
            prev_range = current_data['high'].iloc[i-1] - current_data['low'].iloc[i-1]
            if prev_range > 0:
                range_expansion = current_range / prev_range
            else:
                range_expansion = 1
        else:
            range_expansion = 1
        
        # Breakout Direction
        if i >= 1:
            upward_breakout = (current_data['close'].iloc[i] > current_data['close'].iloc[i-1]) and (current_data['high'].iloc[i] > current_data['high'].iloc[i-1])
            downward_breakout = (current_data['close'].iloc[i] < current_data['close'].iloc[i-1]) and (current_data['low'].iloc[i] < current_data['low'].iloc[i-1])
            
            if upward_breakout:
                breakout_direction = 1
            elif downward_breakout:
                breakout_direction = -1
            else:
                breakout_direction = 0
        else:
            breakout_direction = 0
        
        # True Range calculation for volatility
        if i >= 1:
            tr1 = current_data['high'].iloc[i] - current_data['low'].iloc[i]
            tr2 = abs(current_data['high'].iloc[i] - current_data['close'].iloc[i-1])
            tr3 = abs(current_data['low'].iloc[i] - current_data['close'].iloc[i-1])
            true_range = max(tr1, tr2, tr3)
        else:
            true_range = current_data['high'].iloc[i] - current_data['low'].iloc[i]
        
        # Volatility Ratio: 10-day std(True Range) / 60-day std(True Range)
        if i >= 60:
            # Calculate True Range for recent periods
            tr_values = []
            for j in range(max(0, i-59), i+1):
                if j >= 1:
                    tr1 = current_data['high'].iloc[j] - current_data['low'].iloc[j]
                    tr2 = abs(current_data['high'].iloc[j] - current_data['close'].iloc[j-1])
                    tr3 = abs(current_data['low'].iloc[j] - current_data['close'].iloc[j-1])
                    tr_values.append(max(tr1, tr2, tr3))
                else:
                    tr_values.append(current_data['high'].iloc[j] - current_data['low'].iloc[j])
            
            tr_10day_std = np.std(tr_values[-10:]) if len(tr_values[-10:]) >= 2 else 1
            tr_60day_std = np.std(tr_values) if len(tr_values) >= 2 else 1
            
            if tr_60day_std > 1e-6:
                volatility_ratio = tr_10day_std / tr_60day_std
            else:
                volatility_ratio = 1
        else:
            volatility_ratio = 1
        
        # Composite Alpha Generation
        # Intraday Momentum Score
        intraday_momentum_score = (morning_momentum + afternoon_momentum + gap_magnitude * gap_direction) / 3
        
        # Momentum Acceleration Score
        momentum_acceleration_score = acceleration_divergence_score * range_expansion
        
        # Volume Efficiency Factor
        volume_efficiency_factor = intraday_range_efficiency * volume_asymmetry * volume_price_alignment
        
        # Breakout-Rejection Signal
        breakout_component = breakout_direction * (1 + abs(range_expansion - 1))
        rejection_component = rejection_class
        breakout_rejection_signal = breakout_component + rejection_component
        
        # Final Alpha
        final_alpha = (intraday_momentum_score + momentum_acceleration_score) * volume_efficiency_factor * breakout_rejection_signal * (1 + volatility_ratio)
        
        alpha.iloc[i] = final_alpha
    
    # Fill NaN values with 0 for early periods
    alpha = alpha.fillna(0)
    
    return alpha
