import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Volatility Regime Adaptive Momentum factor
    """
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Initialize output series
    alpha = pd.Series(index=df.index, dtype=float)
    
    # Calculate required periods
    for i in range(20, len(df)):
        current_data = df.iloc[:i+1]
        
        # 1. Volatility Regime Detection
        # Short-term volatility estimation
        if i >= 4:
            high_5d = current_data['high'].iloc[i-4:i+1].max()
            low_5d = current_data['low'].iloc[i-4:i+1].min()
            hl_range_5d = (high_5d - low_5d) / current_data['close'].iloc[i-4]
            
            if i >= 5:
                close_vol_5d = abs(current_data['close'].iloc[i] / current_data['close'].iloc[i-5] - 1)
                short_term_vol = (hl_range_5d + close_vol_5d) / 2
            else:
                short_term_vol = hl_range_5d
        else:
            short_term_vol = 0
        
        # Historical volatility baseline (20-day)
        if i >= 19:
            returns_20d = current_data['close'].iloc[i-19:i+1].pct_change().dropna()
            hist_vol = returns_20d.std()
        else:
            hist_vol = 0
        
        # Volatility regime classification
        if hist_vol > 0 and short_term_vol > 0:
            vol_ratio = short_term_vol / hist_vol
            if vol_ratio > 1.5:
                regime = 'high'
            elif vol_ratio < 0.7:
                regime = 'low'
            else:
                regime = 'normal'
        else:
            regime = 'normal'
        
        # 2. Adaptive Momentum Calculation
        momentum_score = 0
        
        if regime == 'high':
            # High volatility regime - short lookbacks
            if i >= 3:
                mom_3d = current_data['close'].iloc[i] / current_data['close'].iloc[i-3] - 1
                momentum_score += mom_3d * 0.6
            if i >= 7:
                mom_7d = current_data['close'].iloc[i] / current_data['close'].iloc[i-7] - 1
                momentum_score += mom_7d * 0.4
        
        elif regime == 'low':
            # Low volatility regime - longer lookbacks
            if i >= 10:
                mom_10d = current_data['close'].iloc[i] / current_data['close'].iloc[i-10] - 1
                momentum_score += mom_10d * 0.5
            if i >= 15:
                mom_15d = current_data['close'].iloc[i] / current_data['close'].iloc[i-15] - 1
                momentum_score += mom_15d * 0.5
        
        else:  # normal regime
            if i >= 5:
                mom_5d = current_data['close'].iloc[i] / current_data['close'].iloc[i-5] - 1
                momentum_score += mom_5d * 0.5
            if i >= 12:
                mom_12d = current_data['close'].iloc[i] / current_data['close'].iloc[i-12] - 1
                momentum_score += mom_12d * 0.5
        
        # Momentum quality assessment
        if regime == 'high' and i >= 3:
            lookback = 3
        elif regime == 'low' and i >= 10:
            lookback = 10
        elif regime == 'normal' and i >= 5:
            lookback = 5
        else:
            lookback = 0
        
        efficiency_ratio = 1.0
        if lookback > 0:
            # Calculate actual price change
            actual_change = abs(current_data['close'].iloc[i] - current_data['close'].iloc[i-lookback])
            
            # Calculate minimum price path
            daily_returns = []
            for j in range(lookback):
                if i-j-1 >= 0:
                    ret = abs(current_data['close'].iloc[i-j] / current_data['close'].iloc[i-j-1] - 1)
                    daily_returns.append(ret)
            
            if daily_returns:
                min_path = sum(daily_returns) * current_data['close'].iloc[i-lookback]
                if min_path > 0:
                    efficiency_ratio = actual_change / min_path
        
        # Apply efficiency ratio to momentum
        momentum_score *= efficiency_ratio
        
        # 3. Volume-Volatility Interaction
        volume_factor = 0
        
        if regime == 'high' and i >= 19:
            # Volume spike indicator
            avg_volume_20d = current_data['volume'].iloc[i-19:i+1].mean()
            current_volume = current_data['volume'].iloc[i]
            volume_spike = current_volume / avg_volume_20d if avg_volume_20d > 0 else 1
            
            if volume_spike > 1.2:
                volume_factor = 0.3
            elif volume_spike < 0.8:
                volume_factor = -0.1
        
        elif regime == 'low' and i >= 19:
            # Volume compression/breakout detection
            avg_volume_20d = current_data['volume'].iloc[i-19:i+1].mean()
            current_volume = current_data['volume'].iloc[i]
            volume_ratio = current_volume / avg_volume_20d if avg_volume_20d > 0 else 1
            
            if volume_ratio > 1.3:
                volume_factor = 0.4  # Volume breakout
            elif volume_ratio < 0.7:
                volume_factor = -0.2  # Volume compression
        
        else:  # normal regime
            if i >= 5:
                # Traditional volume confirmation
                volume_trend = current_data['volume'].iloc[i] / current_data['volume'].iloc[i-5] - 1
                price_trend = current_data['close'].iloc[i] / current_data['close'].iloc[i-5] - 1
                
                if volume_trend * price_trend > 0:  # Same direction
                    volume_factor = 0.2
                else:
                    volume_factor = -0.1
        
        # 4. Final Alpha Generation
        # Combine momentum and volume factors
        if regime == 'high':
            # Emphasize recent momentum in high volatility
            final_alpha = momentum_score * 0.8 + volume_factor * 0.2
        elif regime == 'low':
            # More balanced approach in low volatility
            final_alpha = momentum_score * 0.6 + volume_factor * 0.4
        else:
            # Standard weighting in normal regime
            final_alpha = momentum_score * 0.7 + volume_factor * 0.3
        
        # Scale by volatility level for signal strength calibration
        if hist_vol > 0:
            final_alpha /= (hist_vol + 0.01)  # Add small constant to avoid division by zero
        
        alpha.iloc[i] = final_alpha
    
    # Fill initial NaN values with 0
    alpha = alpha.fillna(0)
    
    return alpha
