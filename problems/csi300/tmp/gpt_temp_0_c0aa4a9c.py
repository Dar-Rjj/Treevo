import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Volatility-Adjusted Multi-Timeframe Momentum with Volume Confirmation
    """
    # Calculate daily returns
    daily_returns = df['close'].pct_change()
    
    # Initialize result series
    alpha = pd.Series(index=df.index, dtype=float)
    
    for i in range(20, len(df)):
        current_data = df.iloc[:i+1]
        
        # Volatility-Normalized Momentum Calculation
        # Short-term momentum (3-day)
        if i >= 3:
            st_return = (current_data['close'].iloc[i] / current_data['close'].iloc[i-3]) - 1
            st_vol = daily_returns.iloc[i-2:i+1].std()
            st_momentum = st_return / (st_vol + 1e-8) if st_vol > 0 else 0
        else:
            st_momentum = 0
        
        # Medium-term momentum (10-day)
        if i >= 10:
            mt_return = (current_data['close'].iloc[i] / current_data['close'].iloc[i-10]) - 1
            mt_vol = daily_returns.iloc[i-9:i+1].std()
            mt_momentum = mt_return / (mt_vol + 1e-8) if mt_vol > 0 else 0
        else:
            mt_momentum = 0
        
        # Long-term momentum (20-day)
        if i >= 20:
            lt_return = (current_data['close'].iloc[i] / current_data['close'].iloc[i-20]) - 1
            lt_vol = daily_returns.iloc[i-19:i+1].std()
            lt_momentum = lt_return / (lt_vol + 1e-8) if lt_vol > 0 else 0
        else:
            lt_momentum = 0
        
        # Volume-Price Alignment Analysis
        volume_alignment_score = 0
        volume_weights_sum = 0
        
        # Short-term volume alignment
        if i >= 3:
            st_volume_slope = (current_data['volume'].iloc[i] - current_data['volume'].iloc[i-3]) / 3
            st_volume_mag = abs(st_volume_slope)
            st_alignment = 1 if (st_momentum * st_volume_slope) > 0 else -1
            volume_alignment_score += st_alignment * st_volume_mag
            volume_weights_sum += st_volume_mag
        
        # Medium-term volume alignment
        if i >= 10:
            mt_volume_slope = (current_data['volume'].iloc[i] - current_data['volume'].iloc[i-10]) / 10
            mt_volume_mag = abs(mt_volume_slope)
            mt_alignment = 1 if (mt_momentum * mt_volume_slope) > 0 else -1
            volume_alignment_score += mt_alignment * mt_volume_mag
            volume_weights_sum += mt_volume_mag
        
        # Long-term volume alignment
        if i >= 20:
            lt_volume_slope = (current_data['volume'].iloc[i] - current_data['volume'].iloc[i-20]) / 20
            lt_volume_mag = abs(lt_volume_slope)
            lt_alignment = 1 if (lt_momentum * lt_volume_slope) > 0 else -1
            volume_alignment_score += lt_alignment * lt_volume_mag
            volume_weights_sum += lt_volume_mag
        
        # Normalize volume alignment score
        if volume_weights_sum > 0:
            volume_alignment_score /= volume_weights_sum
        else:
            volume_alignment_score = 0
        
        # Market Regime Adaptation
        if i >= 20:
            regime_vol = daily_returns.iloc[i-19:i+1].std()
            median_vol = daily_returns.iloc[max(0, i-59):i+1].std()  # Use 60-day window for median estimation
            
            if regime_vol > median_vol * 1.2:
                regime_multiplier = 1.5
            elif regime_vol < median_vol * 0.8:
                regime_multiplier = 0.7
            else:
                regime_multiplier = 1.0
        else:
            regime_multiplier = 1.0
        
        # Mean Reversion Enhancement
        reversal_component = 0
        if i >= 20:
            ma_20 = current_data['close'].iloc[i-19:i+1].mean()
            price_deviation = (current_data['close'].iloc[i] - ma_20) / ma_20
            vol_20 = daily_returns.iloc[i-19:i+1].std()
            
            if vol_20 > 0:
                normalized_deviation = price_deviation / vol_20
                if abs(normalized_deviation) > 2:
                    reversal_component = -np.sign(price_deviation) * abs(normalized_deviation)
        
        # Composite Alpha Construction
        # Combine normalized momentum components
        momentum_component = (0.45 * st_momentum + 
                             0.35 * mt_momentum + 
                             0.20 * lt_momentum)
        
        # Apply volume alignment
        momentum_component *= (1 + volume_alignment_score)
        
        # Apply regime scaling
        momentum_component *= regime_multiplier
        
        # Final blend: 80% momentum, 20% reversal
        final_alpha = 0.8 * momentum_component + 0.2 * reversal_component
        
        alpha.iloc[i] = final_alpha
    
    # Fill initial NaN values with 0
    alpha = alpha.fillna(0)
    
    return alpha
