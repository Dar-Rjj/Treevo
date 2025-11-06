import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Generate novel alpha factor combining multiple technical approaches:
    - Volatility-adjusted intraday reversal
    - Dynamic breakout with efficiency confirmation
    - Accumulation-divergence factor
    - Momentum acceleration with volume clustering
    """
    result = pd.Series(index=df.index, dtype=float)
    
    for i in range(len(df)):
        if i < 20:  # Minimum window for calculations
            result.iloc[i] = 0
            continue
            
        current_data = df.iloc[:i+1]  # Only use current and past data
        
        # 1. Volatility-Adjusted Intraday Reversal
        high_close_ratio = current_data['high'].iloc[-1] / current_data['close'].iloc[-1] - 1
        low_close_ratio = current_data['low'].iloc[-1] / current_data['close'].iloc[-1] - 1
        
        high_low_range = current_data['high'].iloc[-1] - current_data['low'].iloc[-1]
        if high_low_range > 0:
            reversal_signal = (high_close_ratio - low_close_ratio) / high_low_range
        else:
            reversal_signal = 0
            
        # Volume confirmation for reversal
        if i >= 1:
            volume_change = (current_data['volume'].iloc[-1] / current_data['volume'].iloc[-2] - 1)
            reversal_signal *= np.sign(volume_change)
        
        # 2. Dynamic Breakout with Efficiency Confirmation
        support_level = current_data['low'].iloc[-20:].min()
        resistance_level = current_data['high'].iloc[-20:].max()
        
        # Price movement efficiency
        high_low_range_current = current_data['high'].iloc[-1] - current_data['low'].iloc[-1]
        if high_low_range_current > 0:
            efficiency = abs(current_data['close'].iloc[-1] - current_data['open'].iloc[-1]) / high_low_range_current
        else:
            efficiency = 0
            
        # Breakout detection
        breakout_signal = 0
        if current_data['close'].iloc[-1] > resistance_level:
            breakout_signal = efficiency
        elif current_data['close'].iloc[-1] < support_level:
            breakout_signal = -efficiency
            
        # Volume confirmation for breakout
        avg_volume = current_data['volume'].iloc[-20:].mean()
        if current_data['volume'].iloc[-1] > avg_volume:
            breakout_signal *= 1.5
        
        # 3. Accumulation-Divergence Factor
        if high_low_range_current > 0:
            position_in_range = (current_data['close'].iloc[-1] - current_data['low'].iloc[-1]) / high_low_range_current
            accumulation = (position_in_range - 0.5) * current_data['volume'].iloc[-1]
        else:
            accumulation = 0
            
        # Trend divergence calculation
        if len(current_data) >= 10:
            time_index = np.arange(len(current_data.iloc[-10:]))
            price_trend = np.corrcoef(time_index, current_data['close'].iloc[-10:])[0,1]
            volume_trend = np.corrcoef(time_index, current_data['volume'].iloc[-10:])[0,1]
            
            divergence_signal = 0
            if accumulation > 0 and price_trend < 0:
                divergence_signal = accumulation * abs(price_trend)
            elif accumulation < 0 and price_trend > 0:
                divergence_signal = accumulation * abs(price_trend)
                
            # Scale by absolute price change
            if i >= 1:
                price_change = abs(current_data['close'].iloc[-1] - current_data['close'].iloc[-2])
                divergence_signal *= price_change
        else:
            divergence_signal = 0
        
        # 4. Momentum Acceleration with Volume Clustering
        if len(current_data) >= 5:
            # Price momentum
            momentum_5d = current_data['close'].iloc[-1] / current_data['close'].iloc[-5] - 1
            momentum_3d = current_data['close'].iloc[-1] / current_data['close'].iloc[-3] - 1
            
            # Momentum acceleration
            acceleration = momentum_3d - momentum_5d
            
            # Volume clustering
            volume_percentile = (current_data['volume'].iloc[-5:] > current_data['volume'].iloc[-20:].quantile(0.7)).mean()
            
            # Combine acceleration with volume clusters
            if volume_percentile > 0.6:
                momentum_signal = acceleration * volume_percentile
                
                # Return persistence check
                returns = current_data['close'].pct_change().iloc[-5:]
                if len(returns) >= 3:
                    autocorr = returns.autocorr()
                    if not np.isnan(autocorr):
                        momentum_signal *= (1 + abs(autocorr))
            else:
                momentum_signal = acceleration * 0.5  # Weaken signal during low volume
        else:
            momentum_signal = 0
        
        # Combine all signals with weights
        final_signal = (
            0.3 * reversal_signal +
            0.3 * breakout_signal +
            0.2 * divergence_signal +
            0.2 * momentum_signal
        )
        
        result.iloc[i] = final_signal
    
    return result
