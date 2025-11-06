import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Generate alpha factor combining multiple technical indicators:
    - Multi-timeframe momentum-volume divergence
    - Volatility-normalized momentum
    - Volume-weighted breakout strength
    - Price-volume efficiency ratio
    - Gap-fill momentum with volume
    """
    result = pd.Series(index=df.index, dtype=float)
    
    for i in range(5, len(df)):
        if i < 5:
            continue
            
        current_data = df.iloc[i]
        prev_data = df.iloc[:i+1]
        
        # 1. Multi-Timeframe Momentum-Volume Divergence
        if i >= 5:
            # 3-Day components
            close_3d_ago = df.iloc[i-3]['close'] if i >= 3 else np.nan
            volume_3d_ago = df.iloc[i-3]['volume'] if i >= 3 else np.nan
            
            price_return_3d = (current_data['close'] - close_3d_ago) / close_3d_ago if close_3d_ago > 0 else 0
            volume_change_3d = (current_data['volume'] - volume_3d_ago) / volume_3d_ago if volume_3d_ago > 0 else 0
            divergence_3d = price_return_3d * volume_change_3d
            
            # 5-Day components
            close_5d_ago = df.iloc[i-5]['close']
            volume_5d_ago = df.iloc[i-5]['volume']
            
            price_return_5d = (current_data['close'] - close_5d_ago) / close_5d_ago if close_5d_ago > 0 else 0
            volume_change_5d = (current_data['volume'] - volume_5d_ago) / volume_5d_ago if volume_5d_ago > 0 else 0
            divergence_5d = price_return_5d * volume_change_5d
            
            momentum_divergence = divergence_3d + divergence_5d
        else:
            momentum_divergence = 0
        
        # 2. Volatility-Normalized Momentum
        if i >= 5:
            price_momentum_5d = (current_data['close'] - close_5d_ago) / close_5d_ago if close_5d_ago > 0 else 0
            
            # 5-day volatility (high-low range)
            recent_highs = df.iloc[i-4:i+1]['high']
            recent_lows = df.iloc[i-4:i+1]['low']
            volatility_5d = (recent_highs.max() - recent_lows.min()) / close_5d_ago if close_5d_ago > 0 else 1
            
            normalized_momentum = price_momentum_5d / volatility_5d if volatility_5d > 0 else 0
        else:
            normalized_momentum = 0
        
        # 3. Volume-Weighted Breakout Strength
        if i >= 5:
            # 5-day high
            five_day_high = df.iloc[i-4:i+1]['high'].max()
            breakout = (current_data['close'] - five_day_high) / five_day_high if five_day_high > 0 else 0
            
            # Volume ratio
            volume_3d_ago = df.iloc[i-3]['volume'] if i >= 3 else current_data['volume']
            volume_ratio = current_data['volume'] / volume_3d_ago if volume_3d_ago > 0 else 1
            
            breakout_strength = breakout * volume_ratio
        else:
            breakout_strength = 0
        
        # 4. Price-Volume Efficiency Ratio
        daily_range = current_data['high'] - current_data['low']
        net_move = abs(current_data['close'] - current_data['open'])
        price_efficiency = net_move / daily_range if daily_range > 0 else 0
        
        # Volume confirmation (5-day average)
        if i >= 5:
            recent_volumes = df.iloc[i-4:i+1]['volume']
            volume_avg_5d = recent_volumes.mean()
            volume_confirmation = current_data['volume'] / volume_avg_5d if volume_avg_5d > 0 else 1
        else:
            volume_confirmation = 1
        
        efficiency_ratio = price_efficiency * volume_confirmation
        
        # 5. Gap-Fill Momentum with Volume
        prev_close = df.iloc[i-1]['close']
        gap_size = abs(current_data['open'] - prev_close)
        intraday_momentum = (current_data['close'] - current_data['open']) / current_data['open'] if current_data['open'] > 0 else 0
        
        prev_volume = df.iloc[i-1]['volume']
        daily_volume_ratio = current_data['volume'] / prev_volume if prev_volume > 0 else 1
        
        gap_fill_factor = intraday_momentum * daily_volume_ratio / gap_size if gap_size > 0 else 0
        
        # Combine all factors with equal weighting
        combined_factor = (
            momentum_divergence +
            normalized_momentum +
            breakout_strength +
            efficiency_ratio +
            gap_fill_factor
        )
        
        result.iloc[i] = combined_factor
    
    # Fill initial NaN values with 0
    result = result.fillna(0)
    
    return result
