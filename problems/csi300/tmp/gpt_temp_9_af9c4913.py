import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Generate alpha factor combining multiple technical indicators with volume and volatility adjustments
    """
    # Copy dataframe to avoid modifying original
    data = df.copy()
    
    # Initialize result series
    alpha = pd.Series(index=data.index, dtype=float)
    
    # Calculate rolling windows
    for i in range(len(data)):
        if i < 10:  # Need at least 10 days of data
            alpha.iloc[i] = 0
            continue
            
        # Extract current and historical data
        current = data.iloc[i]
        hist = data.iloc[max(0, i-10):i+1]
        
        # 1. Multi-Timeframe Volatility-Normalized Momentum
        # Short-term momentum (3-day)
        if i >= 3:
            mom_3d = (current['close'] - data.iloc[i-3]['close']) / data.iloc[i-3]['close']
        else:
            mom_3d = 0
            
        # Medium-term momentum (10-day)
        mom_10d = (current['close'] - data.iloc[i-10]['close']) / data.iloc[i-10]['close']
        
        # Calculate ATR for volatility adjustment
        def calculate_atr(window):
            atr_values = []
            for j in range(1, len(window)):
                high_low = window.iloc[j]['high'] - window.iloc[j]['low']
                high_close = abs(window.iloc[j]['high'] - window.iloc[j-1]['close'])
                low_close = abs(window.iloc[j]['low'] - window.iloc[j-1]['close'])
                atr_values.append(max(high_low, high_close, low_close))
            return np.mean(atr_values) if atr_values else 1.0
        
        atr_3d = calculate_atr(hist.iloc[-3:]) if len(hist) >= 3 else 1.0
        atr_10d = calculate_atr(hist.iloc[-10:]) if len(hist) >= 10 else 1.0
        
        # Volume-weighted momentum blend
        vol_5d_avg = np.mean([data.iloc[j]['volume'] for j in range(max(0, i-4), i+1)])
        vol_weight = current['volume'] / vol_5d_avg if vol_5d_avg > 0 else 1.0
        
        momentum_factor = (mom_3d / atr_3d + mom_10d / atr_10d) * vol_weight
        
        # 2. Price Acceleration with Volume Confirmation
        if i >= 5:
            ret_3d = (current['close'] - data.iloc[i-3]['close']) / data.iloc[i-3]['close']
            ret_5d = (current['close'] - data.iloc[i-5]['close']) / data.iloc[i-5]['close']
            ret_10d = mom_10d
            
            accel_short = ret_3d - ret_5d
            accel_medium = ret_5d - ret_10d
            
            accel_factor = (accel_short * accel_medium) * vol_weight
        else:
            accel_factor = 0
            
        # 3. Volatility-Adjusted Breakout System
        if i >= 5:
            # 5-day breakout
            high_5d = max([data.iloc[j]['high'] for j in range(max(0, i-4), i+1)])
            breakout_5d = (current['close'] - high_5d) / high_5d if high_5d > 0 else 0
            
            # 10-day breakout
            high_10d = max([data.iloc[j]['high'] for j in range(max(0, i-9), i+1)])
            breakout_10d = (current['close'] - high_10d) / high_10d if high_10d > 0 else 0
            
            # Recent volatility ratio
            recent_vol = np.mean([data.iloc[j]['high'] - data.iloc[j]['low'] for j in range(max(0, i-4), i+1)])
            prev_vol = np.mean([data.iloc[j]['high'] - data.iloc[j]['low'] for j in range(max(0, i-9), max(i-4, 0))])
            vol_ratio = recent_vol / prev_vol if prev_vol > 0 else 1.0
            
            breakout_factor = (0.6 * breakout_5d + 0.4 * breakout_10d) * vol_ratio * vol_weight
        else:
            breakout_factor = 0
            
        # 4. Gap Fill Momentum with Volatility Scaling
        if i >= 1:
            gap_magnitude = abs(current['open'] - data.iloc[i-1]['close'])
            prev_volatility = data.iloc[i-1]['high'] - data.iloc[i-1]['low']
            fill_direction = 1 if current['open'] > data.iloc[i-1]['close'] else -1
            fill_progress = (current['close'] - current['open']) * fill_direction
            
            vol_3d_avg = np.mean([data.iloc[j]['volume'] for j in range(max(0, i-2), i+1)])
            vol_weight_gap = current['volume'] / vol_3d_avg if vol_3d_avg > 0 else 1.0
            
            if gap_magnitude > 0 and prev_volatility > 0:
                gap_factor = (fill_progress / gap_magnitude) * (gap_magnitude / prev_volatility) * vol_weight_gap
            else:
                gap_factor = 0
        else:
            gap_factor = 0
            
        # 5. Multi-timeframe Support/Resistance Efficiency
        if i >= 5:
            support_5d = min([data.iloc[j]['low'] for j in range(max(0, i-4), i+1)])
            support_10d = min([data.iloc[j]['low'] for j in range(max(0, i-9), i+1)])
            resistance_10d = max([data.iloc[j]['high'] for j in range(max(0, i-9), i+1)])
            
            if resistance_10d - support_5d > 0:
                support_eff = (current['close'] - support_5d) / (resistance_10d - support_5d)
                resistance_eff = (resistance_10d - current['close']) / (resistance_10d - support_5d)
                level_factor = (support_eff - resistance_eff) * vol_weight
            else:
                level_factor = 0
        else:
            level_factor = 0
            
        # 6. Amount-Based Price Impact with Volatility
        if i >= 5:
            current_turnover = current['amount'] / current['volume'] if current['volume'] > 0 else 0
            avg_amount_5d = np.mean([data.iloc[j]['amount'] for j in range(max(0, i-4), i+1)])
            avg_volume_5d = np.mean([data.iloc[j]['volume'] for j in range(max(0, i-4), i+1)])
            avg_turnover_5d = avg_amount_5d / avg_volume_5d if avg_volume_5d > 0 else 1.0
            
            daily_return = (current['close'] - data.iloc[i-1]['close']) / data.iloc[i-1]['close'] if i >= 1 else 0
            intraday_range = current['high'] - current['low']
            
            prev_range_avg = np.mean([data.iloc[j]['high'] - data.iloc[j]['low'] for j in range(max(0, i-4), i)])
            vol_adjust = intraday_range / prev_range_avg if prev_range_avg > 0 else 1.0
            
            impact_factor = daily_return * (current_turnover / avg_turnover_5d) * vol_adjust if avg_turnover_5d > 0 else 0
        else:
            impact_factor = 0
            
        # Combine all factors with equal weighting
        alpha.iloc[i] = (
            momentum_factor + 
            accel_factor + 
            breakout_factor + 
            gap_factor + 
            level_factor + 
            impact_factor
        )
    
    return alpha
