import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Initialize result series
    result = pd.Series(index=data.index, dtype=float)
    
    # Calculate all required components
    for i in range(len(data)):
        if i < 20:  # Need at least 20 days of data
            result.iloc[i] = 0
            continue
            
        current_data = data.iloc[:i+1]
        
        # Momentum Reversal with Volatility Dampening
        if i >= 5:
            # 5-day momentum
            momentum = (current_data['close'].iloc[i-1] / current_data['close'].iloc[i-5] - 1)
            
            # 20-day volatility (using returns from t-20 to t-1)
            returns_20d = current_data['close'].iloc[i-20:i].pct_change().dropna()
            volatility = returns_20d.std()
            
            # Volume trend (slope of volume regression)
            volumes_5d = current_data['volume'].iloc[i-5:i].values
            if len(volumes_5d) == 5:
                x = np.arange(5)
                slope = np.polyfit(x, volumes_5d, 1)[0]
                volume_sign = 1 if slope > 0 else -1
            else:
                volume_sign = 1
            
            # Dampened momentum
            if volatility > 0:
                dampened_momentum = momentum / volatility * np.sqrt(5) * volume_sign
            else:
                dampened_momentum = 0
        else:
            dampened_momentum = 0
        
        # Price-Level Relative Strength with Volume Acceleration
        if i >= 20:
            # Relative price position
            high_20d = current_data['high'].iloc[i-20:i].max()
            low_20d = current_data['low'].iloc[i-20:i].min()
            current_close = current_data['close'].iloc[i-1]
            
            if high_20d != low_20d:
                rel_position = (current_close - low_20d) / (high_20d - low_20d)
            else:
                rel_position = 0.5
            
            # Volume acceleration
            if i >= 10:
                recent_vol = current_data['volume'].iloc[i-5:i].mean()
                prev_vol = current_data['volume'].iloc[i-10:i-5].mean()
                vol_acceleration = recent_vol / prev_vol if prev_vol > 0 else 1
            else:
                vol_acceleration = 1
            
            # Combined signal with cubic root transformation
            price_strength = np.cbrt(rel_position * vol_acceleration)
        else:
            price_strength = 0
        
        # Intraday Persistence with Overnight Gap
        if i >= 5:
            # Intraday strength consistency (variance of daily ranges)
            daily_ranges = []
            overnight_gaps = []
            
            for j in range(1, 6):
                if i-j >= 0:
                    daily_range = (current_data['high'].iloc[i-j] - current_data['low'].iloc[i-j]) / current_data['open'].iloc[i-j]
                    daily_ranges.append(daily_range)
                    
                    if i-j-1 >= 0:
                        gap = abs(current_data['open'].iloc[i-j] / current_data['close'].iloc[i-j-1] - 1)
                        overnight_gaps.append(gap)
            
            if len(daily_ranges) >= 3:
                intraday_consistency = np.var(daily_ranges)
                avg_gap = np.mean(overnight_gaps) if overnight_gaps else 0
                
                if intraday_consistency * avg_gap > 0:
                    intraday_factor = 1 / (intraday_consistency * avg_gap)
                else:
                    intraday_factor = 0
            else:
                intraday_factor = 0
        else:
            intraday_factor = 0
        
        # Volume-Weighted Price Change Skewness
        if i >= 20:
            # Volume-weighted returns
            returns = current_data['close'].iloc[i-20:i].pct_change().dropna()
            volumes = current_data['volume'].iloc[i-19:i].values
            
            if len(returns) == len(volumes) and len(returns) >= 1:
                vw_return = np.sum(volumes * returns.values) / np.sum(volumes)
                
                # Skewness of returns
                if len(returns) >= 3:
                    skewness = returns.skew()
                else:
                    skewness = 0
                
                # Momentum filter (5-day return sign)
                if i >= 5:
                    mom_filter = np.sign(current_data['close'].iloc[i-1] / current_data['close'].iloc[i-5] - 1)
                else:
                    mom_filter = 1
                
                volume_skew_factor = vw_return * skewness * mom_filter
            else:
                volume_skew_factor = 0
        else:
            volume_skew_factor = 0
        
        # Efficiency Ratio with Volume Clustering
        if i >= 20:
            # Market efficiency
            price_changes = []
            for j in range(i-19, i):
                price_changes.append(abs(current_data['close'].iloc[j] - current_data['close'].iloc[j-1]))
            
            net_change = abs(current_data['close'].iloc[i-1] - current_data['close'].iloc[i-20])
            total_variation = sum(price_changes)
            
            efficiency = net_change / total_variation if total_variation > 0 else 0
            
            # Volume clustering (autocorrelation at lag 1)
            volumes_20d = current_data['volume'].iloc[i-20:i].values
            if len(volumes_20d) >= 3:
                vol_autocorr = np.corrcoef(volumes_20d[:-1], volumes_20d[1:])[0,1]
                if np.isnan(vol_autocorr):
                    vol_autocorr = 0
            else:
                vol_autocorr = 0
            
            # Recent price trend for direction
            if i >= 5:
                recent_trend = np.sign(current_data['close'].iloc[i-1] - current_data['close'].iloc[i-5])
            else:
                recent_trend = 1
            
            efficiency_factor = efficiency * (1 + vol_autocorr) * recent_trend
        else:
            efficiency_factor = 0
        
        # Combine all factors with equal weighting
        combined_factor = (
            dampened_momentum + 
            price_strength + 
            intraday_factor + 
            volume_skew_factor + 
            efficiency_factor
        )
        
        result.iloc[i] = combined_factor
    
    return result
