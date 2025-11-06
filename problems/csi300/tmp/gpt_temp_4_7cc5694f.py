import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Short-term Momentum Structure
    # Acceleration Pattern: (Close[t] - Close[t-2]) - 2*(Close[t-1] - Close[t-3])
    data['acceleration'] = (data['close'] - data['close'].shift(2)) - 2 * (data['close'].shift(1) - data['close'].shift(3))
    
    # Momentum Persistence: Count of consecutive up/down closes in last 5 days
    def count_consecutive_direction(series, window=5):
        direction = np.sign(series.diff())
        consecutive_count = pd.Series(index=series.index, dtype=float)
        
        for i in range(len(series)):
            if i < window:
                consecutive_count.iloc[i] = 0
                continue
                
            window_data = direction.iloc[i-window+1:i+1]
            current_dir = window_data.iloc[-1]
            
            if current_dir == 0:
                consecutive_count.iloc[i] = 0
            else:
                count = 1
                for j in range(len(window_data)-2, -1, -1):
                    if window_data.iloc[j] == current_dir:
                        count += 1
                    else:
                        break
                consecutive_count.iloc[i] = count * current_dir
        
        return consecutive_count
    
    data['momentum_persistence'] = count_consecutive_direction(data['close'])
    
    # Medium-term Trend Analysis
    # Trend Strength: 10-day price range / 20-day price range
    data['range_10d'] = data['high'].rolling(window=10).max() - data['low'].rolling(window=10).min()
    data['range_20d'] = data['high'].rolling(window=20).max() - data['low'].rolling(window=20).min()
    data['trend_strength'] = data['range_10d'] / data['range_20d']
    
    # Swing Point Efficiency: Distance from current price to nearest swing high/low
    def swing_point_efficiency(high_series, low_series, close_series, window=5):
        swing_eff = pd.Series(index=close_series.index, dtype=float)
        
        for i in range(len(close_series)):
            if i < window * 2:
                swing_eff.iloc[i] = 0
                continue
            
            # Look for swing highs and lows in the window
            current_window_high = high_series.iloc[i-window:i+1]
            current_window_low = low_series.iloc[i-window:i+1]
            
            # Find nearest swing high (local maximum)
            swing_highs = []
            for j in range(1, len(current_window_high)-1):
                if (current_window_high.iloc[j] > current_window_high.iloc[j-1] and 
                    current_window_high.iloc[j] > current_window_high.iloc[j+1]):
                    swing_highs.append(current_window_high.iloc[j])
            
            # Find nearest swing low (local minimum)
            swing_lows = []
            for j in range(1, len(current_window_low)-1):
                if (current_window_low.iloc[j] < current_window_low.iloc[j-1] and 
                    current_window_low.iloc[j] < current_window_low.iloc[j+1]):
                    swing_lows.append(current_window_low.iloc[j])
            
            current_close = close_series.iloc[i]
            
            if swing_highs and swing_lows:
                nearest_high = min(swing_highs, key=lambda x: abs(x - current_close))
                nearest_low = min(swing_lows, key=lambda x: abs(x - current_close))
                
                dist_to_high = abs(current_close - nearest_high)
                dist_to_low = abs(current_close - nearest_low)
                
                # Efficiency: closer to swing points indicates better positioning
                min_dist = min(dist_to_high, dist_to_low)
                avg_price = (nearest_high + nearest_low) / 2
                swing_eff.iloc[i] = 1 - (min_dist / avg_price) if avg_price > 0 else 0
            else:
                swing_eff.iloc[i] = 0
        
        return swing_eff
    
    data['swing_efficiency'] = swing_point_efficiency(data['high'], data['low'], data['close'])
    
    # Volume-Price Alignment
    # Breakout Confirmation: Volume on up days / Volume on down days (5-day ratio)
    data['price_change'] = data['close'].pct_change()
    data['is_up_day'] = (data['price_change'] > 0).astype(int)
    data['is_down_day'] = (data['price_change'] < 0).astype(int)
    
    up_volume_5d = (data['volume'] * data['is_up_day']).rolling(window=5).sum()
    down_volume_5d = (data['volume'] * data['is_down_day']).rolling(window=5).sum()
    data['breakout_confirmation'] = up_volume_5d / (down_volume_5d + 1e-8)  # Avoid division by zero
    
    # Volume Momentum: (Volume[t] - Volume[t-5]) / Average volume last 10 days
    avg_volume_10d = data['volume'].rolling(window=10).mean()
    data['volume_momentum'] = (data['volume'] - data['volume'].shift(5)) / (avg_volume_10d + 1e-8)
    
    # Divergence Detection
    # Price-Volume Divergence: Correlation between price changes and volume changes (5-day)
    def price_volume_correlation(price_series, volume_series, window=5):
        correlations = pd.Series(index=price_series.index, dtype=float)
        
        for i in range(len(price_series)):
            if i < window:
                correlations.iloc[i] = 0
                continue
            
            price_changes = price_series.iloc[i-window+1:i+1].pct_change().dropna()
            volume_changes = volume_series.iloc[i-window+1:i+1].pct_change().dropna()
            
            if len(price_changes) > 1 and len(volume_changes) > 1:
                correlation = price_changes.corr(volume_changes)
                correlations.iloc[i] = correlation if not np.isnan(correlation) else 0
            else:
                correlations.iloc[i] = 0
        
        return correlations
    
    data['price_volume_divergence'] = price_volume_correlation(data['close'], data['volume'])
    
    # Multi-timeframe Divergence: Short-term momentum vs medium-term trend direction
    short_term_momentum = data['close'].pct_change(3)  # 3-day momentum
    medium_term_trend = data['close'].rolling(window=10).mean().pct_change(5)  # 10-day MA 5-day change
    data['multi_timeframe_divergence'] = np.sign(short_term_momentum) * np.sign(medium_term_trend)
    
    # Signal Synthesis
    # Momentum Quality Score: Momentum persistence * Volume confirmation
    data['momentum_quality'] = data['momentum_persistence'] * data['breakout_confirmation']
    
    # Divergence Strength: Absolute divergence * Volume momentum
    abs_divergence = abs(data['price_volume_divergence']) + abs(data['multi_timeframe_divergence'])
    data['divergence_strength'] = abs_divergence * data['volume_momentum']
    
    # Final alpha factor combining all components
    alpha = (data['acceleration'].fillna(0) * 0.2 +
             data['momentum_quality'].fillna(0) * 0.3 +
             data['trend_strength'].fillna(0) * 0.15 +
             data['swing_efficiency'].fillna(0) * 0.15 +
             data['divergence_strength'].fillna(0) * 0.2)
    
    return alpha
