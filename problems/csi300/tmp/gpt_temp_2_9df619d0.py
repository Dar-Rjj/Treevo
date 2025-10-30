import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Price-Efficiency Momentum
    # Short-Term Price Efficiency (t-5 to t)
    def calculate_efficiency_ratio(window_data):
        """Calculate price path efficiency ratio for a window"""
        if len(window_data) < 2:
            return np.nan
        
        # Actual price movement distance (sum of absolute daily returns)
        actual_distance = np.sum(np.abs(np.diff(window_data['close'])))
        
        # Minimum possible price movement distance (absolute total return)
        min_distance = np.abs(window_data['close'].iloc[-1] - window_data['close'].iloc[0])
        
        # Avoid division by zero
        if min_distance == 0:
            return 0
        
        efficiency_ratio = min_distance / actual_distance if actual_distance > 0 else 0
        return efficiency_ratio
    
    # Calculate rolling efficiency ratios
    efficiency_5d = []
    for i in range(len(data)):
        if i >= 4:  # Need at least 5 days
            window = data.iloc[i-4:i+1]
            eff_ratio = calculate_efficiency_ratio(window)
            efficiency_5d.append(eff_ratio)
        else:
            efficiency_5d.append(np.nan)
    
    data['efficiency_5d'] = efficiency_5d
    
    # Medium-Term Efficiency Trend (t-20 to t)
    efficiency_trend = []
    efficiency_slope = []
    
    for i in range(len(data)):
        if i >= 19:  # Need at least 20 days
            # Get last 4 rolling 5-day efficiency ratios
            recent_efficiencies = []
            for j in range(4):
                if i-j-3 >= 0:
                    window = data.iloc[i-j-4:i-j+1]
                    eff_ratio = calculate_efficiency_ratio(window)
                    recent_efficiencies.append(eff_ratio)
            
            if len(recent_efficiencies) >= 3:
                # Calculate slope using linear regression
                x = np.arange(len(recent_efficiencies))
                slope = np.polyfit(x, recent_efficiencies, 1)[0]
                efficiency_slope.append(slope)
                efficiency_trend.append(np.mean(recent_efficiencies))
            else:
                efficiency_slope.append(np.nan)
                efficiency_trend.append(np.nan)
        else:
            efficiency_slope.append(np.nan)
            efficiency_trend.append(np.nan)
    
    data['efficiency_trend'] = efficiency_trend
    data['efficiency_slope'] = efficiency_slope
    
    # Volatility-Scaled Efficiency
    # Recent Volatility (t-20 to t-1)
    data['returns'] = data['close'].pct_change()
    data['volatility_20d'] = data['returns'].rolling(window=20, min_periods=10).std()
    
    # Efficiency Score
    data['efficiency_score'] = data['efficiency_5d'] * (1 + data['efficiency_slope'].fillna(0))
    data['vol_scaled_efficiency'] = data['efficiency_score'] / (data['volatility_20d'] + 1e-8)
    
    # Volume Acceleration
    # Volume Momentum (t-5 to t)
    data['volume_ma_5'] = data['volume'].rolling(window=5, min_periods=3).mean()
    data['volume_ma_10'] = data['volume'].rolling(window=10, min_periods=5).mean()
    data['volume_momentum'] = (data['volume_ma_5'] - data['volume_ma_10']) / (data['volume_ma_10'] + 1e-8)
    
    # Acceleration Signal
    data['volume_change'] = data['volume'].pct_change()
    data['volume_acceleration'] = data['volume_change'].diff()
    
    # Acceleration Threshold Check
    volume_threshold = data['volume'].rolling(window=20, min_periods=10).quantile(0.7)
    data['high_volume_flag'] = (data['volume'] > volume_threshold).astype(int)
    data['acceleration_signal'] = ((data['volume_acceleration'] > 0) & 
                                  (data['volume_momentum'] > 0)).astype(int)
    
    # Price-Volume Efficiency
    # Efficiency during High Volume Periods
    high_volume_efficiency = []
    for i in range(len(data)):
        if i >= 4 and data['high_volume_flag'].iloc[i] == 1:
            window = data.iloc[i-4:i+1]
            eff_ratio = calculate_efficiency_ratio(window)
            high_volume_efficiency.append(eff_ratio)
        else:
            high_volume_efficiency.append(np.nan)
    
    data['high_volume_efficiency'] = high_volume_efficiency
    
    # Volume-Weighted Price Efficiency
    data['volume_weighted_efficiency'] = data['efficiency_5d'] * data['volume_momentum']
    
    # Combined Factor
    # Efficiency Momentum Score
    data['efficiency_momentum_score'] = (data['vol_scaled_efficiency'].fillna(0) + 
                                        data['efficiency_slope'].fillna(0))
    
    # Volume Acceleration Multiplier
    volume_multiplier_conditions = [
        (data['acceleration_signal'] == 1) & (data['high_volume_flag'] == 1),
        (data['acceleration_signal'] == 1) & (data['high_volume_flag'] == 0),
        (data['acceleration_signal'] == 0) & (data['high_volume_flag'] == 1),
        (data['acceleration_signal'] == 0) & (data['high_volume_flag'] == 0)
    ]
    volume_multiplier_values = [1.5, 1.2, 1.1, 1.0]
    data['volume_acceleration_multiplier'] = np.select(volume_multiplier_conditions, 
                                                      volume_multiplier_values, default=1.0)
    
    # Final Combined Factor
    data['combined_factor'] = (data['efficiency_momentum_score'] * 
                              data['volume_acceleration_multiplier'] * 
                              (1 + data['volume_weighted_efficiency'].fillna(0)))
    
    return data['combined_factor']
