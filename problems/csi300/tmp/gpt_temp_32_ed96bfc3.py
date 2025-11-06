import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Pressure Efficiency Momentum System
    # Multi-Period Pressure Efficiency
    data['Realized_Pressure_Efficiency'] = (data['close'] - data['open']) / (data['high'] - data['low'])
    data['Opening_Pressure_Efficiency'] = (data['open'] - data['close'].shift(1)) / (data['high'].shift(1) - data['low'].shift(1))
    data['Closing_Pressure_Efficiency'] = (data['close'] - (data['high'] + data['low'])/2) / (data['high'] - data['low'])
    
    # Pressure Acceleration Framework
    data['Pressure_Momentum_5d'] = data['Realized_Pressure_Efficiency'] - data['Realized_Pressure_Efficiency'].shift(4)
    data['Pressure_Momentum_20d'] = data['Realized_Pressure_Efficiency'] - data['Realized_Pressure_Efficiency'].shift(19)
    data['Pressure_Acceleration'] = (data['Pressure_Momentum_5d'] - data['Pressure_Momentum_20d']) / 15
    
    # Pressure-Weighted Breakout Returns
    data['Short_term_Pressure_Return'] = (data['close'] / data['close'].shift(4) - 1) * data['Realized_Pressure_Efficiency']
    data['Medium_term_Pressure_Return'] = (data['close'] / data['close'].shift(19) - 1) * data['Opening_Pressure_Efficiency']
    data['Pressure_Return_Convergence'] = data['Short_term_Pressure_Return'] * data['Medium_term_Pressure_Return']
    
    # Volume-Pressure Confluence Validation
    # Volume-Pressure Alignment
    data['Volume_per_Unit_Pressure'] = data['volume'] / (data['high'] - data['low'])
    data['Volume_Slope_5d'] = data['volume'] / data['volume'].shift(4) - 1
    data['Pressure_Volume_Spike'] = data['volume'] / data['volume'].rolling(window=5, min_periods=1).mean()
    
    # Pressure Volatility Structure
    data['Daily_Pressure_Range'] = data['high'] - data['low']
    data['Avg_Pressure_Range_5d'] = data['Daily_Pressure_Range'].rolling(window=5, min_periods=1).mean()
    data['Pressure_Range_Expansion'] = data['Daily_Pressure_Range'] / data['Avg_Pressure_Range_5d']
    
    # Volume-Pressure Divergence Analysis
    data['Pressure_Volume_Divergence'] = data['Volume_per_Unit_Pressure'] / data['Volume_per_Unit_Pressure'].rolling(window=5, min_periods=1).mean()
    
    # Calculate rolling correlation for Volume-Pressure Correlation
    def rolling_corr_5d(x):
        return x['volume'].rolling(window=5, min_periods=1).corr(x['Realized_Pressure_Efficiency'])
    
    data['Volume_Pressure_Correlation'] = rolling_corr_5d(data)
    data['Pressure_Compression'] = (data['high'] - data['open']) / (data['close'] - data['low'])
    
    # Pressure Breakout Quality Assessment
    # Pressure Consistency Metrics
    data['Close_Up_Ratio_5d'] = (data['close'] > data['close'].shift(1)).rolling(window=5, min_periods=1).sum() / 5
    data['Pressure_Efficiency_Stability'] = 1 / data['Realized_Pressure_Efficiency'].rolling(window=5, min_periods=1).std()
    data['Pressure_Return_Stability'] = 1 / (data['close'] / data['close'].shift(5) - 1).rolling(window=5, min_periods=1).std()
    
    # Volume-Pressure Breakout Alignment
    # Calculate high/low volume pressure contributions
    def volume_pressure_contribution(data, window=10, top_pct=0.2):
        result = pd.Series(index=data.index, dtype=float)
        for i in range(window-1, len(data)):
            window_data = data.iloc[i-window+1:i+1]
            volume_threshold_high = window_data['volume'].quantile(0.8)
            volume_threshold_low = window_data['volume'].quantile(0.2)
            
            high_volume_days = window_data[window_data['volume'] >= volume_threshold_high]
            low_volume_days = window_data[window_data['volume'] <= volume_threshold_low]
            
            if len(high_volume_days) > 0:
                result.iloc[i] = high_volume_days['Realized_Pressure_Efficiency'].mean()
            else:
                result.iloc[i] = np.nan
        return result
    
    def volume_pressure_drift(data, window=10, bottom_pct=0.2):
        result = pd.Series(index=data.index, dtype=float)
        for i in range(window-1, len(data)):
            window_data = data.iloc[i-window+1:i+1]
            volume_threshold_low = window_data['volume'].quantile(0.2)
            
            low_volume_days = window_data[window_data['volume'] <= volume_threshold_low]
            
            if len(low_volume_days) > 0:
                result.iloc[i] = low_volume_days['Realized_Pressure_Efficiency'].mean()
            else:
                result.iloc[i] = np.nan
        return result
    
    data['High_Pressure_Volume_Contribution'] = volume_pressure_contribution(data)
    data['Low_Pressure_Volume_Drift'] = volume_pressure_drift(data)
    data['Pressure_Volume_Breakout_Confirmation'] = data['Volume_Pressure_Correlation'] * data['Pressure_Volume_Spike']
    
    # Gap-Pressure Breakout Integration
    # Overnight Gap Pressure Analysis
    data['Gap_Pressure_Magnitude'] = data['open'] / data['close'].shift(1) - 1
    data['Gap_Pressure_Efficiency'] = (data['open'] - data['close'].shift(1)) / (data['high'].shift(1) - data['low'].shift(1))
    data['Gap_Pressure_Alignment'] = data['Gap_Pressure_Efficiency'] * data['Realized_Pressure_Efficiency']
    
    # Intraday Pressure Realization
    data['Intraday_Pressure_Return'] = data['close'] / data['open'] - 1
    data['Gap_Filling_vs_Expansion_Pressure'] = data['Intraday_Pressure_Return'] * data['Gap_Pressure_Magnitude']
    data['Gap_Pressure_Breakout_Consistency'] = data['Gap_Pressure_Alignment'] * data['Pressure_Return_Convergence']
    
    # Composite Pressure Efficiency Alpha
    # Core Pressure Components
    data['Pressure_Momentum_Score'] = data['Pressure_Return_Convergence'] * data['Pressure_Acceleration']
    data['Pressure_Quality'] = data['Close_Up_Ratio_5d'] * data['Pressure_Efficiency_Stability']
    data['Pressure_Breakout_Validation'] = data['Pressure_Volume_Breakout_Confirmation'] * data['Pressure_Range_Expansion']
    
    # Gap-Pressure Enhancement
    data['Gap_Pressure_Multiplier'] = data['Gap_Pressure_Breakout_Consistency']
    data['Volume_Pressure_Alignment'] = data['High_Pressure_Volume_Contribution'] - data['Low_Pressure_Volume_Drift']
    
    # Final Alpha Integration
    data['Primary_Pressure_Factor'] = data['Pressure_Momentum_Score'] * data['Pressure_Quality']
    data['Secondary_Validation_Factor'] = data['Pressure_Breakout_Validation'] * data['Volume_Pressure_Alignment']
    data['Tertiary_Enhancement_Factor'] = data['Gap_Pressure_Multiplier'] * data['Pressure_Return_Stability']
    
    # Final Alpha
    alpha = data['Primary_Pressure_Factor'] * data['Secondary_Validation_Factor'] * data['Tertiary_Enhancement_Factor']
    
    return alpha
