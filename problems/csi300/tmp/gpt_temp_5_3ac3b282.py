import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Morning Volatility Fractal
    data['Morning_Volatility_Range'] = (data['high'] - data['open']) / (data['high'] - data['low'] + 0.001)
    data['Morning_Volatility_Intensity'] = data['Morning_Volatility_Range'] * (data['high'] - data['open'])
    
    # Afternoon Volatility Fractal
    data['Afternoon_Volatility_Range'] = (data['close'] - data['low']) / (data['high'] - data['low'] + 0.001)
    data['Afternoon_Volatility_Intensity'] = data['Afternoon_Volatility_Range'] * (data['close'] - data['low'])
    
    # Volatility Fractal Asymmetry
    data['Volatility_Range_Asymmetry'] = data['Morning_Volatility_Range'] - data['Afternoon_Volatility_Range']
    data['Volatility_Intensity_Asymmetry'] = data['Morning_Volatility_Intensity'] - data['Afternoon_Volatility_Intensity']
    
    # Persistence calculations
    def calculate_persistence(series, threshold):
        persistence = pd.Series(index=series.index, dtype=float)
        current_streak = 0
        for i in range(len(series)):
            if series.iloc[i] > threshold:
                current_streak += 1
            else:
                current_streak = 0
            persistence.iloc[i] = current_streak
        return persistence
    
    data['Morning_Volatility_Persistence'] = calculate_persistence(data['Morning_Volatility_Range'], 0.3)
    data['Afternoon_Volatility_Persistence'] = calculate_persistence(data['Afternoon_Volatility_Range'], 0.3)
    data['Volatility_Persistence_Asymmetry'] = data['Morning_Volatility_Persistence'] - data['Afternoon_Volatility_Persistence']
    
    # Multi-Timeframe Volatility Acceleration
    data['Short_Term_Volatility_Accel'] = ((data['high'] - data['low']) - (data['high'].shift(1) - data['low'].shift(1))) / \
                                         (data['high'].shift(1) - data['low'].shift(1) + 0.001) * np.sign(data['Volatility_Range_Asymmetry'])
    
    # Medium-term volatility (5-day windows)
    data['High_5d'] = data['high'].rolling(window=5, min_periods=1).max()
    data['Low_5d'] = data['low'].rolling(window=5, min_periods=1).min()
    data['High_5d_lag'] = data['high'].shift(5).rolling(window=5, min_periods=1).max()
    data['Low_5d_lag'] = data['low'].shift(5).rolling(window=5, min_periods=1).min()
    
    data['Medium_Term_Volatility_Accel'] = ((data['High_5d'] - data['Low_5d']) - (data['High_5d_lag'] - data['Low_5d_lag'])) / \
                                          (data['High_5d_lag'] - data['Low_5d_lag'] + 0.001) * np.sign(data['Volatility_Intensity_Asymmetry'])
    
    # Long-term volatility (20-day windows)
    data['High_20d'] = data['high'].rolling(window=20, min_periods=1).max()
    data['Low_20d'] = data['low'].rolling(window=20, min_periods=1).min()
    data['High_20d_lag'] = data['high'].shift(20).rolling(window=20, min_periods=1).max()
    data['Low_20d_lag'] = data['low'].shift(20).rolling(window=20, min_periods=1).min()
    
    data['Long_Term_Volatility_Accel'] = ((data['High_20d'] - data['Low_20d']) - (data['High_20d_lag'] - data['Low_20d_lag'])) / \
                                        (data['High_20d_lag'] - data['Low_20d_lag'] + 0.001) * np.sign(data['Volatility_Persistence_Asymmetry'])
    
    # Multi-Timeframe Volatility Acceleration composite
    data['Multi_Timeframe_Volatility_Accel'] = (data['Short_Term_Volatility_Accel'] + 
                                               data['Medium_Term_Volatility_Accel'] + 
                                               data['Long_Term_Volatility_Accel']) / 3
    
    # Volume-Volatility Fractal Coupling
    data['Morning_Volume_Volatility'] = data['volume'] * data['Morning_Volatility_Range']
    data['Afternoon_Volume_Volatility'] = data['volume'] * data['Afternoon_Volatility_Range']
    data['Volume_Volatility_Divergence'] = data['Morning_Volume_Volatility'] - data['Afternoon_Volume_Volatility']
    
    # Volume Volatility Momentum
    data['Volume_Volatility_Momentum'] = np.sign(data['volume'] - data['volume'].shift(3)) * \
                                        np.sign((data['high'] - data['low']) - (data['high'].shift(3) - data['low'].shift(3)))
    
    # Volume Volatility Persistence
    data['Volume_Volatility_Persistence'] = calculate_persistence(data['Volume_Volatility_Divergence'], 0)
    
    # Volume Volatility Stability
    def calculate_stability(series):
        stability = pd.Series(index=series.index, dtype=float)
        sign_changes = 0
        for i in range(1, len(series)):
            if np.sign(series.iloc[i]) != np.sign(series.iloc[i-1]):
                sign_changes += 1
            stability.iloc[i] = data['Volume_Volatility_Persistence'].iloc[i] / (sign_changes + 1)
        return stability
    
    data['Volume_Volatility_Stability'] = calculate_stability(data['Volume_Volatility_Divergence'])
    
    # Pressure Distribution Dynamics
    data['Upper_Pressure_Distribution'] = (data['high'] - data['open']) / (data['high'] - data['low'] + 0.001)
    data['Lower_Pressure_Distribution'] = (data['close'] - data['low']) / (data['high'] - data['low'] + 0.001)
    data['Pressure_Distribution_Imbalance'] = data['Upper_Pressure_Distribution'] - data['Lower_Pressure_Distribution']
    
    # Volatility Regime Classification
    regime_conditions = [
        (data['Morning_Volatility_Range'] > 0.4) & (data['Volatility_Range_Asymmetry'] > 0.1),
        (data['Morning_Volatility_Range'] > 0.4) & (data['Volatility_Range_Asymmetry'] <= 0.1),
        (data['Morning_Volatility_Range'] <= 0.4) & (data['Volatility_Range_Asymmetry'] > 0.1),
        (data['Morning_Volatility_Range'] <= 0.4) & (data['Volatility_Range_Asymmetry'] <= 0.1)
    ]
    
    # Regime 1: High Morning Volatility & High Asymmetry
    data['Volume_Expansion_Ratio'] = data['volume'] / ((data['volume'].shift(1) + data['volume'].shift(2) + data['volume'].shift(3)) / 3 + 0.001)
    data['Price_Momentum_Efficiency'] = np.abs(data['close'] - data['open']) / (data['high'] - data['low'] + 0.001)
    regime1_signal = data['Volume_Expansion_Ratio'] * data['Price_Momentum_Efficiency'] * data['Pressure_Distribution_Imbalance']
    
    # Regime 2: High Morning Volatility & Low Asymmetry
    data['Volume_Continuity'] = data['volume'] / (data['volume'].shift(2) + 0.001)
    data['Range_Expansion'] = (data['high'] - data['low']) / (data['high'].shift(4).rolling(window=4, min_periods=1).max() - 
                                                             data['low'].shift(4).rolling(window=4, min_periods=1).min() + 0.001)
    regime2_signal = data['Volume_Continuity'] * data['Range_Expansion'] * data['Volume_Volatility_Divergence']
    
    # Regime 3: Low Morning Volatility & High Asymmetry
    data['Volume_Concentration_Ratio'] = data['volume'] / (data['volume'] + data['volume'].shift(1) + data['volume'].shift(2) + 0.001)
    data['Volatility_Efficiency'] = (data['high'] - data['low']) / (data['close'] + 0.001) * data['Volume_Concentration_Ratio']
    regime3_signal = data['Volume_Concentration_Ratio'] * data['Volatility_Efficiency'] * data['Pressure_Distribution_Imbalance']
    
    # Regime 4: Low Morning Volatility & Low Asymmetry
    data['Volume_Volatility_Momentum_Ratio'] = ((data['volume'] - data['volume'].shift(3)) / (data['volume'].shift(3) + 0.001)) * \
                                              data['volume'] / (data['volume'] + data['volume'].shift(1) + data['volume'].shift(2) + 0.001)
    data['Intraday_Efficiency'] = np.abs(data['close'] - data['open']) / (data['high'] - data['low'] + 0.001)
    regime4_signal = data['Volume_Volatility_Momentum_Ratio'] * data['Intraday_Efficiency'] * data['Volume_Volatility_Stability']
    
    # Combine regime signals
    data['Regime_Signal'] = np.select(regime_conditions, [regime1_signal, regime2_signal, regime3_signal, regime4_signal], default=0)
    
    # Gap Volatility Transmission
    data['Overnight_Volatility_Gap'] = ((data['open'] - data['close'].shift(1)) / (data['close'].shift(1) + 0.001)) * \
                                      ((data['high'] - data['low']) / (data['high'].shift(1) - data['low'].shift(1) + 0.001))
    
    data['Intraday_Volatility_Recovery'] = ((data['close'] - data['open']) / (np.abs(data['open'] - data['close'].shift(1)) + 0.001)) * \
                                          ((data['high'] - data['low']) / (data['high'].shift(1) - data['low'].shift(1) + 0.001))
    
    data['Gap_Volatility_Transmission'] = data['Overnight_Volatility_Gap'] * data['Intraday_Volatility_Recovery'] * data['Multi_Timeframe_Volatility_Accel']
    
    # Fractal Volatility Quality
    # Volatility Pattern Quality
    data['Volatility_Pattern_Consistency'] = calculate_persistence(data['Volatility_Range_Asymmetry'], 0)
    
    def calculate_pattern_stability(series):
        stability = pd.Series(index=series.index, dtype=float)
        sign_changes = 0
        for i in range(1, len(series)):
            if np.sign(series.iloc[i]) != np.sign(series.iloc[i-1]):
                sign_changes += 1
            stability.iloc[i] = data['Volatility_Pattern_Consistency'].iloc[i] / (sign_changes + 1)
        return stability
    
    data['Volatility_Pattern_Stability'] = calculate_pattern_stability(data['Volatility_Range_Asymmetry'])
    data['Volatility_Pattern_Quality'] = data['Volatility_Pattern_Consistency'] * data['Volatility_Pattern_Stability']
    
    # Volume Volatility Quality
    data['Volume_Volatility_Consistency'] = calculate_persistence(data['Volume_Volatility_Divergence'], 0)
    
    def calculate_volume_stability(series):
        stability = pd.Series(index=series.index, dtype=float)
        sign_changes = 0
        for i in range(1, len(series)):
            if np.sign(series.iloc[i]) != np.sign(series.iloc[i-1]):
                sign_changes += 1
            stability.iloc[i] = data['Volume_Volatility_Consistency'].iloc[i] / (sign_changes + 1)
        return stability
    
    data['Volume_Volatility_Stability_Quality'] = calculate_volume_stability(data['Volume_Volatility_Divergence'])
    data['Volume_Volatility_Quality'] = data['Volume_Volatility_Consistency'] * data['Volume_Volatility_Stability_Quality']
    
    # Final Alpha Construction
    data['Core_Volatility_Signal'] = data['Multi_Timeframe_Volatility_Accel'] * data['Regime_Signal']
    data['Enhanced_Volatility_Signal'] = data['Core_Volatility_Signal'] * data['Gap_Volatility_Transmission'] * data['Volume_Volatility_Stability']
    data['Volatility_Quality_Score'] = data['Volatility_Pattern_Quality'] * data['Volume_Volatility_Quality']
    data['Quality_Enhanced_Volatility_Alpha'] = data['Enhanced_Volatility_Signal'] * data['Pressure_Distribution_Imbalance'] * data['Volatility_Quality_Score']
    
    return data['Quality_Enhanced_Volatility_Alpha']
