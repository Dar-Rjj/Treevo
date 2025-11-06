import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Price Momentum Analysis
    data['ROC_5'] = data['close'] / data['close'].shift(5) - 1
    data['ROC_10'] = data['close'] / data['close'].shift(10) - 1
    data['Price_Acceleration'] = data['ROC_5'] - data['ROC_10']
    
    # Volume Momentum Analysis
    data['Vol_ROC_5'] = data['volume'] / data['volume'].shift(5) - 1
    data['Vol_ROC_10'] = data['volume'] / data['volume'].shift(10) - 1
    data['Volume_Acceleration'] = data['Vol_ROC_5'] - data['Vol_ROC_10']
    
    # Volatility Analysis
    data['ROC_1'] = data['close'].pct_change()
    data['Price_Volatility'] = data['ROC_1'].rolling(window=20).std()
    data['Volume_ROC_1'] = data['volume'].pct_change()
    data['Volume_Volatility'] = data['Volume_ROC_1'].rolling(window=20).std()
    
    # True Range calculation
    data['TR1'] = data['high'] - data['low']
    data['TR2'] = abs(data['high'] - data['close'].shift(1))
    data['TR3'] = abs(data['low'] - data['close'].shift(1))
    data['True_Range'] = data[['TR1', 'TR2', 'TR3']].max(axis=1)
    data['Range_Volatility'] = data['True_Range'].rolling(window=5).mean()
    
    # Market Regime Detection
    data['Vol_Regime'] = data['True_Range'] > data['True_Range'].rolling(window=20).median()
    
    data['Trend_5'] = data['close'] / data['close'].shift(5) - 1
    data['Trend_20'] = data['close'] / data['close'].shift(20) - 1
    data['Trend_60'] = data['close'] / data['close'].shift(60) - 1
    data['Trend_Strength'] = (
        (np.sign(data['Trend_5']) == np.sign(data['Trend_20'])) & 
        (np.sign(data['Trend_20']) == np.sign(data['Trend_60']))
    )
    
    # Volatility-Weighted Divergence Analysis
    data['Volatility_Adjusted_Price_Accel'] = data['Price_Acceleration'] / (data['Price_Volatility'] + 1e-8)
    data['Volatility_Adjusted_Volume_Accel'] = data['Volume_Acceleration'] / (data['Volume_Volatility'] + 1e-8)
    data['Direction_Alignment'] = np.sign(data['Price_Acceleration']) == np.sign(data['Volume_Acceleration'])
    data['Base_Divergence_Signal'] = data['Volatility_Adjusted_Price_Accel'] * data['Volatility_Adjusted_Volume_Accel']
    data['Range_Adjusted_Momentum'] = data['ROC_5'] / (data['Range_Volatility'] + 1e-8)
    
    # Regime-Adaptive Alpha Generation
    alpha_values = []
    
    for i in range(len(data)):
        if i < 60:  # Need enough data for calculations
            alpha_values.append(0)
            continue
            
        row = data.iloc[i]
        vol_regime = row['Vol_Regime']
        trend_strength = row['Trend_Strength']
        direction_alignment = row['Direction_Alignment']
        
        # Divergence multiplier
        divergence_multiplier = 1.5 if direction_alignment else 0.7
        
        if vol_regime and trend_strength:
            # High Volatility + Strong Trend Regime
            high_window = data['high'].iloc[max(0, i-20):i+1]
            breakout_component = (row['high'] - high_window.max()) / (data['True_Range'].iloc[max(0, i-20):i+1].mean() + 1e-8)
            volume_boost = row['volume'] / (data['volume'].iloc[max(0, i-20):i+1].mean() + 1e-8)
            regime_alpha = (row['Base_Divergence_Signal'] * breakout_component * volume_boost * 
                          divergence_multiplier + row['Range_Adjusted_Momentum'])
            
        elif vol_regime and not trend_strength:
            # High Volatility + Weak Trend Regime
            low_window = data['low'].iloc[max(0, i-5):i+1]
            high_window = data['high'].iloc[max(0, i-5):i+1]
            mean_reversion_component = (row['close'] - low_window.min()) / (high_window.max() - low_window.min() + 1e-8)
            volume_dampening = 1 / (abs(row['Volume_Acceleration']) + 1)
            regime_alpha = (row['Base_Divergence_Signal'] * mean_reversion_component * volume_dampening * 
                          divergence_multiplier + row['Range_Adjusted_Momentum'])
            
        elif not vol_regime and trend_strength:
            # Low Volatility + Strong Trend Regime
            price_changes_sum = sum(abs(data['close'].iloc[i-j] / data['close'].iloc[i-j-1] - 1) 
                                  for j in range(min(10, i)))
            efficiency_ratio = abs(row['ROC_10']) / (price_changes_sum + 1e-8)
            volume_confidence = 1 / (1 + abs(row['Volume_Acceleration']))
            regime_alpha = (row['Base_Divergence_Signal'] * efficiency_ratio * volume_confidence * 
                          row['ROC_10'] * divergence_multiplier + row['Range_Adjusted_Momentum'])
            
        else:
            # Low Volatility + Weak Trend Regime
            volume_window = data['volume'].iloc[max(0, i-20):i+1]
            volume_anomaly = (row['volume'] - volume_window.mean()) / (volume_window.std() + 1e-8)
            contrarian_signal = -np.sign(row['Price_Acceleration'])
            regime_alpha = (row['Base_Divergence_Signal'] * volume_anomaly * contrarian_signal * 
                          divergence_multiplier + row['Range_Adjusted_Momentum'])
        
        alpha_values.append(regime_alpha)
    
    return pd.Series(alpha_values, index=data.index, name='regime_adaptive_alpha')
