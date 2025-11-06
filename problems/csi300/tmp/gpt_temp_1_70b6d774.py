import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Hierarchical Volatility Decomposition
    data['Intraday_Volatility'] = (data['high'] - data['low']) / data['open']
    data['Gap_Volatility'] = abs(data['open'] - data['close'].shift(1)) / (data['close'].shift(1) + 1e-8)
    
    # Volatility regime detection
    intraday_vol_median = data['Intraday_Volatility'].rolling(window=5, min_periods=1).median()
    data['Volatility_Regime'] = 'Transition'
    data.loc[data['Intraday_Volatility'] > 1.5 * intraday_vol_median, 'Volatility_Regime'] = 'Explosive'
    data.loc[data['Intraday_Volatility'] < 0.6 * intraday_vol_median, 'Volatility_Regime'] = 'Compressed'
    
    # Volume Pattern Recognition
    data['Volume_Momentum'] = data['volume'] / (data['volume'].shift(1) + 1e-8) - 1
    
    # Volume clustering calculation
    volume_increase_count = pd.Series(index=data.index, dtype=float)
    for i in range(len(data)):
        if i >= 4:
            window = data.iloc[i-4:i+1]
            count = sum(window['volume'].iloc[j] > window['volume'].iloc[j-1] for j in range(1, 5))
            volume_increase_count.iloc[i] = count
        else:
            volume_increase_count.iloc[i] = 0
    data['Volume_Clustering'] = volume_increase_count
    
    # Volume-Price Dislocation
    data['Volume_Price_Divergence'] = np.sign(data['close'] - data['close'].shift(1)) * np.sign(data['Volume_Momentum'])
    data['Volume_Efficiency_Gap'] = abs(data['close'] - data['open']) / (data['volume'] + 1)
    data['Volume_Regime_Sensitivity'] = data['volume'] / (data['volume'].rolling(window=3, min_periods=1).median() + 1e-8)
    
    # Multi-Frequency Momentum Structure
    data['Ultra_Short_Momentum'] = data['close'] / (data['close'].shift(2) + 1e-8) - 1
    data['Short_Medium_Momentum'] = data['close'] / (data['close'].shift(6) + 1e-8) - 1
    data['Momentum_Regime_Divergence'] = data['Ultra_Short_Momentum'] - data['Short_Medium_Momentum']
    
    # Momentum Quality Assessment
    data['Path_Efficiency_Ratio'] = abs(data['close'] - data['close'].shift(4)) / (
        abs(data['close'] - data['close'].shift(1)) + 
        abs(data['close'].shift(1) - data['close'].shift(2)) + 
        abs(data['close'].shift(2) - data['close'].shift(3)) + 
        abs(data['close'].shift(3) - data['close'].shift(4)) + 1e-8
    )
    data['Gap_Absorption'] = abs(data['open'] - data['close'].shift(1)) / (data['high'] - data['low'] + 0.001)
    data['Momentum_Purity'] = data['Path_Efficiency_Ratio'] * (1 - data['Gap_Absorption'])
    
    # Regime-Adaptive Signal Architecture
    # Frequency-Adaptive Momentum
    data['Frequency_Adaptive_Momentum'] = 0.0
    explosive_mask = data['Volatility_Regime'] == 'Explosive'
    compressed_mask = data['Volatility_Regime'] == 'Compressed'
    transition_mask = data['Volatility_Regime'] == 'Transition'
    
    data.loc[explosive_mask, 'Frequency_Adaptive_Momentum'] = data['close'] / (data['close'].shift(1) + 1e-8) - 1
    data.loc[compressed_mask, 'Frequency_Adaptive_Momentum'] = data['close'] / (data['close'].shift(10) + 1e-8) - 1
    data.loc[transition_mask, 'Frequency_Adaptive_Momentum'] = data['close'] / (data['close'].shift(4) + 1e-8) - 1
    
    # Volume-Momentum Synchronization
    data['Volume_Momentum_Coherence'] = np.sign(data['Frequency_Adaptive_Momentum']) * np.sign(data['Volume_Price_Divergence'])
    
    # Volume-Price Momentum Correlation (3-day rolling)
    price_momentum = data['close'] / (data['close'].shift(1) + 1e-8) - 1
    vol_momentum = data['Volume_Momentum']
    
    correlation_values = []
    for i in range(len(data)):
        if i >= 2:
            window_prices = price_momentum.iloc[i-2:i+1]
            window_volumes = vol_momentum.iloc[i-2:i+1]
            if len(window_prices) >= 2:
                corr = window_prices.corr(window_volumes)
                correlation_values.append(corr if not np.isnan(corr) else 0)
            else:
                correlation_values.append(0)
        else:
            correlation_values.append(0)
    
    data['Volume_Price_Momentum_Correlation'] = correlation_values
    data['Synchronization_Score'] = data['Volume_Momentum_Coherence'] * data['Volume_Price_Momentum_Correlation']
    
    # Quality-Weighted Enhancement
    data['Regime_Quality_Factor'] = 1.0
    data.loc[explosive_mask, 'Regime_Quality_Factor'] = 0.5
    data.loc[compressed_mask, 'Regime_Quality_Factor'] = 1.5
    
    data['Quality_Score'] = data['Momentum_Purity'] * data['Regime_Quality_Factor']
    
    # Order Flow Imbalance Metrics
    data['Buy_Side_Pressure'] = ((data['close'] - data['low']) / (data['high'] - data['low'] + 0.001)) * data['volume']
    data['Sell_Side_Pressure'] = ((data['high'] - data['close']) / (data['high'] - data['low'] + 0.001)) * data['volume']
    data['Net_Order_Flow_Imbalance'] = (data['Buy_Side_Pressure'] - data['Sell_Side_Pressure']) / (
        data['Buy_Side_Pressure'] + data['Sell_Side_Pressure'] + 1
    )
    
    # Signal Fusion and Refinement
    data['Core_Divergence_Signal'] = data['Frequency_Adaptive_Momentum'] * data['Synchronization_Score']
    data['Quality_Enhanced_Signal'] = data['Core_Divergence_Signal'] * data['Quality_Score']
    data['Flow_Confirmed_Signal'] = data['Quality_Enhanced_Signal'] * data['Net_Order_Flow_Imbalance']
    data['Volume_Clustering_Filter'] = data['Volume_Clustering'] / 4
    
    # Market Microstructure Stability
    data['Price_Compression'] = 1 - (data['Intraday_Volatility'] / (intraday_vol_median + 1e-8))
    data['Volume_Consistency'] = 1 - abs(data['Volume_Momentum'])
    data['Microstructure_Stability_Index'] = (data['Price_Compression'] + data['Volume_Consistency']) / 2
    
    # Final Alpha Synthesis
    data['Base_Alpha_Signal'] = data['Flow_Confirmed_Signal'] * data['Volume_Clustering_Filter']
    data['Stability_Weighted_Signal'] = data['Base_Alpha_Signal'] * data['Microstructure_Stability_Index']
    
    # Regime-Specific Amplification
    data['Regime_Amplified_Signal'] = data['Stability_Weighted_Signal']
    data.loc[explosive_mask, 'Regime_Amplified_Signal'] = 0.6 * data['Stability_Weighted_Signal']
    data.loc[compressed_mask, 'Regime_Amplified_Signal'] = 1.4 * data['Stability_Weighted_Signal']
    
    # Final Alpha Output
    data['Final_Alpha'] = data['Regime_Amplified_Signal'] * data['Volume_Regime_Sensitivity']
    
    return data['Final_Alpha']
