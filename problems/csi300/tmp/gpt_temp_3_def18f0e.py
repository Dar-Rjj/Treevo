import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Fractal Efficiency-Weighted Rejection Momentum with Microstructure Adaptation
    """
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Multi-Scale Fractal Efficiency Analysis
    # Price Efficiency Fractal Structure
    data['Raw_Efficiency'] = (data['close'] - data['open']) / (data['high'] - data['low'])
    data['Raw_Efficiency'] = data['Raw_Efficiency'].replace([np.inf, -np.inf], np.nan).fillna(0)
    
    data['Absolute_Efficiency'] = np.abs(data['close'] - data['open']) / (data['high'] - data['low'])
    data['Absolute_Efficiency'] = data['Absolute_Efficiency'].replace([np.inf, -np.inf], np.nan).fillna(0)
    
    data['Directional_Efficiency'] = np.sign(data['close'] - data['open']) * data['Raw_Efficiency']
    
    data['Efficiency_Fractal_Momentum'] = data['Raw_Efficiency'] - data['Raw_Efficiency'].shift(4)
    data['Efficiency_Fractal_Momentum'] = data['Efficiency_Fractal_Momentum'].fillna(0)
    
    # Volume Efficiency Alignment
    # Volume-Efficiency Correlation (10-day)
    data['Volume_Efficiency_Correlation'] = data['Raw_Efficiency'].rolling(window=10).corr(data['volume'])
    data['Volume_Efficiency_Correlation'] = data['Volume_Efficiency_Correlation'].fillna(0)
    
    data['High_Efficiency_Volume'] = data['amount'] / (data['volume'] * np.abs(data['close'] - data['open']))
    data['High_Efficiency_Volume'] = data['High_Efficiency_Volume'].replace([np.inf, -np.inf], np.nan).fillna(0)
    
    data['Low_Efficiency_Volume'] = data['amount'] / (data['volume'] * (data['high'] - data['low']))
    data['Low_Efficiency_Volume'] = data['Low_Efficiency_Volume'].replace([np.inf, -np.inf], np.nan).fillna(0)
    
    data['Volume_Flow_Persistence'] = (np.sign(data['volume'] - data['volume'].shift(1)) * 
                                     np.sign(data['volume'].shift(1) - data['volume'].shift(2)))
    data['Volume_Flow_Persistence'] = data['Volume_Flow_Persistence'].fillna(0)
    
    # Fractal Dimension Efficiency
    # Price fractal dimension estimation
    data['Daily_Range'] = data['high'] - data['low']
    data['Abs_Return'] = np.abs(data['close'] - data['close'].shift(1))
    data['Abs_Return'] = data['Abs_Return'].fillna(0)
    
    # Simplified Hurst-like calculation for price
    def calculate_hurst_like(series, window=20):
        hurst_values = []
        for i in range(len(series)):
            if i < window:
                hurst_values.append(0.5)
            else:
                window_data = series.iloc[i-window:i]
                if len(window_data) > 1 and window_data.std() > 0:
                    # Simplified rescaled range approximation
                    mean_val = window_data.mean()
                    deviations = window_data - mean_val
                    cumulative = deviations.cumsum()
                    r = cumulative.max() - cumulative.min()
                    s = window_data.std()
                    if s > 0:
                        hurst = np.log(r/s) / np.log(window) if r > 0 else 0.5
                    else:
                        hurst = 0.5
                else:
                    hurst = 0.5
                hurst_values.append(hurst)
        return pd.Series(hurst_values, index=series.index)
    
    data['Price_FD'] = calculate_hurst_like(data['close'], 20)
    
    # Efficiency fractal dimension
    data['Efficiency_Change'] = np.abs(data['Raw_Efficiency'] - data['Raw_Efficiency'].shift(1))
    data['Efficiency_Change'] = data['Efficiency_Change'].fillna(0)
    data['Efficiency_FD'] = calculate_hurst_like(data['Raw_Efficiency'], 20)
    
    data['Efficiency_Fractal_Divergence'] = data['Price_FD'] / data['Efficiency_FD']
    data['Efficiency_Fractal_Divergence'] = data['Efficiency_Fractal_Divergence'].replace([np.inf, -np.inf], np.nan).fillna(1)
    
    # Multi-Timeframe Rejection Asymmetry
    # Core Rejection Components
    data['Upper_Rejection'] = (data['high'] - np.maximum(data['open'], data['close'])) / (data['high'] - data['low'])
    data['Upper_Rejection'] = data['Upper_Rejection'].replace([np.inf, -np.inf], np.nan).fillna(0)
    
    data['Lower_Rejection'] = (np.minimum(data['open'], data['close']) - data['low']) / (data['high'] - data['low'])
    data['Lower_Rejection'] = data['Lower_Rejection'].replace([np.inf, -np.inf], np.nan).fillna(0)
    
    data['Net_Rejection_Asymmetry'] = data['Upper_Rejection'] - data['Lower_Rejection']
    
    # Extended Window Rejection
    data['3day_Window_Rejection'] = (data['high'] - data['close'].rolling(window=3).max()) - (data['close'].rolling(window=3).min() - data['low'])
    data['3day_Window_Rejection'] = data['3day_Window_Rejection'].fillna(0)
    
    data['10day_Window_Rejection'] = (data['high'] - data['close'].rolling(window=10).max()) - (data['close'].rolling(window=10).min() - data['low'])
    data['10day_Window_Rejection'] = data['10day_Window_Rejection'].fillna(0)
    
    # Rejection Persistence
    data['High_Rejection_Flag'] = (np.abs(data['Net_Rejection_Asymmetry']) > 0.3).astype(int)
    data['Rejection_Persistence'] = data['High_Rejection_Flag'].rolling(window=5, min_periods=1).sum()
    
    # Fractal Rejection Patterns
    data['Session_Rejection_Bias'] = (np.abs((data['high'] + data['low'])/2 - data['open']) / (data['high'] - data['low']) - 
                                    np.abs(data['close'] - (data['high'] + data['low'])/2) / (data['high'] - data['low']))
    data['Session_Rejection_Bias'] = data['Session_Rejection_Bias'].replace([np.inf, -np.inf], np.nan).fillna(0)
    
    data['Rejection_Fractal_Dimension'] = calculate_hurst_like(data['Net_Rejection_Asymmetry'], 20)
    
    data['Rejection_Efficiency_Correlation'] = data['Net_Rejection_Asymmetry'].rolling(window=10).corr(data['Raw_Efficiency'])
    data['Rejection_Efficiency_Correlation'] = data['Rejection_Efficiency_Correlation'].fillna(0)
    
    # Microstructure Regime Classification
    # Efficiency-Fractal Matrix
    data['High_Efficiency'] = (data['Raw_Efficiency'] > data['Raw_Efficiency'].rolling(window=5).mean()).astype(int)
    data['High_Fractal_Persistence'] = (data['Price_FD'] > 0.6).astype(int)
    data['Low_Fractal_Persistence'] = (data['Price_FD'] < 0.4).astype(int)
    
    # Rejection-Volume Regimes
    data['High_Rejection_Persistence'] = (data['Rejection_Persistence'] > 3).astype(int)
    data['Low_Rejection_Persistence'] = (data['Rejection_Persistence'] < 1).astype(int)
    data['Volume_Clustering'] = (data['Volume_Flow_Persistence'] > 0).astype(int)
    
    # Transition Detection
    data['Efficiency_Volatility_Change'] = (data['Raw_Efficiency'].rolling(window=5).std() / 
                                          data['Raw_Efficiency'].rolling(window=20).std())
    data['Efficiency_Volatility_Change'] = data['Efficiency_Volatility_Change'].replace([np.inf, -np.inf], np.nan).fillna(1)
    
    data['Fractal_Transition'] = np.abs(data['Price_FD'] - data['Price_FD'].shift(5))
    data['Fractal_Transition'] = data['Fractal_Transition'].fillna(0)
    
    data['Rejection_Regime_Shift'] = np.abs(data['Rejection_Persistence'] - data['Rejection_Persistence'].shift(5))
    data['Rejection_Regime_Shift'] = data['Rejection_Regime_Shift'].fillna(0)
    
    # Adaptive Factor Construction
    # High Efficiency, High Fractal Persistence Regime
    regime1_factor = (data['Efficiency_Fractal_Momentum'] * data['Efficiency_Fractal_Divergence'] * 
                     data['Net_Rejection_Asymmetry'] * data['Volume_Efficiency_Correlation'])
    
    # Low Efficiency, Low Fractal Persistence Regime
    regime2_factor = (data['10day_Window_Rejection'] * data['Volume_Flow_Persistence'] * 
                     (1 - data['Absolute_Efficiency']) * (1 - data['Price_FD']))
    
    # Transition Regime (High Efficiency Volatility)
    regime3_factor = (data['Rejection_Fractal_Dimension'] * data['Efficiency_Volatility_Change'] * 
                     data['Volume_Flow_Persistence'] * data['Session_Rejection_Bias'])
    
    # Mixed Fractal-Rejection Factor
    regime4_factor = (data['Efficiency_Fractal_Divergence'] * data['Rejection_Efficiency_Correlation'] * 
                     (1 / (1 + np.abs(data['Volume_Flow_Persistence']))) * data['Absolute_Efficiency'])
    
    # Final Alpha Synthesis
    # Multi-regime classification and application
    alpha_signal = pd.Series(index=data.index, dtype=float)
    
    for i in range(len(data)):
        if i < 20:  # Ensure enough data for calculations
            alpha_signal.iloc[i] = 0
            continue
            
        # Regime classification
        if (data['High_Efficiency'].iloc[i] == 1 and data['High_Fractal_Persistence'].iloc[i] == 1):
            # High Efficiency, High Fractal Persistence
            alpha_signal.iloc[i] = regime1_factor.iloc[i]
        elif (data['High_Efficiency'].iloc[i] == 0 and data['Low_Fractal_Persistence'].iloc[i] == 1):
            # Low Efficiency, Low Fractal Persistence
            alpha_signal.iloc[i] = regime2_factor.iloc[i]
        elif (data['Efficiency_Volatility_Change'].iloc[i] > 1.5):
            # Transition Regime
            alpha_signal.iloc[i] = regime3_factor.iloc[i]
        else:
            # Mixed Regime
            alpha_signal.iloc[i] = regime4_factor.iloc[i]
    
    # Microstructure validation and signal enhancement
    volume_efficiency_threshold = data['Volume_Efficiency_Correlation'].abs().rolling(window=10).mean()
    rejection_confirmation = data['Rejection_Persistence'] / 5.0  # Normalize to [0,1]
    fractal_consistency = 1 - np.abs(data['Efficiency_Fractal_Divergence'] - 1)
    
    # Final enhancement
    enhanced_alpha = (alpha_signal * 
                     np.where(volume_efficiency_threshold > 0.1, 1, 0.5) *  # Volume-Efficiency alignment
                     rejection_confirmation *  # Rejection persistence
                     fractal_consistency)  # Fractal divergence consistency
    
    # Normalize and clean
    enhanced_alpha = enhanced_alpha.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    return enhanced_alpha
