import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Regime-Adaptive Asymmetric Price-Volume Divergence Alpha Factor
    """
    data = df.copy()
    
    # Volatility Regime Classification
    # True Range Calculation
    data['TR1'] = data['high'] - data['low']
    data['TR2'] = abs(data['high'] - data['close'].shift(1))
    data['TR3'] = abs(data['low'] - data['close'].shift(1))
    data['Volatility_Proxy'] = data[['TR1', 'TR2', 'TR3']].max(axis=1)
    
    # Volatility Ratio and Regime Classification
    data['Volatility_Ratio'] = data['Volatility_Proxy'] / data['Volatility_Proxy'].shift(5)
    data['High_Volatility'] = data['Volatility_Ratio'] > 1.5
    data['Low_Volatility'] = data['Volatility_Ratio'] < 0.8
    data['Normal_Volatility'] = (data['Volatility_Ratio'] >= 0.8) & (data['Volatility_Ratio'] <= 1.5)
    
    # Asymmetric Price Movement Analysis
    # Intraday Asymmetry Metrics
    denominator_up = np.where(data['open'] - data['low'] != 0, data['open'] - data['low'], 1)
    denominator_down = np.where(data['high'] - data['close'] != 0, data['high'] - data['close'], 1)
    
    data['Upward_Pressure'] = (data['high'] - data['open']) / denominator_up
    data['Downward_Resistance'] = (data['close'] - data['low']) / denominator_down
    
    # Multi-day Asymmetry Patterns
    data['High_Open_Sum3'] = (data['high'] - data['open']).rolling(window=3).sum()
    data['Open_Low_Sum3'] = (data['open'] - data['low']).rolling(window=3).sum()
    data['Close_Low_Sum3'] = (data['close'] - data['low']).rolling(window=3).sum()
    data['High_Close_Sum3'] = (data['high'] - data['close']).rolling(window=3).sum()
    
    denominator_bull = np.where(data['Open_Low_Sum3'] != 0, data['Open_Low_Sum3'], 1)
    denominator_bear = np.where(data['High_Close_Sum3'] != 0, data['High_Close_Sum3'], 1)
    
    data['Bullish_Asymmetry_3d'] = data['High_Open_Sum3'] / denominator_bull
    data['Bearish_Asymmetry_3d'] = data['Close_Low_Sum3'] / denominator_bear
    
    # Asymmetry Acceleration
    data['Intraday_Asymmetry_Change'] = data['Upward_Pressure'] / data['Upward_Pressure'].shift(1)
    data['Multi_Asymmetry_Trend'] = data['Bullish_Asymmetry_3d'] / data['Upward_Pressure']
    
    # Volume Distribution Anomalies
    # Volume Concentration Analysis
    data['Up_Day'] = data['close'] > data['open']
    data['Down_Day'] = data['close'] < data['open']
    
    data['High_Volume_Ratio'] = data['volume'].rolling(window=5).apply(
        lambda x: x[x.index[x > x.mean()]].sum() / x[x.index[x <= x.mean()]].sum() if len(x[x.index[x > x.mean()]]) > 0 and len(x[x.index[x <= x.mean()]]) > 0 else 1
    )
    
    # Volume Persistence Patterns
    data['Volume_Persistence'] = (data['volume'] > data['volume'].shift(1)).rolling(window=3).sum()
    data['Volume_Spike_Clustering'] = data['volume'] / ((data['volume'].shift(1) + data['volume'].shift(2)) / 2)
    
    # Volume-Price Distribution Divergence
    data['Daily_Price_Efficiency'] = data['amount'] / data['volume']
    
    # Regime-Adaptive Divergence Signals
    # High Volatility Strategy Components
    data['Down_Day_Volume_Pressure'] = data['Downward_Resistance'] * data['Daily_Price_Efficiency']
    data['Volume_Concentration_Signal'] = data['High_Volume_Ratio'] * data['Volume_Spike_Clustering']
    
    data['Range_Momentum_Divergence'] = data['Volatility_Proxy'] / data['Volatility_Proxy'].shift(1)
    data['Range_Expansion_Signal'] = -data['Range_Momentum_Divergence']
    data['Bearish_Asymmetry_Momentum'] = data['Bearish_Asymmetry_3d'] * data['Intraday_Asymmetry_Change']
    
    data['High_Vol_Signal'] = (
        data['Range_Expansion_Signal'] * 
        data['Bearish_Asymmetry_Momentum'] * 
        data['Volume_Concentration_Signal']
    )
    
    # Low Volatility Strategy Components
    data['Up_Day_Volume_Momentum'] = data['Upward_Pressure'] * data['Daily_Price_Efficiency']
    
    up_day_volume = data['volume'].rolling(window=10).apply(
        lambda x: x[x.index[data.loc[x.index, 'Up_Day']]].mean() if len(x[x.index[data.loc[x.index, 'Up_Day']]]) > 0 else 1
    )
    down_day_volume = data['volume'].rolling(window=10).apply(
        lambda x: x[x.index[data.loc[x.index, 'Down_Day']]].mean() if len(x[x.index[data.loc[x.index, 'Down_Day']]]) > 0 else 1
    )
    
    data['Distribution_Efficiency_Signal'] = up_day_volume / down_day_volume
    
    data['Range_Compression_Signal'] = data['Range_Momentum_Divergence']
    data['Bullish_Asymmetry_Momentum'] = data['Bullish_Asymmetry_3d'] * data['Intraday_Asymmetry_Change']
    
    data['Low_Vol_Signal'] = (
        data['Range_Compression_Signal'] * 
        data['Bullish_Asymmetry_Momentum'] * 
        data['Distribution_Efficiency_Signal']
    )
    
    # Normal Volatility Strategy Components
    data['Base_Asymmetry_Signal'] = data['Bullish_Asymmetry_Momentum'] - data['Bearish_Asymmetry_Momentum']
    data['Volume_Efficiency_Component'] = data['Daily_Price_Efficiency'] * data['Volume_Spike_Clustering']
    
    up_volume_ratio = data['volume'].rolling(window=10).apply(
        lambda x: x[x.index[data.loc[x.index, 'Up_Day']]].sum() / x.sum() if x.sum() > 0 else 0.5
    )
    down_volume_ratio = data['volume'].rolling(window=10).apply(
        lambda x: x[x.index[data.loc[x.index, 'Down_Day']]].sum() / x.sum() if x.sum() > 0 else 0.5
    )
    
    data['Asymmetric_Balance'] = (up_volume_ratio - down_volume_ratio) * data['Daily_Price_Efficiency']
    data['Volatility_Scaling'] = data['Base_Asymmetry_Signal'] / data['Volatility_Proxy']
    
    data['Normal_Vol_Signal'] = (
        data['Volatility_Scaling'] * 
        data['Volume_Efficiency_Component'] * 
        data['Asymmetric_Balance']
    )
    
    # Adaptive Alpha Integration
    # Regime Detection & Transition
    data['Regime_Persistence'] = data['Volatility_Ratio'] / data['Volatility_Ratio'].shift(1)
    
    # Signal Weighting & Amplification
    data['Regime_Confidence'] = 1 / abs(data['Volatility_Ratio'] - 1)
    data['Volume_Consistency'] = data['Volume_Spike_Clustering'] * (up_volume_ratio + down_volume_ratio) / 2
    
    # Transition Amplification
    volatility_breakout = (data['Low_Volatility'].shift(1) & data['High_Volatility'])
    volume_breakout = (data['volume'] > data['volume'].rolling(window=10).mean() * 1.5)
    combined_shift = volatility_breakout & volume_breakout
    
    data['Transition_Amplification'] = 1.0
    data.loc[volume_breakout, 'Transition_Amplification'] = 1.5
    data.loc[volatility_breakout, 'Transition_Amplification'] = -1.0
    data.loc[combined_shift, 'Transition_Amplification'] = 1.3
    
    data['Combined_Weight'] = (
        data['Regime_Confidence'] * 
        data['Volume_Consistency'] * 
        data['Transition_Amplification']
    )
    
    # Final Alpha Output
    # Regime-Specific Signal Selection
    regime_signals = pd.Series(index=data.index, dtype=float)
    regime_signals[data['High_Volatility']] = data['High_Vol_Signal']
    regime_signals[data['Low_Volatility']] = data['Low_Vol_Signal']
    regime_signals[data['Normal_Volatility']] = data['Normal_Vol_Signal']
    
    # Weighted Signal Combination
    final_alpha = regime_signals * data['Combined_Weight']
    
    # Clean and return
    final_alpha = final_alpha.replace([np.inf, -np.inf], np.nan)
    final_alpha = final_alpha.fillna(method='ffill').fillna(0)
    
    return final_alpha
