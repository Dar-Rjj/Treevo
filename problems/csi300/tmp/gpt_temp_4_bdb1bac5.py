import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    data = df.copy()
    
    # Efficiency Momentum Analysis
    data['Volume_Weighted_Efficiency'] = ((data['close'] - data['close'].shift(1)) * data['volume']) / (data['high'] - data['low'])
    data['Intraday_Efficiency'] = ((data['close'] - data['open']) * data['volume']) / (data['high'] - data['low'])
    data['Gap_Efficiency'] = ((data['open'] - data['close'].shift(1)) * data['volume']) / (data['high'] - data['low'])
    data['Efficiency_Momentum'] = data['Volume_Weighted_Efficiency'] - data['Volume_Weighted_Efficiency'].shift(1)
    
    # Microstructure Pressure Signals
    data['Opening_Pressure_Divergence'] = ((data['open'] - data['close'].shift(1)) * data['volume']) / (data['high'] - data['low'])
    data['Closing_Pressure_Persistence'] = (data['close'] - (data['high'] + data['low'])/2) * np.sign(data['close'] - data['open']) * data['volume']
    data['Effective_Spread_Asymmetry'] = 2 * np.abs(data['close'] - (data['high'] + data['low'])/2) / data['close'] * np.sign(data['close'] - data['open'])
    data['Pressure_Persistence'] = np.sign(data['Closing_Pressure_Persistence']) * np.sign(data['Opening_Pressure_Divergence'])
    
    # Volume Convergence Patterns
    data['Volume_per_Movement'] = data['volume'] / np.abs(data['close'] - data['open'])
    data['Volume_to_Range'] = data['volume'] / (data['high'] - data['low'])
    data['Liquidity_Absorption'] = data['volume'] / np.abs(data['close'] - data['close'].shift(1))
    
    data['Volume_Acceleration'] = np.sign(data['Volume_per_Movement'] - data['Volume_per_Movement'].shift(1)) * np.sign(data['Efficiency_Momentum'])
    data['Liquidity_Momentum'] = np.sign(data['Liquidity_Absorption'] - data['Liquidity_Absorption'].shift(1)) * np.sign(data['Efficiency_Momentum'])
    data['Volume_Pressure_Confirmation'] = np.sign(data['Volume_to_Range'] - data['Volume_to_Range'].shift(1)) * np.sign(data['Closing_Pressure_Persistence'])
    data['Volume_Asymmetry'] = np.sign(data['Volume_per_Movement']) * np.sign(data['Effective_Spread_Asymmetry'])
    
    # Efficiency-Microstructure Divergence Detection
    data['High_Efficiency_Low_Pressure'] = ((data['Volume_Weighted_Efficiency'] > data['Volume_Weighted_Efficiency'].shift(1)) & (data['Closing_Pressure_Persistence'] < 0)).astype(int)
    data['Low_Efficiency_High_Pressure'] = ((data['Volume_Weighted_Efficiency'] < data['Volume_Weighted_Efficiency'].shift(1)) & (data['Closing_Pressure_Persistence'] > 0)).astype(int)
    data['Volume_Efficiency_Confirming'] = ((data['Volume_Acceleration'] > 0) & (data['Efficiency_Momentum'] > 0)).astype(int)
    data['Volume_Efficiency_Diverging'] = ((data['Volume_Acceleration'] < 0) & (data['Efficiency_Momentum'] > 0)).astype(int)
    data['Positive_Convergence'] = ((data['Effective_Spread_Asymmetry'] > 0) & (data['Intraday_Efficiency'] > 0)).astype(int)
    data['Negative_Convergence'] = ((data['Effective_Spread_Asymmetry'] < 0) & (data['Intraday_Efficiency'] < 0)).astype(int)
    
    # Multi-Timeframe Analysis
    # Ultra-Short Term (1-day)
    data['Intraday_Efficiency_Pressure_Alignment'] = data['Intraday_Efficiency'] * data['Closing_Pressure_Persistence']
    data['Opening_Gap_Impact'] = np.abs(data['open'] - data['close'].shift(1)) / (data['high'] - data['low'])
    data['Immediate_Volume_Confirmation'] = data['Volume_Acceleration'] * data['Pressure_Persistence']
    
    # Short-Term (5-day)
    data['Efficiency_Trend'] = data['Volume_Weighted_Efficiency'] / data['Volume_Weighted_Efficiency'].shift(5)
    data['Pressure_Trend'] = data['Closing_Pressure_Persistence'] / data['Closing_Pressure_Persistence'].shift(5)
    data['Volume_Momentum'] = data['Volume_per_Movement'] / data['Volume_per_Movement'].shift(5)
    
    # Medium-Term (20-day)
    data['Efficiency_Persistence'] = (data['Volume_Weighted_Efficiency'] > data['Volume_Weighted_Efficiency'].shift(1)).rolling(window=20, min_periods=1).sum()
    data['Pressure_Consistency'] = (data['Closing_Pressure_Persistence'].rolling(window=20, min_periods=1).apply(
        lambda x: len(set(np.sign(x.dropna()))) == 1 if len(x.dropna()) > 0 else 0
    ))
    data['Volume_Stability'] = data['Volume_to_Range'] / data['Volume_to_Range'].shift(20)
    
    # Convergence Factor Integration
    data['Efficiency_Microstructure_Alignment'] = data['Efficiency_Momentum'] * data['Effective_Spread_Asymmetry'] * data['Pressure_Persistence']
    data['Volume_Pressure_Convergence'] = data['Volume_Acceleration'] * data['Closing_Pressure_Persistence'] * data['Volume_Pressure_Confirmation']
    
    # Multi-Timeframe Consistency
    data['Multi_Timeframe_Alignment'] = (
        np.sign(data['Intraday_Efficiency_Pressure_Alignment']) + 
        np.sign(data['Efficiency_Trend']) + 
        np.sign(data['Efficiency_Persistence'])
    )
    
    # Divergence Strength
    data['Divergence_Strength'] = (
        data['High_Efficiency_Low_Pressure'] + 
        data['Low_Efficiency_High_Pressure'] + 
        data['Volume_Efficiency_Confirming'] + 
        data['Positive_Convergence']
    )
    
    # Risk Assessment Components
    data['Pressure_Exhaustion'] = (np.sign(data['Closing_Pressure_Persistence']) != np.sign(data['Opening_Pressure_Divergence'])).astype(int)
    data['Liquidity_Quality'] = data['Volume_per_Movement'] * data['Liquidity_Absorption']
    data['Efficiency_Stability'] = data['Volume_Weighted_Efficiency'] / data['Volume_Weighted_Efficiency'].shift(5)
    data['Microstructure_Consistency'] = (data['Pressure_Persistence'].rolling(window=5, min_periods=1).apply(
        lambda x: len(set(x.dropna())) == 1 if len(x.dropna()) > 0 else 0
    ))
    
    # Signal Validation
    data['Immediate_Confirmation'] = (
        np.sign(data['Intraday_Efficiency_Pressure_Alignment']) + 
        np.sign(data['Immediate_Volume_Confirmation'])
    )
    data['Trend_Validation'] = np.sign(data['Efficiency_Trend']) + np.sign(data['Pressure_Trend'])
    data['Persistence_Check'] = data['Efficiency_Persistence'] / 20 + data['Pressure_Consistency']
    
    # Final Alpha Output - Bullish Convergence Score
    bullish_convergence = (
        data['Efficiency_Microstructure_Alignment'] * 
        data['Volume_Pressure_Convergence'] * 
        data['Multi_Timeframe_Alignment'] * 
        (1 - data['Pressure_Exhaustion']) * 
        data['Liquidity_Quality'] * 
        data['Immediate_Confirmation'] * 
        data['Trend_Validation'] * 
        data['Persistence_Check']
    )
    
    # Bearish Divergence Score
    bearish_divergence = (
        data['Divergence_Strength'] * 
        data['Pressure_Exhaustion'] * 
        (1 - data['Volume_Efficiency_Confirming']) * 
        (1 - data['Microstructure_Consistency']) * 
        (1 / (data['Liquidity_Quality'] + 1e-8))
    )
    
    # Final Alpha Factor
    alpha_factor = bullish_convergence - bearish_divergence
    
    # Clean infinite values and normalize
    alpha_factor = alpha_factor.replace([np.inf, -np.inf], np.nan)
    alpha_factor = (alpha_factor - alpha_factor.rolling(window=20, min_periods=1).mean()) / alpha_factor.rolling(window=20, min_periods=1).std()
    
    return alpha_factor
