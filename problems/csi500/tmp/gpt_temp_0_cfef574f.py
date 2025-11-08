import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Multi-Timeframe Momentum with Volatility Regime Adaptation
    # Calculate Multi-Timeframe Momentum Signals
    df['M1'] = (df['close'] / df['close'].shift(2) - 1) * 100
    df['M2'] = (df['close'] / df['close'].shift(5) - 1) * 100
    df['M3'] = (df['close'] / df['close'].shift(10) - 1) * 100
    df['M4'] = (df['close'] / df['close'].shift(20) - 1) * 100
    
    # Calculate True Range
    df['TR'] = np.maximum(df['high'] - df['low'], 
                         np.maximum(np.abs(df['high'] - df['close'].shift(1)), 
                                   np.abs(df['low'] - df['close'].shift(1))))
    
    # Compute Volatility Metrics
    df['ST_Vol'] = df['TR'].rolling(5).mean()
    df['MT_Vol'] = df['TR'].rolling(10).mean()
    df['Vol_Ratio'] = df['ST_Vol'] / df['MT_Vol']
    
    # Classify Regime
    conditions = [
        df['Vol_Ratio'] > 1.5,
        (df['Vol_Ratio'] >= 0.8) & (df['Vol_Ratio'] <= 1.5),
        df['Vol_Ratio'] < 0.8
    ]
    choices = ['High', 'Normal', 'Low']
    df['Vol_Regime'] = np.select(conditions, choices, default='Normal')
    
    # Calculate Volume Confirmation
    df['Vol_MA_5'] = df['volume'].rolling(5).mean()
    df['Volume_Surge'] = df['volume'] / df['Vol_MA_5']
    df['Volume_Trend'] = np.sign(df['volume'] - df['volume'].shift(3))
    
    # Combine Momentum Based on Regime
    df['Base_Factor_Momentum'] = 0
    df.loc[df['Vol_Regime'] == 'High', 'Base_Factor_Momentum'] = df['M1'] * df['Volume_Surge']
    df.loc[df['Vol_Regime'] == 'Normal', 'Base_Factor_Momentum'] = (df['M1'] + df['M2']) * df['Volume_Surge']
    df.loc[df['Vol_Regime'] == 'Low', 'Base_Factor_Momentum'] = (df['M1'] + df['M2'] + df['M3'] + df['M4']) * df['Volume_Surge']
    
    # Apply Intraday Strength Adjustment
    df['Intraday_Strength'] = (df['close'] - df['open']) / (df['high'] - df['low'])
    df['Factor_Momentum'] = df['Base_Factor_Momentum'] * (1 + df['Intraday_Strength']) * df['Volume_Trend']
    
    # Price-Volume Divergence with Trend Acceleration
    # Calculate Price Trend Acceleration
    def linear_regression_slope(series):
        x = np.arange(len(series))
        return np.polyfit(x, series, 1)[0]
    
    df['Short_Trend'] = df['close'].rolling(4).apply(linear_regression_slope, raw=False)
    df['Medium_Trend'] = df['close'].rolling(9).apply(linear_regression_slope, raw=False)
    df['Price_Acceleration'] = df['Short_Trend'] - df['Medium_Trend']
    
    # Calculate Volume-Price Divergence
    df['Volume_Trend_Slope'] = df['volume'].rolling(6).apply(linear_regression_slope, raw=False)
    
    def rolling_corr(x):
        if len(x) < 6:
            return np.nan
        returns = x[:3] / x[1:4] - 1
        volume_changes = x[3:] / x[2:5] - 1
        return np.corrcoef(returns, volume_changes)[0, 1]
    
    df['Price_Volume_Corr'] = df['close'].rolling(6).apply(rolling_corr, raw=False)
    
    df['Divergence_Signal'] = df['Price_Acceleration'] * df['Volume_Trend_Slope'] * df['Price_Volume_Corr']
    
    # Apply Range Normalization
    df['ATR_5'] = df['TR'].rolling(5).mean()
    df['Factor_Divergence'] = df['Divergence_Signal'] / df['ATR_5']
    
    # Support-Resistance Breakout with Volume Persistence
    # Identify Key Price Levels
    df['Resistance'] = df['high'].rolling(15).max().shift(1)
    df['Support'] = df['low'].rolling(15).min().shift(1)
    df['Midpoint'] = (df['Resistance'] + df['Support']) / 2
    
    # Detect Breakout Signals
    df['Range_Position'] = (df['close'] - df['Midpoint']) / (df['Resistance'] - df['Support'])
    
    # Apply Volume Persistence Filter
    df['Vol_MA_10'] = df['volume'].rolling(10).mean()
    df['Volume_Surge_10'] = df['volume'] / df['Vol_MA_10']
    
    def count_above_avg(window):
        avg = window[:-1].mean()
        return (window[:-1] > avg).sum()
    
    df['Volume_Persistence'] = df['volume'].rolling(4).apply(count_above_avg, raw=False)
    df['Volume_Momentum'] = df['volume'] / df['volume'].shift(5) - 1
    
    df['Volume_Multiplier'] = df['Volume_Surge_10'] * df['Volume_Persistence'] * (1 + df['Volume_Momentum'])
    df['Factor_Breakout'] = df['Range_Position'] * 100 * df['Volume_Multiplier']
    
    # Mean Reversion with Liquidity-Adjusted Timing
    # Calculate Mean Reversion Signal
    df['MA_10'] = df['close'].rolling(10).mean()
    df['Std_10'] = df['close'].rolling(10).std()
    df['Z_Score'] = (df['close'] - df['MA_10']) / df['Std_10']
    
    # Assess Liquidity Conditions
    df['Volume_Amount_Ratio'] = df['volume'] / df['amount']
    df['VAR_MA_5'] = df['Volume_Amount_Ratio'].rolling(5).mean()
    df['Liquidity_Change'] = df['Volume_Amount_Ratio'] / df['VAR_MA_5']
    
    # Classify Liquidity Regime
    liq_conditions = [
        df['Liquidity_Change'] > 1.2,
        (df['Liquidity_Change'] >= 0.8) & (df['Liquidity_Change'] <= 1.2),
        df['Liquidity_Change'] < 0.8
    ]
    liq_choices = [1.5, 1.0, 0.5]
    df['Liquidity_Multiplier'] = np.select(liq_conditions, liq_choices, default=1.0)
    
    # Calculate Intraday Reversal Strength
    df['Intraday_Reversal_Strength'] = np.abs((df['close'] - df['open']) / (df['high'] - df['low']))
    
    # Combine Signals
    df['Factor_Reversion'] = -df['Z_Score'] * df['Liquidity_Multiplier'] * (1 + df['Intraday_Reversal_Strength'])
    
    # Combine all factors with equal weights
    factors = ['Factor_Momentum', 'Factor_Divergence', 'Factor_Breakout', 'Factor_Reversion']
    df['Final_Factor'] = df[factors].mean(axis=1)
    
    return df['Final_Factor']
