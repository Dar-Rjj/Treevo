import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate Daily Price Movement Range
    df['PriceRange'] = df['high'] - df['low']
    
    # Determine Daily Return Deviation
    df['DailyReturnDeviation'] = df['close'].diff()
    
    # Identify Trend Reversal Potential
    df['TrendReversalPotential'] = (df['DailyReturnDeviation'] > df['DailyReturnDeviation'].shift(1)) & (df['volume'] > df['volume'].rolling(window=14).mean())
    
    # Compute Moving Average of Daily Returns
    n = 20
    df['MovingAverageReturns'] = df['DailyReturnDeviation'].rolling(window=n).sum()
    
    # Calculate Volume Weight
    df['VolumeWeight'] = df['volume'] / df['volume'].rolling(window=n).mean()
    
    # Combine Moving Average and Volume Weight
    df['CombinedMAVolume'] = df['MovingAverageReturns'] * df['VolumeWeight']
    
    # Compute Price Momentum Factor
    df['PriceMomentum'] = df['DailyReturnDeviation'].rolling(window=n).sum()
    
    # Compute Volume Momentum Factor
    m = 10
    df['VolumeMomentum'] = df['volume'].diff().rolling(window=m).sum()
    
    # Combine Price and Volume Momentum
    k1 = 0.5
    k2 = 0.5
    df['CombinedMomentum'] = k1 * df['PriceMomentum'] + k2 * df['VolumeMomentum']
    
    # Adjust Combined Momentum for Volume Volatility
    df['VolumeMA'] = df['volume'].rolling(window=20).mean()
    df['VolumeDeviation'] = df['volume'] - df['VolumeMA']
    df['VolumeAdjustmentFactor'] = df['VolumeDeviation'] + 1e-6
    df['AdjustedMomentum'] = df['CombinedMomentum'] / df['VolumeAdjustmentFactor']
    
    # Calculate Price Breakout Ratio
    df['BreakoutRatio'] = (df['high'] - df['open']) / df['PriceRange']
    
    # Calculate Volume-Weighted Breakout Indicator
    df['VolumeWeightedBreakout'] = (df['close'] - df['open']) * df['volume'] * df['BreakoutRatio']
    
    # Aggregate Indicators
    df['AggregateBreakout'] = df['VolumeWeightedBreakout'].rolling(window=n).sum() * (df['volume'] / df['volume'].rolling(window=n).mean())
    
    # Smooth Aggregate Breakout Indicators
    smoothing_factor = 0.1
    df['SmoothedBreakout'] = df['AggregateBreakout'].ewm(alpha=smoothing_factor).mean()
    
    # Incorporate Price Action Context
    df['PriceChange'] = df['close'] - df['open']
    df['PriceActionContext'] = df['PriceChange'].apply(lambda x: 1 if x > 0 else -1)
    df['AdjustedBreakout'] = df['SmoothedBreakout'] * df['PriceActionContext']
    
    # Calculate Intraday Return
    df['IntradayReturn'] = (df['high'] - df['low']) / df['open']
    
    # Adjust for Volume
    df['VolumeMA_220'] = df['volume'].rolling(window=220).mean()
    df['IntradayReturn_Adjusted'] = (df['volume'] - df['VolumeMA_220']) * df['IntradayReturn']
    
    # Generate Final Alpha Factor
    alpha_factor = (
        df['TrendReversalPotential'] * 
        df['AdjustedMomentum'] * 
        df['AdjustedBreakout'] * 
        df['IntradayReturn_Adjusted']
    )
    
    # Enhance Alpha Factor
    alpha_factor = alpha_factor * df['DailyReturnDeviation']
    
    # Adjust for Volatility
    df['Volatility'] = (df[['high', 'low', 'close']].max(axis=1) - df[['high', 'low', 'close']].min(axis=1)).rolling(window=20).std()
    alpha_factor = alpha_factor / df['Volatility']
    
    # Introduce Seasonality Adjustment
    df['Month'] = df.index.month
    monthly_avg_returns = df.groupby(df.index.month)['DailyReturnDeviation'].transform('mean')
    df['MonthlySeasonalityFactor'] = df['DailyReturnDeviation'] - monthly_avg_returns
    alpha_factor = alpha_factor * df['MonthlySeasonalityFactor']
    
    return alpha_factor
