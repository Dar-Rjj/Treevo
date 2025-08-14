import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate Intraday Move
    df['IntradayMove'] = df['High'] - df['Close']
    
    # Calculate Intraday Volatility
    df['IntradayVolatility'] = df['High'] - df['Low']
    
    # Calculate Adjusted Daily Momentum
    df['DailyMomentum'] = df['Close'] - df['Close'].shift(1)
    df['AdjustedDailyMomentum'] = df['DailyMomentum'] / df['IntradayVolatility']
    
    # Estimate Trade Intensity
    df['TradeIntensity'] = df['Volume'] / ((df['High'] + df['Low']) / 2)
    
    # Weight Intraday Move by Trade Intensity
    df['WeightedIntradayMove'] = df['IntradayMove'] * df['TradeIntensity']
    
    # Weight Adjusted Daily Momentum by Trade Intensity
    df['WeightedAdjustedDailyMomentum'] = df['AdjustedDailyMomentum'] * df['TradeIntensity']
    
    # Calculate Intraday Reversal
    df['IntradayReversal'] = df['High'] - df['Open']
    
    # Weight Intraday Reversal by Trade Intensity
    df['WeightedIntradayReversal'] = df['IntradayReversal'] * df['TradeIntensity']
    
    # Calculate Intraday Gap
    df['IntradayGap'] = df['Open'] - df['Close'].shift(1)
    
    # Weight Intraday Gap by Trade Intensity
    df['WeightedIntradayGap'] = df['IntradayGap'] * df['TradeIntensity']
    
    # Combine All Weighted Components
    df['FactorValue'] = (df['WeightedIntradayMove'] + 
                         df['WeightedAdjustedDailyMomentum'] + 
                         df['WeightedIntradayReversal'] + 
                         df['WeightedIntradayGap'])
    
    return df['FactorValue']
