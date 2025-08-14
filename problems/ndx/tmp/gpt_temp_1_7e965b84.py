import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Intraday Momentum with Extended Insight
    df['IntradayMomentum'] = df['High'] - df['Low']
    
    # Advanced Volume Shock
    df['VolumeShock'] = (df['Volume'] > 1.5 * df['Volume'].shift(1)).astype(int)
    
    # Short-Term and Long-Term Price Momentum
    df['ShortTermAvgClose'] = df['Close'].rolling(window=10).mean()
    df['LongTermAvgClose'] = df['Close'].rolling(window=60).mean()
    df['PriceMomentumDiff'] = df['ShortTermAvgClose'] - df['LongTermAvgClose']
    
    # Short-Term and Long-Term Volume Surge
    df['ShortTermAvgVolume'] = df['Volume'].rolling(window=10).mean()
    df['LongTermAvgVolume'] = df['Volume'].rolling(window=60).mean()
    df['VolumeSurgeDiff'] = df['ShortTermAvgVolume'] - df['LongTermAvgVolume']
    
    # Intraday Volatility
    df['IntradayVolatility'] = df['High'] - df['Low']
    
    # Short-Term Momentum
    avg_close_4_days = df['Close'].rolling(window=4).mean()
    current_momentum = df['Close'] - avg_close_4_days
    df['AdjustedMomentum'] = current_momentum / df['IntradayVolatility']
    
    # Synthesize Indicators for Comprehensive Analysis
    df['AlphaFactor'] = (
        df['IntradayMomentum'] * df['VolumeShock'] +
        df['PriceMomentumDiff'] +
        df['VolumeSurgeDiff'] +
        df['AdjustedMomentum']
    )
    
    return df['AlphaFactor']
