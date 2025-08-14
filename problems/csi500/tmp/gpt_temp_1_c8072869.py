import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate 12-Day Momentum
    df['12DayMomentum'] = df['close'] / df['close'].shift(12) - 1
    
    # Calculate Daily Return
    df['DailyReturn'] = df['close'] / df['close'].shift(1) - 1
    
    # Apply Moving Average to Daily Returns (N-day)
    N = 10
    df['MA_DailyReturn'] = df['DailyReturn'].rolling(window=N).mean()
    
    # Calculate 12-Day Volume Change
    df['12DayVolumeChange'] = df['volume'] / df['volume'].shift(12) - 1
    
    # Calculate Volume Adjusted Return
    df['VolumeAdjustedReturn'] = df['volume'] * df['DailyReturn']
    
    # Sum Volume Adjusted Returns over a window (e.g., 30 days)
    df['Sum_VolumeAdjustedReturn'] = df['VolumeAdjustedReturn'].rolling(window=30).sum()
    
    # Calculate Volume Acceleration
    df['5DayMA_VolumeChange'] = df['12DayVolumeChange'].rolling(window=5).mean()
    df['VolumeAcceleration'] = df['12DayVolumeChange'] - df['5DayMA_VolumeChange']
    
    # Combine Momentum and Volume Acceleration
    df['Combined_Momentum_VolumeAccel'] = df['12DayMomentum'] + 0.5 * df['VolumeAcceleration']
    
    # Calculate Price Change Velocity
    df['AvgDailyPriceChange'] = df['close'] - df['close'].shift(1)
    df['PriceChangeVelocity'] = df['AvgDailyPriceChange'].rolling(window=10).mean()
    
    # Identify Volume Spike
    spike_factor = 2
    df['VolumeSpike'] = (df['volume'] / df['volume'].shift(1)) > spike_factor
    
    # Calculate Volume-Weighted N-day Momentum
    N_momentum = 10
    df['VolumeWeightedMomentum'] = (df['DailyReturn'] * df['volume']).rolling(window=N_momentum).sum()
    df['VolumeWeightedMomentum_SpikeAdjusted'] = df.apply(
        lambda row: row['VolumeWeightedMomentum'] * spike_factor if row['VolumeSpike'] else row['VolumeWeightedMomentum'],
        axis=1
    )
    
    # Apply Volume Filter
    M = 100000  # Threshold for volume difference
    df['VolumeDifference'] = df['volume'] - df['volume'].shift(1)
    df['VolumeFilter'] = df['VolumeDifference'] > M
    
    # Calculate Momentum Adjusted Volume
    df['VolumeChangeRatio'] = df['volume'] / df['volume'].shift(1)
    df['EMA_Volume'] = df['volume'].ewm(span=10, adjust=False).mean()
    df['MomentumAdjustedVolume'] = df['VolumeChangeRatio'] * df['EMA_Volume']
    
    # Final Alpha Factor
    df['FinalAlphaFactor'] = df.apply(
        lambda row: (row['PriceChangeVelocity'] * row['MomentumAdjustedVolume'] * row['Combined_Momentum_VolumeAccel']) - 
                    (row['Sum_VolumeAdjustedReturn'] - row['Sum_VolumeAdjustedReturn'].rolling(window=252).mean()) 
                    if row['VolumeFilter'] else 0,
        axis=1
    )
    
    return df['FinalAlphaFactor']
