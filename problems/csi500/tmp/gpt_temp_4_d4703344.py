import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate 12-Day Momentum
    df['12DayMomentum'] = df['close'].pct_change(12)
    
    # Calculate Daily Return
    df['DailyReturn'] = df['close'].pct_change()
    
    # Calculate 12-Day Volume Change
    df['12DayVolumeChange'] = df['volume'].pct_change(12)
    
    # Calculate Volume Adjusted Return
    df['VolumeAdjustedReturn'] = df['DailyReturn'] * df['volume']
    
    # Sum Volume Adjusted Returns over a window (e.g., 30 days)
    df['SumVolumeAdjustedReturns'] = df['VolumeAdjustedReturn'].rolling(window=30).sum()
    
    # Calculate Volume Acceleration
    df['5DayMovingAverageVolumeChange'] = df['12DayVolumeChange'].rolling(window=5).mean()
    df['VolumeAcceleration'] = df['12DayVolumeChange'] - df['5DayMovingAverageVolumeChange']
    
    # Apply Moving Average to Daily Returns
    df['MovingAverageDailyReturn'] = df['DailyReturn'].rolling(window=30).mean()
    
    # Apply Volume Difference Filter
    M = 0.05  # Example threshold
    volume_difference = df['volume'] - df['volume'].shift(1)
    df['FilteredMovingAverage'] = df['MovingAverageDailyReturn'].where(volume_difference > M, other=0)
    
    # Calculate 14-Day Price Volatility
    df['14DayPriceVolatility'] = df['close'].rolling(window=14).std()
    
    # Combine Momentum, Volume Acceleration, and Price Volatility
    combined_momentum = df['12DayMomentum'] + df['VolumeAcceleration'] * 0.5 - df['14DayPriceVolatility'] * 0.3
    
    # Calculate Price Change Velocity
    df['DailyPriceChange'] = df['close'] - df['close'].shift(1)
    df['PriceChangeVelocity'] = df['DailyPriceChange'].rolling(window=30).mean()
    
    # Calculate Momentum Adjusted Volume
    df['VolumeChangeRatio'] = df['volume'] / df['volume'].shift(1)
    df['ExponentialMovingAverageVolume'] = df['volume'].ewm(span=30).mean()
    df['MomentumAdjustedVolume'] = df['VolumeChangeRatio'] * df['ExponentialMovingAverageVolume']
    
    # Combine Price Change Velocity and Momentum Adjusted Volume
    df['CombinedSignal'] = df['PriceChangeVelocity'] * df['MomentumAdjustedVolume']
    
    # Calculate Daily Price Movement
    df['DailyPriceMovement'] = df['close'] - df['open']
    df['SignMovement'] = df['DailyPriceMovement'].apply(lambda x: 1 if x > 0 else -1)
    
    # Determine Adjusted Volume
    df['AdjustedVolume'] = df['volume'] * df['SignMovement']
    
    # Create Weighted Price Trend
    df['WeightedPriceTrend'] = (df['DailyPriceMovement'] * df['AdjustedVolume']).rolling(window=30).sum() / df['AdjustedVolume'].rolling(window=30).sum()
    
    # Integrate Final Signal
    df['FinalSignal'] = df.apply(
        lambda row: (combined_momentum + df['WeightedPriceTrend']) / 2 if row['FilteredMovingAverage'] != 0 else combined_momentum,
        axis=1
    )
    
    return df['FinalSignal']
