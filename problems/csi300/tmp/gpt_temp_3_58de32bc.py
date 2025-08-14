import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Volume-Weighted Price Returns
    df['NextDayOpen'] = df['open'].shift(-1)
    df['SimpleReturns'] = (df['NextDayOpen'] - df['close']) / df['close']
    df['VolumeWeightedReturns'] = df['SimpleReturns'] * df['volume']

    # Identify Volume Surge Days
    df['VolumeChange'] = df['volume'] - df['volume'].shift(1)
    df['RollingVolMean'] = df['volume'].rolling(window=5).mean()
    df['VolumeSurge'] = (df['volume'] > df['RollingVolMean']).astype(int)

    # Calculate Adaptive Volatility
    df['DailyReturns'] = df['close'].pct_change()
    volatility_window = 20 + 40 * (df['DailyReturns'].rolling(window=20).std() > df['DailyReturns'].std()).astype(int)
    df['AdaptiveVolatility'] = df['DailyReturns'].rolling(window=volatility_window, min_periods=20).std()

    # Adjust for Volume Trends
    volume_ma = df['volume'].rolling(window=20).mean()
    volume_std = df['volume'].rolling(window=20).std()
    volume_zscore = (df['volume'] - volume_ma) / volume_std
    df['AdjustedAdaptiveVolatility'] = df['AdaptiveVolatility'] * (1 + np.abs(volume_zscore))

    # Refine Volume Surge Factors
    df['VolumeSurgeRatio'] = df['volume'] / df['volume'].shift(1)
    surge_factors = np.select(
        [df['VolumeSurgeRatio'] > 2.5, (df['VolumeSurgeRatio'] > 2.0) & (df['VolumeSurgeRatio'] <= 2.5), 
         (df['VolumeSurgeRatio'] > 1.5) & (df['VolumeSurgeRatio'] <= 2.0)],
        [1.8, 1.5, 1.2],
        default=1.0
    )
    df['RefinedSurgeFactor'] = df['VolumeSurgeRatio'] * surge_factors

    # Adjust Volume-Weighted Returns by Adaptive Volatility
    df['AdjustedReturns'] = df['VolumeWeightedReturns'] / df['AdjustedAdaptiveVolatility']

    # Combine Adjusted Returns with Refined Volume Surge Indicator
    df['FinalAlphaFactor'] = df['AdjustedReturns'] * np.where(df['VolumeSurge'] == 1, df['RefinedSurgeFactor'], 1)

    return df['FinalAlphaFactor'].dropna()
