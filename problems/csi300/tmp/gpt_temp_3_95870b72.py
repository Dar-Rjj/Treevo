import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate 20-Day and 50-Day Price Momentum
    price_mom_20 = df['close'].diff(20)
    price_mom_50 = df['close'].diff(50)

    # Calculate 20-Day and 50-Day Volume Difference
    vol_diff_20 = df['volume'].diff(20)
    vol_diff_50 = df['volume'].diff(50)

    # Combine 20-Day and 50-Day Price and Volume Momentum
    combined_mom_vol_20 = (price_mom_20 / df['close'].shift(20)) * vol_diff_20
    combined_mom_vol_50 = (price_mom_50 / df['close'].shift(50)) * vol_diff_50

    # Calculate 20-Day and 50-Day Weighted Moving Averages (WMA)
    weights_20 = np.arange(1, 21)
    weights_50 = np.arange(1, 51)
    
    wma_20 = df['close'].rolling(window=20).apply(lambda x: np.average(x, weights=weights_20), raw=True)
    wma_50 = df['close'].rolling(window=50).apply(lambda x: np.average(x, weights=weights_50), raw=True)

    # Calculate 20-Day and 50-Day WMA Momentum
    wma_mom_20 = wma_20.diff(20)
    wma_mom_50 = wma_50.diff(50)

    # Combine 20-Day and 50-Day WMA and Volume Momentum
    combined_wma_vol_20 = (wma_mom_20 / wma_20.shift(20)) * vol_diff_20
    combined_wma_vol_50 = (wma_mom_50 / wma_50.shift(50)) * vol_diff_50

    # Calculate 20-Day and 50-Day Volume-Weighted Prices
    vwp_20 = df.rolling(window=20).apply(
        lambda x: np.average(x['close'], weights=x['volume']), raw=False
    )
    vwp_50 = df.rolling(window=50).apply(
        lambda x: np.average(x['close'], weights=x['volume']), raw=False
    )

    # Calculate 20-Day and 50-Day Volume-Weighted Price Momentum
    vwp_mom_20 = vwp_20.diff(20)
    vwp_mom_50 = vwp_50.diff(50)

    # Combine 20-Day and 50-Day Volume-Weighted Price and Volume Momentum
    combined_vwp_vol_20 = (vwp_mom_20 / vwp_20.shift(20)) * vol_diff_20
    combined_vwp_vol_50 = (vwp_mom_50 / vwp_50.shift(50)) * vol_diff_50

    # Final Factor Calculation
    final_factor = (combined_mom_vol_20 + combined_mom_vol_50 +
                    combined_wma_vol_20 + combined_wma_vol_50 +
                    combined_vwp_vol_20 + combined_vwp_vol_50)

    return final_factor
