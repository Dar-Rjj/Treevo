import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df, sector_df, w1=0.4, w2=0.3, w3=0.3):
    # Short-Term Momentum and Volatility Alpha Factor
    ema_close = df['close'].ewm(span=5, adjust=False).mean()
    std_close = df['close'].rolling(window=10).std()
    momentum_volatility_factor = ema_close / std_close
    
    # Volume Trend Alpha Factor
    ema_volume = df['volume'].ewm(span=5, adjust=False).mean()
    volume_trend_factor = df['volume'] - ema_volume
    
    # Sector Performance Alpha Factor
    sector_avg_close = sector_df.groupby(sector_df.index)['close'].transform('mean')
    sector_performance_factor = df['close'] - sector_avg_close
    
    # Dynamic Weighting and Integration
    integrated_factor = (w1 * momentum_volatility_factor + 
                         w2 * volume_trend_factor + 
                         w3 * sector_performance_factor)
    
    # Adjust for Liquidity
    median_volume = df['volume'].rolling(window=10).median()
    liquidity_adjustment = df['volume'] / median_volume
    final_alpha_factor = integrated_factor * liquidity_adjustment
    
    return final_alpha_factor
