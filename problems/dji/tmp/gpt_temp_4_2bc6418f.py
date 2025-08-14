import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate the price-to-volume ratio
    price_volume_ratio = df['close'] / df['volume']
    
    # Calculate the log of the price-to-volume ratio
    log_pv_ratio = np.log(price_volume_ratio)
    
    # Calculate the 30-day EWMA of the log of the price-to-volume ratio
    ewma_log_pv_ratio = log_pv_ratio.ewm(span=30, adjust=False).mean()
    
    return heuristics_matrix
