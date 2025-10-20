import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Volume-Weighted Convergence Factor that combines multi-horizon price alignment 
    with volume confirmation across different timeframes.
    """
    close = df['close']
    volume = df['volume']
    
    # Calculate exponential weighted returns
    def calc_ewr(series, window, half_life):
        returns = series.pct_change()
        weights = np.array([np.exp(-np.log(2) * i / half_life) for i in range(window)])
        weights = weights / weights.sum()
        
        ewr = pd.Series(index=series.index, dtype=float)
        for i in range(window, len(series)):
            window_returns = returns.iloc[i-window:i]
            ewr.iloc[i] = (window_returns * weights).sum()
        
        return ewr
    
    # Calculate exponential weighted volume trends
    def calc_ewv(series, window, half_life):
        weights = np.array([np.exp(-np.log(2) * i / half_life) for i in range(window)])
        weights = weights / weights.sum()
        
        ewv = pd.Series(index=series.index, dtype=float)
        for i in range(window, len(series)):
            window_volume = series.iloc[i-window:i]
            ewv.iloc[i] = (window_volume * weights).sum()
        
        return ewv
    
    # Calculate all required components
    EWR_20 = calc_ewr(close, 20, 10)
    EWR_50 = calc_ewr(close, 50, 25)
    
    VT_20 = calc_ewv(volume, 20, 10)
    VT_50 = calc_ewv(volume, 50, 25)
    VT_5 = calc_ewv(volume, 5, 2.5)
    VT_10 = calc_ewv(volume, 10, 5)
    
    # Calculate Convergence Score
    CS = np.sign(EWR_20) * np.sign(EWR_50) * np.minimum(np.abs(EWR_20), np.abs(EWR_50))
    
    # Calculate Volume Alignment
    VA = np.sign(VT_20 - VT_5) * np.sign(VT_50 - VT_10)
    
    # Construct final factor based on convergence cases
    factor = pd.Series(index=df.index, dtype=float)
    
    # Strong Convergence Case
    strong_mask = (CS > 0) & (VA > 0)
    factor[strong_mask] = CS[strong_mask] * (1 + np.abs(VA[strong_mask]))
    
    # Moderate Convergence Case
    moderate_mask = (CS > 0) & (VA == 0)
    factor[moderate_mask] = CS[moderate_mask] * 1.0
    
    # Weak Convergence Case
    weak_mask = (CS > 0) & (VA < 0)
    factor[weak_mask] = CS[weak_mask] * 0.5
    
    # Divergence Case
    divergence_mask = CS <= 0
    factor[divergence_mask] = CS[divergence_mask] * 0.25
    
    return factor
