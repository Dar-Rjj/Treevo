import pandas as pd

def heuristics_v2(df):
    # Calculate components
    log_return = (df['close'].apply(np.log)).diff(20)
    volume_roc = df['volume'].pct_change()
    mod_volatility = (df['high'] - df['low']) / df['close']
    
    # Compute rolling 5-day forward return for weight calculation
    forward_return = df['close'].shift(-5).pct_change().rolling(window=60).corr(df['close'])
    
    # Calculate dynamic weights
    corr_log_return = log_return.rolling(window=60).corr(forward_return)
    corr_volume_roc = volume_roc.rolling(window=60).corr(forward_return)
    corr_mod_volatility = mod_volatility.rolling(window=60).corr(forward_return)
    
    # Normalize weights to sum up to 1
    total_corr = corr_log_return + corr_volume_roc + corr_mod_volatility
    weight_log_return = corr_log_return / total_corr
    weight_volume_roc = corr_volume_roc / total_corr
    weight_mod_volatility = corr_mod_volatility / total_corr
    
    # Combine components with their respective weights
    heuristics_matrix = (weight_log_return * log_return + 
                         weight_volume_roc * volume_roc + 
                         weight_mod_volatility * mod_volatility)
    return heuristics_matrix
