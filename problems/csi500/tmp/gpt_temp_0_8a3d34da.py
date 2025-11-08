import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate 10-day Rate of Change of Close
    data['roc_10'] = data['close'].pct_change(periods=10)
    
    # Compute 20-day Rolling Z-Score of ROC (using only past data)
    data['roc_zscore'] = data['roc_10'].rolling(window=20).apply(
        lambda x: (x.iloc[-1] - x[:-1].mean()) / x[:-1].std() if len(x[:-1]) > 1 and x[:-1].std() != 0 else 0
    )
    
    # Calculate Daily Turnover Rate (using volume as proxy)
    # Since we don't have shares outstanding, use volume relative to 20-day average
    data['turnover_rate'] = data['volume'] / data['volume'].rolling(window=20).mean()
    
    # Compute 5-day Acceleration of Turnover Rate
    data['turnover_accel'] = data['turnover_rate'] - data['turnover_rate'].shift(5)
    
    # Calculate Liquidity Volatility (10-day StdDev of Turnover Rate)
    data['liquidity_vol'] = data['turnover_rate'].rolling(window=10).std()
    
    # Calculate 5-day average of liquidity volatility
    data['liquidity_vol_avg'] = data['liquidity_vol'].rolling(window=5).mean()
    
    # Identify conditions
    momentum_extreme = (data['roc_zscore'] > 2) | (data['roc_zscore'] < -2)
    positive_accel = data['turnover_accel'] > 0
    high_vol = data['liquidity_vol'] > data['liquidity_vol_avg']
    
    # Combined reversal condition
    reversal_condition = momentum_extreme & positive_accel & high_vol
    
    # Generate alpha factor
    alpha_factor = pd.Series(0, index=data.index)
    alpha_factor[reversal_condition] = (
        data['roc_zscore'] * data['turnover_accel'] * data['liquidity_vol']
    )[reversal_condition]
    
    return alpha_factor
