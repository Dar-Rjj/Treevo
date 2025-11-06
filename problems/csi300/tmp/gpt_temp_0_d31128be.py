import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import linregress

def heuristics_v2(df):
    # Calculate intraday volatility ratio
    df['high_low_range'] = df['high'] - df['low']
    df['close_open_gap'] = abs(df['close'] - df['open'])
    df['volatility_ratio'] = df['high_low_range'] / (df['close_open_gap'] + 1e-8)
    
    # Calculate volatility persistence (autocorrelation over 8 days)
    volatility_persistence = []
    for i in range(len(df)):
        if i < 8:
            volatility_persistence.append(np.nan)
        else:
            window = df['volatility_ratio'].iloc[i-7:i+1]
            if len(window) >= 2:
                correlation = window.autocorr()
                volatility_persistence.append(correlation if not pd.isna(correlation) else 0)
            else:
                volatility_persistence.append(0)
    df['volatility_persistence'] = volatility_persistence
    
    # Calculate volume efficiency
    df['volume_efficiency'] = df['volume'] / (df['amount'] + 1e-8)
    
    # Calculate efficiency momentum (linear regression slope over 6 days)
    efficiency_momentum = []
    for i in range(len(df)):
        if i < 6:
            efficiency_momentum.append(np.nan)
        else:
            window = df['volume_efficiency'].iloc[i-5:i+1]
            if len(window) >= 2:
                x = np.arange(len(window))
                slope, _, _, _, _ = linregress(x, window)
                efficiency_momentum.append(slope)
            else:
                efficiency_momentum.append(0)
    df['efficiency_momentum'] = efficiency_momentum
    
    # Combine components into final factor
    # Use volatility persistence as weight for efficiency momentum
    df['factor'] = df['volatility_persistence'] * df['efficiency_momentum']
    
    return df['factor']
