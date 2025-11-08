import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate returns for different timeframes
    df = df.copy()
    df['ret_5d'] = df['close'].pct_change(5)
    df['ret_20d'] = df['close'].pct_change(20)
    df['ret_60d'] = df['ret_20d'].pct_change(3)  # Approximate 60d return from 20d returns
    
    # Calculate momentum divergence between adjacent timeframes
    df['div_short_med'] = np.abs(df['ret_5d'] - df['ret_20d'])
    df['div_med_long'] = np.abs(df['ret_20d'] - df['ret_60d'])
    
    # Combine divergences with weighting (shorter-term divergence more important)
    df['momentum_divergence'] = 0.6 * df['div_short_med'] + 0.4 * df['div_med_long']
    
    # Calculate volume growth rates
    df['volume_5d_growth'] = df['volume'].pct_change(5)
    df['volume_20d_growth'] = df['volume'].pct_change(20)
    
    # Volume-persistence filter: check if volume growth aligns with momentum direction
    df['volume_momentum_alignment'] = 0
    df.loc[(df['ret_5d'] > 0) & (df['volume_5d_growth'] > 0), 'volume_momentum_alignment'] = 1
    df.loc[(df['ret_5d'] < 0) & (df['volume_5d_growth'] < 0), 'volume_momentum_alignment'] = 1
    
    # Track consecutive periods of alignment
    df['alignment_streak'] = 0
    current_streak = 0
    
    for i in range(len(df)):
        if df['volume_momentum_alignment'].iloc[i] == 1:
            current_streak += 1
        else:
            current_streak = 0
        df['alignment_streak'].iloc[i] = current_streak
    
    # Apply volume-persistence filter to momentum divergence
    df['filtered_divergence'] = df['momentum_divergence'] * (1 + df['alignment_streak'] * 0.1)
    
    # Final factor: higher values indicate stronger momentum divergence with volume confirmation
    factor = df['filtered_divergence']
    
    return factor
