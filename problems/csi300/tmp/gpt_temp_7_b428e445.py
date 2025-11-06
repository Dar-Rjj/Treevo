import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Price-Volume Divergence with Adaptive Persistence factor
    """
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Price momentum components
    df['mom_3d'] = (df['close'] - df['close'].shift(3)) / df['close'].shift(3)
    df['mom_5d'] = (df['close'] - df['close'].shift(5)) / df['close'].shift(5)
    df['mom_accel'] = df['mom_3d'] - df['mom_5d']
    
    # Volume dynamics
    df['vol_mom'] = (df['volume'] - df['volume'].shift(3)) / df['volume'].shift(3)
    df['vol_std_5d'] = df['volume'].rolling(window=5).std()
    df['vol_volatility'] = df['volume'] / df['vol_std_5d']
    
    # Volume trend persistence
    vol_ma_3d = df['volume'].rolling(window=3).mean()
    df['vol_above_ma'] = (df['volume'] > vol_ma_3d).astype(int)
    df['vol_persistence'] = df['vol_above_ma'].rolling(window=5, min_periods=1).apply(
        lambda x: np.sum(x * np.arange(1, len(x)+1) / np.arange(1, len(x)+1)), raw=False
    )
    
    # Price-Volume divergence detection
    df['directional_align'] = np.sign(df['mom_3d']) * np.sign(df['vol_mom'])
    df['magnitude_div'] = np.abs(df['mom_3d']) - np.abs(df['vol_mom'])
    
    # Correlation divergence
    df['price_ret'] = df['close'].pct_change()
    df['corr_5d'] = df['price_ret'].rolling(window=5).corr(df['volume'])
    df['corr_div'] = df['corr_5d'] - df['corr_5d'].rolling(window=20).mean()
    
    # Adaptive persistence filtering
    # Momentum persistence score
    mom_sign = np.sign(df['mom_3d'])
    df['mom_persistence'] = 0
    for i in range(1, len(df)):
        if mom_sign.iloc[i] == mom_sign.iloc[i-1] and not pd.isna(mom_sign.iloc[i]) and not pd.isna(mom_sign.iloc[i-1]):
            df.iloc[i, df.columns.get_loc('mom_persistence')] = df.iloc[i-1, df.columns.get_loc('mom_persistence')] + 1
        else:
            df.iloc[i, df.columns.get_loc('mom_persistence')] = 1
    
    df['mom_persistence_score'] = df['mom_persistence'] * np.abs(df['mom_3d'])
    
    # Volume confirmation filter
    vol_threshold = df['vol_mom'].rolling(window=20).quantile(0.3)
    df['vol_confirmation'] = (df['vol_mom'] > vol_threshold).astype(float)
    
    # Divergence persistence
    divergence_signal = (df['directional_align'] < 0).astype(int)
    df['div_persistence'] = 0
    for i in range(1, len(df)):
        if divergence_signal.iloc[i] == 1 and divergence_signal.iloc[i-1] == 1:
            df.iloc[i, df.columns.get_loc('div_persistence')] = df.iloc[i-1, df.columns.get_loc('div_persistence')] + 1
        else:
            df.iloc[i, df.columns.get_loc('div_persistence')] = divergence_signal.iloc[i]
    
    # Factor integration
    # Base divergence factor
    df['base_divergence'] = df['directional_align'] * df['magnitude_div'] * (1 + df['corr_div'])
    
    # Persistence-weighted signal
    df['weighted_signal'] = (df['base_divergence'] * 
                           df['mom_persistence_score'] * 
                           df['vol_confirmation'] * 
                           (1 + df['div_persistence']))
    
    # Final alpha factor
    alpha_factor = df['weighted_signal']
    
    return alpha_factor
