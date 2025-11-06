import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Price-Volume Divergence Momentum
    # Short-term Momentum Components
    df['mom_5'] = df['close'] / df['close'].shift(5) - 1
    df['mom_10'] = df['close'] / df['close'].shift(10) - 1
    df['mom_ratio'] = (df['close'] / df['close'].shift(5)) / (df['close'] / df['close'].shift(10))
    
    # Volume Confirmation Signals
    df['vol_trend'] = df['volume'] / df['volume'].shift(5)
    df['vol_mom'] = (df['volume'] / df['volume'].shift(5)) / (df['volume'].shift(5) / df['volume'].shift(10))
    
    # Volume persistence: count(volume > volume_{t-1})_{t-5:t}
    vol_persistence = []
    for i in range(len(df)):
        if i < 5:
            vol_persistence.append(np.nan)
        else:
            count = sum(df['volume'].iloc[i-j] > df['volume'].iloc[i-j-1] for j in range(5))
            vol_persistence.append(count)
    df['vol_persistence'] = vol_persistence
    
    # Divergence Detection
    df['bullish_div'] = (df['mom_5'] > 0) & (df['vol_trend'] < 1)
    df['bearish_div'] = (df['mom_5'] < 0) & (df['vol_trend'] > 1)
    df['div_strength'] = abs(df['mom_5'] - df['vol_trend']) * df['vol_persistence']
    
    # Volatility-Scaled Range Efficiency
    # Volatility-Adjusted Components
    df['true_range_eff'] = abs(df['close'] - df['close'].shift(1)) / (df['high'] - df['low'])
    
    # 3-day cumulative efficiency
    close_diff_3d = abs(df['close'] - df['close'].shift(1)).rolling(window=3).sum()
    high_low_range_3d = (df['high'] - df['low']).rolling(window=3).sum()
    df['cum_eff_3d'] = close_diff_3d / high_low_range_3d
    
    # Gap-adjusted efficiency
    df['gap_adj_eff'] = abs(df['close'] - df['close'].shift(1)) / (
        np.maximum(df['high'], df['close'].shift(1)) - np.minimum(df['low'], df['close'].shift(1))
    )
    
    # Volatility Scaling
    df['price_range_10d'] = (df['high'].rolling(window=10).max() - df['low'].rolling(window=10).min()) / df['close'].shift(10)
    df['vol_ratio'] = df['close'].rolling(window=5).std() / df['close'].shift(5).rolling(window=5).std()
    df['range_persistence'] = (df['high'] - df['low']) / (df['high'] - df['low']).rolling(window=5).mean()
    
    # Efficiency Signals
    df['eff_momentum'] = df['true_range_eff'] / df['true_range_eff'].rolling(window=5).mean()
    
    # Volume-Regime Extreme Reversal
    # Regime Detection
    df['vol_regime'] = df['volume'] / df['volume'].rolling(window=10).median()
    df['price_regime'] = (df['high'] - df['low']) / (df['high'] - df['low']).rolling(window=10).mean()
    
    # Regime persistence
    regime_persistence = []
    for i in range(len(df)):
        if i < 3:
            regime_persistence.append(np.nan)
        else:
            current_regime = (df['vol_regime'].iloc[i] > 1) & (df['price_regime'].iloc[i] > 1)
            count = sum(
                (df['vol_regime'].iloc[i-j] > 1) & (df['price_regime'].iloc[i-j] > 1) == current_regime 
                for j in range(1, 4)
            )
            regime_persistence.append(count)
    df['regime_persistence'] = regime_persistence
    
    # Extreme Move Identification
    df['price_extreme'] = (df['close'] - df['close'].shift(1)) / (df['high'].shift(1) - df['low'].shift(1))
    df['vol_extreme'] = df['volume'] / df['volume'].shift(1)
    df['combined_extreme'] = df['price_extreme'] * df['vol_extreme']
    
    # Amount-Flow Direction Persistence
    # Flow Direction Analysis
    df['flow_dir'] = np.sign(df['close'] - df['close'].shift(1))
    df['flow_mag'] = df['amount'] * abs(df['close'] - df['close'].shift(1)) / df['close'].shift(1)
    df['net_flow_3d'] = (df['flow_mag'] * df['flow_dir']).rolling(window=3).sum()
    
    # Persistence Metrics
    # Consecutive same-direction days
    consec_days = []
    current_streak = 0
    for i in range(len(df)):
        if i == 0 or df['flow_dir'].iloc[i] != df['flow_dir'].iloc[i-1]:
            current_streak = 1
        else:
            current_streak += 1
        consec_days.append(current_streak)
    df['consec_days'] = consec_days
    
    df['flow_accel'] = df['net_flow_3d'] / df['net_flow_3d'].shift(3)
    
    # Flow consistency
    flow_consistency = []
    for i in range(len(df)):
        if i < 5:
            flow_consistency.append(np.nan)
        else:
            count = sum(df['flow_dir'].iloc[i-j] == df['flow_dir'].iloc[i] for j in range(5))
            flow_consistency.append(count / 5)
    df['flow_consistency'] = flow_consistency
    
    df['flow_regime'] = df['net_flow_3d'] / df['net_flow_3d'].rolling(window=10).mean()
    
    # Multi-Timeframe Momentum Convergence
    # Timeframe Momentum
    df['mom_3d'] = df['close'] / df['close'].shift(3) - 1
    df['mom_5d'] = df['close'] / df['close'].shift(5) - 1
    df['mom_10d'] = df['close'] / df['close'].shift(10) - 1
    
    # Momentum alignment
    mom_alignment = []
    for i in range(len(df)):
        if i < 10:
            mom_alignment.append(np.nan)
        else:
            count = sum([
                df['mom_3d'].iloc[i] > 0,
                df['mom_5d'].iloc[i] > 0,
                df['mom_10d'].iloc[i] > 0
            ])
            mom_alignment.append(count)
    df['mom_alignment'] = mom_alignment
    
    # Volume Confirmation
    df['vol_alignment'] = (df['volume'] / df['volume'].shift(3) > 1) & (df['volume'] / df['volume'].shift(5) > 1)
    
    # Volume momentum convergence
    vol_ratios = pd.DataFrame({
        'vol_3d': df['volume'] / df['volume'].shift(3),
        'vol_5d': df['volume'] / df['volume'].shift(5),
        'vol_10d': df['volume'] / df['volume'].shift(10)
    })
    df['vol_std'] = vol_ratios.std(axis=1)
    
    # Final alpha factor combining all components
    alpha = (
        # Price-Volume Divergence
        df['div_strength'] * np.where(df['bullish_div'], 1, -1) +
        
        # Volatility Efficiency
        df['eff_momentum'] * (1 - df['price_range_10d']) +
        
        # Volume-Regime Reversal
        df['combined_extreme'] * df['regime_persistence'] * np.where(df['vol_regime'] > 1, 1, 0.5) +
        
        # Flow Persistence
        df['flow_consistency'] * df['flow_accel'] * df['flow_regime'] +
        
        # Momentum Convergence
        df['mom_alignment'] * (1 - df['vol_std'])
    )
    
    return alpha
