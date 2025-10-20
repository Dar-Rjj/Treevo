import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    data = df.copy()
    
    # Multi-Timeframe Momentum Components
    data['mom_5d'] = data['close'] / data['close'].shift(5) - 1
    data['mom_10d'] = data['close'] / data['close'].shift(10) - 1
    data['mom_accel'] = data['mom_5d'] - data['mom_10d']
    
    # Momentum persistence (sign consistency over 5 days)
    data['mom_sign_1d'] = np.sign(data['close'] / data['close'].shift(1) - 1)
    data['mom_sign_5d'] = np.sign(data['mom_5d'])
    data['mom_persistence'] = data['mom_sign_1d'].rolling(window=5).apply(
        lambda x: np.mean(x == x.iloc[-1]) if len(x) == 5 else np.nan, raw=False
    )
    
    # Volume Dynamics and Convergence
    data['vol_mom_5d'] = data['volume'] / data['volume'].shift(5) - 1
    data['vol_mom_10d'] = data['volume'] / data['volume'].shift(10) - 1
    data['vol_accel'] = data['vol_mom_5d'] - data['vol_mom_10d']
    
    # Momentum-volume directional alignment
    data['mom_vol_alignment'] = np.sign(data['mom_5d']) * np.sign(data['vol_mom_5d'])
    data['alignment_strength'] = (np.sign(data['mom_5d']) * np.sign(data['vol_mom_5d']) * 
                                 np.minimum(np.abs(data['mom_5d']), np.abs(data['vol_mom_5d'])))
    
    # Volume-adjusted price range
    data['vol_adj_range'] = (data['high'] - data['low']) * data['volume']
    
    # Volatility Context and Regimes
    # True Range components
    data['TR1'] = data['high'] - data['low']
    data['TR2'] = np.abs(data['high'] - data['close'].shift(1))
    data['TR3'] = np.abs(data['low'] - data['close'].shift(1))
    data['true_range'] = data[['TR1', 'TR2', 'TR3']].max(axis=1)
    data['ATR_10d'] = data['true_range'].rolling(window=10).mean()
    
    # Return volatility
    data['ret_20d_std'] = (data['close'] / data['close'].shift(1) - 1).rolling(window=20).std()
    
    # Volatility regime
    data['ATR_60d_median'] = data['ATR_10d'].rolling(window=60).median()
    data['vol_regime'] = np.where(data['ATR_10d'] > data['ATR_60d_median'], 'high', 'low')
    
    # Convergence-Divergence Signals
    data['convergence_strength'] = data['mom_accel'] * data['vol_accel']
    
    # Apply directional consistency weighting
    data['directional_weight'] = data['mom_persistence'] * data['alignment_strength']
    data['weighted_convergence'] = data['convergence_strength'] * data['directional_weight']
    
    # Volatility-Regime Adaptive Filtering
    # High volatility: emphasize momentum continuation
    high_vol_mask = data['vol_regime'] == 'high'
    data['high_vol_signal'] = np.where(
        high_vol_mask & (data['alignment_strength'] > 0),
        data['weighted_convergence'] * data['ATR_10d'],
        0
    )
    
    # Low volatility: emphasize mean reversion
    low_vol_mask = data['vol_regime'] == 'low'
    data['low_vol_signal'] = np.where(
        low_vol_mask & (data['alignment_strength'] < 0),
        data['weighted_convergence'] / (data['ATR_10d'] + 1e-8),
        0
    )
    
    # Generate final alpha factor
    data['alpha_factor'] = data['high_vol_signal'] + data['low_vol_signal']
    
    # Apply trend confirmation filter
    data['final_alpha'] = np.where(
        data['mom_persistence'] > 0.6,
        data['alpha_factor'],
        data['alpha_factor'] * 0.5  # Reduce signal strength for weak trends
    )
    
    return data['final_alpha']
