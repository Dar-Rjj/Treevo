import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Price-Volume Efficiency Momentum Divergence factor that captures regime transitions
    and efficiency momentum patterns across multiple timeframes.
    """
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Calculate basic price efficiency (return per unit range)
    df['price_efficiency'] = (df['close'] - df['close'].shift(1)) / (df['high'] - df['low']).replace(0, np.nan)
    
    # Calculate volume efficiency (volume per unit price range)
    df['volume_efficiency'] = df['volume'] / (df['high'] - df['low']).replace(0, np.nan)
    
    # Multi-timeframe price efficiency calculations
    df['price_eff_5d'] = (df['close'] - df['close'].shift(5)) / (df['high'].rolling(5).max() - df['low'].rolling(5).min()).replace(0, np.nan)
    df['price_eff_10d'] = (df['close'] - df['close'].shift(10)) / (df['high'].rolling(10).max() - df['low'].rolling(10).min()).replace(0, np.nan)
    df['price_eff_21d'] = (df['close'] - df['close'].shift(21)) / (df['high'].rolling(21).max() - df['low'].rolling(21).min()).replace(0, np.nan)
    
    # Price efficiency momentum
    df['price_eff_mom_5d'] = df['price_eff_5d'] - df['price_eff_5d'].shift(5)
    df['price_eff_mom_10d'] = df['price_eff_10d'] - df['price_eff_10d'].shift(10)
    
    # Price efficiency acceleration
    df['price_eff_accel_5v21'] = df['price_eff_5d'] - df['price_eff_21d']
    df['price_eff_accel_5v10'] = df['price_eff_5d'] - df['price_eff_10d']
    
    # Volume efficiency momentum
    df['vol_eff_5d'] = df['volume_efficiency'].rolling(5).mean()
    df['vol_eff_10d'] = df['volume_efficiency'].rolling(10).mean()
    df['vol_eff_21d'] = df['volume_efficiency'].rolling(21).mean()
    
    df['vol_eff_mom_5d'] = df['vol_eff_5d'] - df['vol_eff_5d'].shift(5)
    df['vol_eff_mom_10d'] = df['vol_eff_10d'] - df['vol_eff_10d'].shift(10)
    
    # Volume efficiency acceleration
    df['vol_eff_accel_5v21'] = df['vol_eff_5d'] - df['vol_eff_21d']
    df['vol_eff_accel_5v10'] = df['vol_eff_5d'] - df['vol_eff_10d']
    
    # Price-volume efficiency divergence
    df['pv_divergence'] = np.sign(df['price_eff_mom_5d']) * np.sign(df['vol_eff_mom_5d'])
    df['pv_divergence_strength'] = abs(df['price_eff_mom_5d'] - df['vol_eff_mom_5d'])
    
    # Efficiency regime classification
    df['price_eff_regime'] = np.where(df['price_eff_5d'] > df['price_eff_5d'].rolling(20).mean(), 1, -1)
    df['vol_eff_regime'] = np.where(df['vol_eff_5d'] > df['vol_eff_5d'].rolling(20).mean(), 1, -1)
    
    # Range momentum analysis
    df['daily_range'] = df['high'] - df['low']
    df['range_mom_5d'] = (df['daily_range'] - df['daily_range'].shift(5)) / df['daily_range'].shift(5)
    df['range_mom_10d'] = (df['daily_range'] - df['daily_range'].shift(10)) / df['daily_range'].shift(10)
    
    # Efficiency persistence analysis
    df['price_eff_persistence'] = 0
    df['vol_eff_persistence'] = 0
    
    for i in range(1, len(df)):
        if df['price_eff_mom_5d'].iloc[i] > df['price_eff_mom_5d'].iloc[i-1]:
            df.loc[df.index[i], 'price_eff_persistence'] = df['price_eff_persistence'].iloc[i-1] + 1
        elif df['price_eff_mom_5d'].iloc[i] < df['price_eff_mom_5d'].iloc[i-1]:
            df.loc[df.index[i], 'price_eff_persistence'] = df['price_eff_persistence'].iloc[i-1] - 1
        else:
            df.loc[df.index[i], 'price_eff_persistence'] = df['price_eff_persistence'].iloc[i-1]
            
        if df['vol_eff_mom_5d'].iloc[i] > df['vol_eff_mom_5d'].iloc[i-1]:
            df.loc[df.index[i], 'vol_eff_persistence'] = df['vol_eff_persistence'].iloc[i-1] + 1
        elif df['vol_eff_mom_5d'].iloc[i] < df['vol_eff_mom_5d'].iloc[i-1]:
            df.loc[df.index[i], 'vol_eff_persistence'] = df['vol_eff_persistence'].iloc[i-1] - 1
        else:
            df.loc[df.index[i], 'vol_eff_persistence'] = df['vol_eff_persistence'].iloc[i-1]
    
    # Efficiency regime transition analysis
    df['price_regime_change'] = df['price_eff_regime'] != df['price_eff_regime'].shift(1)
    df['vol_regime_change'] = df['vol_eff_regime'] != df['vol_eff_regime'].shift(1)
    
    df['regime_transition_momentum'] = (df['price_regime_change'].astype(int) + df['vol_regime_change'].astype(int)) * \
                                     (df['price_eff_mom_5d'] + df['vol_eff_mom_5d'])
    
    # Cross-regime efficiency divergence
    df['cross_regime_divergence'] = np.where(df['price_eff_regime'] != df['vol_eff_regime'], 
                                           abs(df['price_eff_mom_5d'] - df['vol_eff_mom_5d']), 0)
    
    # Range-adjusted efficiency weighting
    range_regime = np.where(df['daily_range'] > df['daily_range'].rolling(20).mean(), 1, -1)
    df['range_adjusted_efficiency'] = df['price_eff_5d'] * range_regime
    
    # Composite efficiency divergence factor
    # Strong positive signals: multi-timeframe convergence + volume confirmation + range expansion
    strong_efficiency_signal = (
        (df['price_eff_accel_5v10'] > 0) & 
        (df['price_eff_accel_5v21'] > 0) &
        (df['vol_eff_mom_5d'] > 0) &
        (df['range_mom_5d'] > 0) &
        (df['price_eff_persistence'] > 0)
    )
    
    # Divergence reversal signals
    divergence_reversal_signal = (
        (df['pv_divergence'] < 0) &
        (df['cross_regime_divergence'] > df['cross_regime_divergence'].rolling(10).mean()) &
        (df['regime_transition_momentum'] > 0)
    )
    
    # Final composite factor
    factor = (
        # Base efficiency momentum
        df['price_eff_mom_5d'] * 0.3 +
        df['vol_eff_mom_5d'] * 0.2 +
        
        # Multi-timeframe alignment
        (df['price_eff_accel_5v10'] + df['price_eff_accel_5v21']) * 0.15 +
        
        # Regime transition signals
        df['regime_transition_momentum'] * 0.1 +
        
        # Range-adjusted efficiency
        df['range_adjusted_efficiency'] * 0.1 +
        
        # Persistence strength
        (df['price_eff_persistence'] + df['vol_eff_persistence']) * 0.05 +
        
        # Signal amplification
        strong_efficiency_signal.astype(float) * 0.05 +
        divergence_reversal_signal.astype(float) * 0.05
    )
    
    # Normalize the factor
    factor = (factor - factor.rolling(63).mean()) / factor.rolling(63).std()
    
    return factor
