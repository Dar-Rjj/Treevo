import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Efficiency-Adaptive Gap Momentum Factor
    Combines gap efficiency, volatility regimes, and volume coordination
    to predict future stock returns.
    """
    data = df.copy()
    
    # Gap Efficiency Component
    data['overnight_gap'] = (data['open'] / data['close'].shift(1) - 1) * data['volume']
    data['intraday_efficiency'] = (data['close'] / data['open'] - 1) * data['volume']
    data['gap_efficiency'] = data['overnight_gap'] + data['intraday_efficiency']
    
    # Gap persistence streak (3-day direction consistency)
    data['gap_direction'] = np.sign(data['overnight_gap'])
    data['gap_streak'] = 0
    for i in range(2, len(data)):
        if (data['gap_direction'].iloc[i] == data['gap_direction'].iloc[i-1] == 
            data['gap_direction'].iloc[i-2]):
            data.loc[data.index[i], 'gap_streak'] = 1
        else:
            data.loc[data.index[i], 'gap_streak'] = 0
    
    # Volatility-Efficiency Regime
    # True Range calculation
    data['tr1'] = data['high'] - data['low']
    data['tr2'] = abs(data['high'] - data['close'].shift(1))
    data['tr3'] = abs(data['low'] - data['close'].shift(1))
    data['true_range'] = data[['tr1', 'tr2', 'tr3']].max(axis=1)
    data['atr_10'] = data['true_range'].rolling(window=10).mean()
    data['true_range_efficiency'] = data['atr_10'] / data['atr_10'].shift(10)
    
    # Price efficiency momentum (5-day net price change)
    data['price_efficiency'] = (data['close'] - data['close'].shift(5)) / data['close'].shift(5)
    
    # Volume Coordination Multiplier
    # Cumulative signed volume flow
    data['signed_volume'] = np.sign(data['close'] - data['open']) * data['volume']
    data['cum_signed_volume_10'] = data['signed_volume'].rolling(window=10).sum()
    data['cum_abs_volume_10'] = data['volume'].rolling(window=10).sum()
    data['volume_efficiency_ratio'] = data['cum_signed_volume_10'] / data['cum_abs_volume_10']
    
    # Volume momentum (5-day change in volume efficiency)
    data['volume_momentum'] = data['volume_efficiency_ratio'] - data['volume_efficiency_ratio'].shift(5)
    
    # Regime classification
    vol_threshold = data['true_range_efficiency'].rolling(window=20).mean()
    eff_threshold = data['price_efficiency'].abs().rolling(window=20).mean()
    
    data['high_vol_regime'] = (data['true_range_efficiency'] > vol_threshold).astype(int)
    data['low_efficiency_regime'] = (data['price_efficiency'].abs() < eff_threshold).astype(int)
    
    # Adaptive Combination
    # Base factor
    data['base_factor'] = data['gap_efficiency'] * data['volume_efficiency_ratio']
    
    # Regime-specific multipliers
    high_vol_low_eff_mask = (data['high_vol_regime'] == 1) & (data['low_efficiency_regime'] == 1)
    low_vol_high_eff_mask = (data['high_vol_regime'] == 0) & (data['low_efficiency_regime'] == 0)
    
    data['regime_multiplier'] = 1.0
    data.loc[high_vol_low_eff_mask, 'regime_multiplier'] = data['gap_streak']
    data.loc[low_vol_high_eff_mask, 'regime_multiplier'] = data['volume_momentum']
    
    # Final alpha factor
    data['alpha_factor'] = data['base_factor'] * data['regime_multiplier']
    
    # Clean up intermediate columns
    result = data['alpha_factor'].copy()
    
    return result
