import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Volatility-Volume Adaptive Momentum Reversal Factor
    """
    # Copy data to avoid modifying original
    data = df.copy()
    
    # Volatility Regime Detection
    # True Range Calculation
    data['prev_close'] = data['close'].shift(1)
    data['tr1'] = data['high'] - data['low']
    data['tr2'] = abs(data['high'] - data['prev_close'])
    data['tr3'] = abs(data['low'] - data['prev_close'])
    data['true_range'] = data[['tr1', 'tr2', 'tr3']].max(axis=1)
    data['norm_true_range'] = data['true_range'] / data['prev_close']
    
    # Volatility Baseline (20-day rolling median)
    data['vol_baseline'] = data['norm_true_range'].rolling(window=20, min_periods=10).median()
    
    # Volume Acceleration Analysis
    data['vol_ma_3'] = data['volume'].rolling(window=3, min_periods=2).mean()
    data['vol_ma_10'] = data['volume'].rolling(window=10, min_periods=5).mean()
    data['volume_ratio'] = data['vol_ma_3'] / data['vol_ma_10']
    
    # Amount Confirmation
    data['amount_vol_ratio'] = data['amount'] / data['volume']
    data['avg_amount_vol_ratio'] = data['amount_vol_ratio'].rolling(window=5, min_periods=3).mean()
    data['large_trade_indicator'] = data['amount_vol_ratio'] > (1.2 * data['avg_amount_vol_ratio'])
    
    # Price Momentum Reversal Signal
    # Multi-Timeframe Momentum
    data['mom_2d'] = data['close'] / data['close'].shift(2) - 1
    data['mom_5d'] = data['close'] / data['close'].shift(5) - 1
    data['mom_10d'] = data['close'] / data['close'].shift(10) - 1
    
    # Momentum Divergence Detection
    data['short_medium_div'] = data['mom_2d'] - data['mom_10d']
    data['acceleration_signal'] = (data['mom_2d'] - data['mom_5d']) * (data['mom_5d'] - data['mom_10d'])
    data['reversal_potential'] = -1 * data['acceleration_signal']
    
    # Price Extremes Positioning
    data['high_5d'] = data['high'].rolling(window=5, min_periods=3).max()
    data['low_5d'] = data['low'].rolling(window=5, min_periods=3).min()
    data['price_pos_5d'] = (data['close'] - data['low_5d']) / (data['high_5d'] - data['low_5d'])
    
    data['high_10d'] = data['high'].rolling(window=10, min_periods=5).max()
    data['low_10d'] = data['low'].rolling(window=10, min_periods=5).min()
    data['price_pos_10d'] = (data['close'] - data['low_10d']) / (data['high_10d'] - data['low_10d'])
    
    data['extreme_reversion'] = -1 * (abs(data['price_pos_5d'] - 0.5) + abs(data['price_pos_10d'] - 0.5))
    
    # Initialize base signal
    data['base_signal'] = 0.0
    
    # Adaptive Signal Integration based on volatility regimes and volume acceleration
    for i in range(len(data)):
        if pd.isna(data['vol_baseline'].iloc[i]) or pd.isna(data['norm_true_range'].iloc[i]):
            continue
            
        # Volatility Regime Classification
        vol_ratio = data['norm_true_range'].iloc[i] / data['vol_baseline'].iloc[i]
        
        # Volume Acceleration Classification
        vol_accel = data['volume_ratio'].iloc[i] if not pd.isna(data['volume_ratio'].iloc[i]) else 1.0
        
        # Get reversal components
        reversal_pot = data['reversal_potential'].iloc[i] if not pd.isna(data['reversal_potential'].iloc[i]) else 0.0
        extreme_rev = data['extreme_reversion'].iloc[i] if not pd.isna(data['extreme_reversion'].iloc[i]) else 0.0
        
        # Apply multipliers based on volatility regime and volume acceleration
        if vol_ratio > 1.8:  # High Volatility
            if vol_accel > 1.4:
                data.loc[data.index[i], 'base_signal'] = 2.5 * reversal_pot + 1.5 * extreme_rev
            elif vol_accel > 1.1:
                data.loc[data.index[i], 'base_signal'] = 2.0 * reversal_pot + 1.2 * extreme_rev
            elif vol_accel >= 0.9:
                data.loc[data.index[i], 'base_signal'] = 1.5 * reversal_pot + 1.0 * extreme_rev
            else:
                data.loc[data.index[i], 'base_signal'] = 1.0 * reversal_pot + 0.7 * extreme_rev
                
        elif vol_ratio > 1.2:  # Elevated Volatility
            if vol_accel > 1.4:
                data.loc[data.index[i], 'base_signal'] = 2.0 * reversal_pot + 1.3 * extreme_rev
            elif vol_accel > 1.1:
                data.loc[data.index[i], 'base_signal'] = 1.5 * reversal_pot + 1.0 * extreme_rev
            elif vol_accel >= 0.9:
                data.loc[data.index[i], 'base_signal'] = 1.2 * reversal_pot + 0.8 * extreme_rev
            else:
                data.loc[data.index[i], 'base_signal'] = 0.8 * reversal_pot + 0.5 * extreme_rev
                
        elif vol_ratio >= 0.8:  # Normal Volatility
            if vol_accel > 1.4:
                data.loc[data.index[i], 'base_signal'] = 1.5 * reversal_pot + 1.0 * extreme_rev
            elif vol_accel > 1.1:
                data.loc[data.index[i], 'base_signal'] = 1.2 * reversal_pot + 0.8 * extreme_rev
            elif vol_accel >= 0.9:
                data.loc[data.index[i], 'base_signal'] = 1.0 * reversal_pot + 0.7 * extreme_rev
            else:
                data.loc[data.index[i], 'base_signal'] = 0.6 * reversal_pot + 0.4 * extreme_rev
                
        else:  # Low Volatility
            if vol_accel > 1.4:
                data.loc[data.index[i], 'base_signal'] = 1.0 * reversal_pot + 0.5 * extreme_rev
            elif vol_accel > 1.1:
                data.loc[data.index[i], 'base_signal'] = 0.7 * reversal_pot + 0.3 * extreme_rev
            elif vol_accel >= 0.9:
                data.loc[data.index[i], 'base_signal'] = 0.5 * reversal_pot + 0.2 * extreme_rev
            else:
                data.loc[data.index[i], 'base_signal'] = 0.3 * reversal_pot + 0.1 * extreme_rev
    
    # Final Alpha Output with multipliers
    data['trade_boost'] = 1.3
    data['amount_confirmation'] = 1.2
    
    # Apply conditional multipliers
    data.loc[~data['large_trade_indicator'], 'trade_boost'] = 1.0
    data.loc[data['amount_vol_ratio'] <= data['avg_amount_vol_ratio'], 'amount_confirmation'] = 1.0
    
    # Final factor calculation
    data['alpha_factor'] = data['base_signal'] * data['trade_boost'] * data['amount_confirmation']
    
    return data['alpha_factor']
