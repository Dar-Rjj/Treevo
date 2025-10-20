import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate Multi-Period Momentum
    data['ret_1d'] = data['close'].pct_change(1)
    data['ret_5d'] = data['close'].pct_change(5)
    data['ret_21d'] = data['close'].pct_change(21)
    
    # Calculate Volume Dynamics
    data['vol_chg_5d'] = data['volume'].pct_change(5)
    data['vol_chg_21d'] = data['volume'].pct_change(21)
    data['vol_adj_range'] = (data['high'] - data['low']) * data['volume']
    
    # Derive Acceleration Signals
    data['mom_accel'] = data['ret_5d'] - data['ret_21d']
    data['vol_accel'] = data['vol_chg_5d'] - data['vol_chg_21d']
    data['range_accel'] = data['vol_adj_range'] / data['vol_adj_range'].rolling(window=5, min_periods=1).mean() - 1
    
    # Detect Volatility Regime
    data['volatility_20d'] = data['ret_1d'].rolling(window=20, min_periods=10).std()
    data['volatility_median_60d'] = data['volatility_20d'].rolling(window=60, min_periods=30).median()
    data['high_vol_regime'] = data['volatility_20d'] > data['volatility_median_60d']
    
    # Initialize factor values
    factor_values = pd.Series(index=data.index, dtype=float)
    
    # Generate Regime-Adaptive Signal
    for i in range(len(data)):
        if i < 21:  # Ensure we have enough data for calculations
            factor_values.iloc[i] = 0
            continue
            
        current = data.iloc[i]
        
        if current['high_vol_regime']:
            # High Volatility Regime
            if (current['mom_accel'] > 0 and current['vol_accel'] > 0) or (current['mom_accel'] < 0 and current['vol_accel'] < 0):
                # Momentum continuation
                signal_strength = current['mom_accel'] * (1 + abs(current['vol_accel']))
                if current['range_accel'] > 0:
                    signal_strength *= (1 + current['range_accel'])
                factor_values.iloc[i] = signal_strength
            else:
                # Strong reversal
                signal_strength = -current['mom_accel'] * (1 + abs(current['vol_accel']))
                if current['range_accel'] > 0:
                    signal_strength *= (1 + current['range_accel'])
                factor_values.iloc[i] = signal_strength
        else:
            # Low Volatility Regime
            if (current['mom_accel'] > 0 and current['vol_accel'] > 0) or (current['mom_accel'] < 0 and current['vol_accel'] < 0):
                # Trend persistence
                signal_strength = current['mom_accel'] * (1 + 0.5 * abs(current['vol_accel']))
                if current['range_accel'] > 0:
                    signal_strength *= (1 + 0.3 * current['range_accel'])
                factor_values.iloc[i] = signal_strength
            else:
                # Weak mean reversion
                signal_strength = -0.5 * current['mom_accel'] * (1 + abs(current['vol_accel']))
                if current['range_accel'] > 0:
                    signal_strength *= (1 + 0.2 * current['range_accel'])
                factor_values.iloc[i] = signal_strength
    
    # Fill any remaining NaN values
    factor_values = factor_values.fillna(0)
    
    return factor_values
