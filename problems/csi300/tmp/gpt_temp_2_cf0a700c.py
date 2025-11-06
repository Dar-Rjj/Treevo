import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Adaptive Momentum-Volume Regime Factor
    Combines multi-timeframe momentum with volume acceleration and regime detection
    """
    # Copy data to avoid modifying original
    data = df.copy()
    
    # Multi-Timeframe Momentum with Volatility Adjustment
    # Calculate momentum components
    data['momentum_3d'] = data['close'] / data['close'].shift(3) - 1
    data['momentum_8d'] = data['close'] / data['close'].shift(8) - 1
    data['momentum_21d'] = data['close'] / data['close'].shift(21) - 1
    
    # Momentum blend (product of all timeframes)
    data['momentum_blend'] = data['momentum_3d'] * data['momentum_8d'] * data['momentum_21d']
    
    # Volatility scaling using Average True Range (5-day)
    # True Range components
    data['tr_hl'] = data['high'] - data['low']
    data['tr_hc'] = abs(data['high'] - data['close'].shift(1))
    data['tr_lc'] = abs(data['low'] - data['close'].shift(1))
    
    # True Range = max of three components
    data['true_range'] = data[['tr_hl', 'tr_hc', 'tr_lc']].max(axis=1)
    
    # ATR_5 = Average of True Range over 5 days
    data['atr_5'] = data['true_range'].rolling(window=5).mean()
    
    # Volume Acceleration with Adaptive Smoothing
    # Volume momentum components
    data['volume_change_3d'] = data['volume'] / data['volume'].shift(3)
    data['volume_change_8d'] = data['volume'] / data['volume'].shift(8)
    data['volume_acceleration'] = data['volume_change_3d'] / data['volume_change_8d']
    
    # Adaptive smoothing based on recent volatility
    data['volatility_regime'] = data['true_range'] > data['atr_5']
    
    # Apply different EMA smoothing based on volatility regime
    vol_acc_smoothed = []
    for i in range(len(data)):
        if i < 5:  # Need enough data for EMA
            vol_acc_smoothed.append(np.nan)
            continue
            
        if data['volatility_regime'].iloc[i]:
            # High volatility: 2-day EMA
            window_data = data['volume_acceleration'].iloc[max(0, i-1):i+1]
            ema_value = window_data.ewm(span=2).mean().iloc[-1]
        else:
            # Low volatility: 5-day EMA
            window_data = data['volume_acceleration'].iloc[max(0, i-4):i+1]
            ema_value = window_data.ewm(span=5).mean().iloc[-1]
        vol_acc_smoothed.append(ema_value)
    
    data['smoothed_volume_acceleration'] = vol_acc_smoothed
    
    # Regime-Aware Combination
    # Calculate EMA(Close, 10) for regime detection
    data['ema_10'] = data['close'].ewm(span=10).mean()
    
    # Time-Decay Emphasis
    data['recent_perf_weight'] = (data['close'] / data['close'].shift(1)) ** 2
    data['volume_trend_emphasis'] = (data['volume'] / data['volume'].shift(3)) ** 0.5
    data['time_decay_emphasis'] = data['recent_perf_weight'] * data['volume_trend_emphasis']
    
    # Final Alpha Factor Construction
    alpha_values = []
    
    for i in range(len(data)):
        if (pd.isna(data['momentum_blend'].iloc[i]) or 
            pd.isna(data['smoothed_volume_acceleration'].iloc[i]) or
            pd.isna(data['time_decay_emphasis'].iloc[i]) or
            pd.isna(data['atr_5'].iloc[i]) or
            pd.isna(data['ema_10'].iloc[i])):
            alpha_values.append(np.nan)
            continue
        
        # Regime-based core factor
        if data['close'].iloc[i] > data['ema_10'].iloc[i]:
            # Bull regime
            core_factor = data['momentum_blend'].iloc[i] * data['smoothed_volume_acceleration'].iloc[i]
        else:
            # Bear regime
            core_factor = data['momentum_blend'].iloc[i] / data['smoothed_volume_acceleration'].iloc[i]
        
        # Apply time decay and volatility normalization
        final_factor = core_factor * data['time_decay_emphasis'].iloc[i] * (1 / data['atr_5'].iloc[i])
        alpha_values.append(final_factor)
    
    # Return as pandas Series with same index as input
    return pd.Series(alpha_values, index=data.index, name='adaptive_momentum_volume_factor')
