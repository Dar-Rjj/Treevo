import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    data = df.copy()
    
    # Price Momentum Divergence Analysis
    data['short_term_momentum'] = data['close'] / data['close'].shift(3) - 1
    data['medium_term_momentum'] = data['close'] / data['close'].shift(8) - 1
    data['momentum_acceleration'] = (data['short_term_momentum'] - data['medium_term_momentum']) / 5
    data['divergence_ratio'] = data['short_term_momentum'] / data['medium_term_momentum']
    
    # Volume Behavior Patterns
    data['volume_trend'] = data['volume'] / data['volume'].shift(5)
    
    # Volume-price correlation (4-day window)
    volume_price_corr = []
    for i in range(len(data)):
        if i >= 4:
            vol_window = data['volume'].iloc[i-4:i+1]
            close_window = data['close'].iloc[i-4:i+1]
            corr_val = vol_window.corr(close_window) if len(vol_window) > 1 and vol_window.std() > 0 and close_window.std() > 0 else 0
            volume_price_corr.append(corr_val)
        else:
            volume_price_corr.append(0)
    data['volume_price_alignment'] = volume_price_corr
    
    data['volume_persistence'] = ((data['volume'] > data['volume'].shift(1)) & 
                                 (data['volume'].shift(1) > data['volume'].shift(2))).astype(int)
    
    # Reversal Detection Signals
    # Recent High-Low Reversals (past 5 days)
    high_low_reversals = []
    for i in range(len(data)):
        if i >= 5:
            count = 0
            for j in range(i-4, i+1):
                if (data['high'].iloc[j] > data['high'].iloc[j-1]) and (data['close'].iloc[j] < data['close'].iloc[j-1]):
                    count += 1
            high_low_reversals.append(count)
        else:
            high_low_reversals.append(0)
    data['Recent_High_Low_Reversals'] = high_low_reversals
    
    # Gap Recovery Strength
    data['Gap_Recovery_Strength'] = (data['open'] - data['close'].shift(1)) / (data['high'] - data['low'])
    data['Gap_Recovery_Strength'] = data['Gap_Recovery_Strength'].replace([np.inf, -np.inf], 0).fillna(0)
    
    # Failed Breakouts (past 3 days)
    failed_breakouts = []
    for i in range(len(data)):
        if i >= 3:
            count = 0
            for j in range(i-2, i+1):
                if (data['high'].iloc[j] > data['high'].iloc[j-1]) and (data['close'].iloc[j] < data['open'].iloc[j]):
                    count += 1
            failed_breakouts.append(count)
        else:
            failed_breakouts.append(0)
    data['Failed_Breakouts'] = failed_breakouts
    
    # Market Regime Classification
    data['high_volatility_regime'] = (data['high'] - data['low']) > (0.02 * data['close'])
    data['strong_trend_regime'] = abs(data['close'] / data['close'].shift(10) - 1) > 0.05
    data['volume_surge_regime'] = data['volume'] > (1.5 * data['volume'].shift(1).rolling(window=5, min_periods=1).mean())
    
    # Regime-Adaptive Alpha Signals
    volatile_signal = -data['divergence_ratio'] * data['volume_trend'] * data['Recent_High_Low_Reversals']
    trending_signal = data['momentum_acceleration'] * data['volume_price_alignment'] * data['Gap_Recovery_Strength']
    volume_driven_signal = data['divergence_ratio'] * data['volume_persistence'] * data['Failed_Breakouts']
    
    # Combine signals based on regimes
    alpha_signal = np.where(data['high_volatility_regime'], volatile_signal,
                   np.where(data['strong_trend_regime'], trending_signal,
                   np.where(data['volume_surge_regime'], volume_driven_signal, 0)))
    
    return pd.Series(alpha_signal, index=data.index)
