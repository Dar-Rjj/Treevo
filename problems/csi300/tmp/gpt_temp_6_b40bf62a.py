import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Momentum Reversal Detection
    # Short-term Reversal Patterns
    data['failed_breakout'] = ((data['high'] > data['high'].shift(1)) & 
                              (data['close'] < data['open'])).astype(int)
    data['failed_breakout_count'] = data['failed_breakout'].rolling(window=3, min_periods=1).sum()
    
    data['gap_recovery_strength'] = (data['open'] - data['close'].shift(1)) / (data['high'] - data['low'])
    data['gap_recovery_strength'] = data['gap_recovery_strength'].replace([np.inf, -np.inf], np.nan)
    
    # Multi-timeframe Momentum Divergence
    data['short_term_momentum'] = data['close'] / data['close'].shift(5) - 1
    data['medium_term_momentum'] = data['close'] / data['close'].shift(20) - 1
    data['momentum_divergence'] = (data['close'] / data['close'].shift(5)) - (data['close'] / data['close'].shift(20))
    
    data['momentum_acceleration'] = data['short_term_momentum'] / data['medium_term_momentum']
    data['momentum_acceleration'] = data['momentum_acceleration'].replace([np.inf, -np.inf], np.nan)
    
    # Volume-Price Synchronization Analysis
    # Volume Divergence Signals
    data['volume_ma_10'] = data['volume'].rolling(window=10, min_periods=1).mean()
    data['abnormal_volume'] = data['volume'] / data['volume_ma_10'].shift(1)
    
    data['volume_price_divergence'] = (data['volume'] / data['volume'].shift(1)) - (data['close'] / data['close'].shift(1))
    
    # Volume-Momentum Correlation
    data['volume_ma_20'] = data['volume'].rolling(window=20, min_periods=1).mean()
    data['volume_stability'] = data['volume'] / data['volume_ma_20'].shift(1)
    
    # Calculate 5-day volume-price correlation
    volume_price_corr = []
    for i in range(len(data)):
        if i >= 4:
            vol_window = data['volume'].iloc[i-4:i+1]
            price_window = data['close'].iloc[i-4:i+1]
            corr = vol_window.corr(price_window)
            volume_price_corr.append(corr if not pd.isna(corr) else 0)
        else:
            volume_price_corr.append(0)
    data['volume_price_correlation'] = volume_price_corr
    
    # Intraday Efficiency Assessment
    # Momentum Efficiency
    data['intraday_efficiency'] = (data['close'] - data['open']) / (data['high'] - data['low'])
    data['intraday_efficiency'] = data['intraday_efficiency'].replace([np.inf, -np.inf], np.nan)
    
    data['cumulative_pressure'] = data['intraday_efficiency'].rolling(window=3, min_periods=1).sum()
    
    # Range Analysis
    data['daily_range'] = data['high'] - data['low']
    data['range_ma_5'] = data['daily_range'].rolling(window=5, min_periods=1).mean()
    data['range_expansion'] = data['daily_range'] / data['range_ma_5'].shift(1)
    
    data['absolute_efficiency'] = abs(data['intraday_efficiency'])
    
    # Composite Alpha Generation
    # Reversal signals (negative for reversal detection)
    reversal_signal = -data['failed_breakout_count'] * data['gap_recovery_strength']
    
    # Volume synchronization component
    volume_sync = (data['abnormal_volume'] * data['volume_stability'] * 
                   (1 + data['volume_price_correlation']))
    
    # Combine reversal with volume synchronization
    reversal_volume_component = reversal_signal * volume_sync
    
    # Intraday efficiency weighting
    efficiency_weight = data['cumulative_pressure'] * data['absolute_efficiency']
    
    # Momentum divergence component
    momentum_component = data['momentum_divergence'] * data['momentum_acceleration']
    
    # Final composite alpha
    alpha = (reversal_volume_component * efficiency_weight * momentum_component)
    
    # Clean and return
    alpha = alpha.replace([np.inf, -np.inf], np.nan).fillna(0)
    return alpha
