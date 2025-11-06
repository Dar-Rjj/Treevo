import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Volume-Range Momentum Divergence Factor
    """
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Range-Momentum Divergence Detection
    # Range Expansion Component
    data['range_ratio'] = (data['high'] - data['low']) / data['close']
    data['avg_range_ratio_5d'] = data['range_ratio'].rolling(window=5, min_periods=5).mean()
    data['range_expansion'] = data['range_ratio'] / data['avg_range_ratio_5d']
    
    # Multi-Timeframe Momentum Divergence
    data['momentum_short'] = data['close'] - data['close'].shift(1)
    data['momentum_medium'] = data['close'] - data['close'].shift(5)
    
    # Divergence Type
    data['positive_divergence'] = (
        (data['range_expansion'] > 1.2) & 
        (data['momentum_short'] < 0) & 
        (data['momentum_medium'] > 0)
    )
    data['negative_divergence'] = (
        (data['range_expansion'] > 1.2) & 
        (data['momentum_short'] > 0) & 
        (data['momentum_medium'] < 0)
    )
    
    # Volume-Weighted Acceleration Integration
    # Price Acceleration Calculation
    data['momentum_change'] = (
        (data['close'] - data['close'].shift(1)) - 
        (data['close'].shift(1) - data['close'].shift(2))
    )
    data['volume_weighted_accel'] = data['momentum_change'] * data['volume']
    
    # Multi-Timeframe Acceleration
    data['cumulative_accel_3d'] = data['volume_weighted_accel'].rolling(window=3, min_periods=3).sum()
    data['cumulative_accel_10d'] = data['volume_weighted_accel'].rolling(window=10, min_periods=10).sum()
    
    # Volatility-Regime Assessment
    # True Range Calculation
    data['tr1'] = data['high'] - data['low']
    data['tr2'] = abs(data['high'] - data['close'].shift(1))
    data['tr3'] = abs(data['low'] - data['close'].shift(1))
    data['true_range'] = data[['tr1', 'tr2', 'tr3']].max(axis=1)
    
    # Volatility Regime Classification
    data['current_atr'] = data['true_range'].rolling(window=14, min_periods=14).mean()
    data['previous_atr'] = data['true_range'].shift(14).rolling(window=14, min_periods=14).mean()
    data['volatility_ratio'] = data['current_atr'] / data['previous_atr']
    
    # Volume-Range Confirmation
    data['volume_change_ratio'] = data['volume'] / data['volume'].shift(5)
    data['strong_confirmation'] = (data['range_expansion'] > 1.2) & (data['volume_change_ratio'] > 1.3)
    data['weak_confirmation'] = (data['range_expansion'] > 1.2) & (data['volume_change_ratio'] < 0.8)
    
    # Combined Alpha Signal Generation
    alpha_signal = pd.Series(index=data.index, dtype=float)
    
    for i in range(len(data)):
        if pd.isna(data.iloc[i]['current_atr']) or pd.isna(data.iloc[i]['previous_atr']):
            alpha_signal.iloc[i] = 0
            continue
            
        volatility_ratio = data.iloc[i]['volatility_ratio']
        positive_div = data.iloc[i]['positive_divergence']
        negative_div = data.iloc[i]['negative_divergence']
        strong_conf = data.iloc[i]['strong_confirmation']
        
        # High Volatility Regime
        if volatility_ratio > 1.3:
            if negative_div and strong_conf:
                # Bearish signal with 10-day acceleration weighting
                alpha_signal.iloc[i] = -data.iloc[i]['cumulative_accel_10d']
            else:
                alpha_signal.iloc[i] = 0
                
        # Low Volatility Regime
        elif volatility_ratio < 0.7:
            if positive_div and strong_conf:
                # Bullish signal with 3-day acceleration weighting
                alpha_signal.iloc[i] = data.iloc[i]['cumulative_accel_3d']
            else:
                alpha_signal.iloc[i] = 0
                
        # Normal Volatility Regime
        else:
            if (positive_div or negative_div) and strong_conf:
                # Directional signal with blended acceleration
                direction = 1 if positive_div else -1
                accel_blend = (
                    0.6 * data.iloc[i]['cumulative_accel_3d'] + 
                    0.4 * data.iloc[i]['cumulative_accel_10d']
                )
                alpha_signal.iloc[i] = direction * accel_blend
            else:
                alpha_signal.iloc[i] = 0
    
    # Fill NaN values with 0
    alpha_signal = alpha_signal.fillna(0)
    
    return alpha_signal
