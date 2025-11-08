import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Volatility Regime Detection
    # 5-day High-Low Volatility
    data['high_low_range'] = (data['high'] - data['low']) / data['close']
    data['current_vol'] = data['high_low_range'].rolling(window=5, min_periods=3).mean()
    
    # 10-day Historical Volatility
    data['returns'] = data['close'].pct_change()
    data['hist_vol'] = data['returns'].rolling(window=10, min_periods=5).std()
    
    # Volatility regime classification
    data['vol_regime'] = 'normal'
    high_vol_condition = data['current_vol'] > (1.2 * data['hist_vol'])
    low_vol_condition = data['current_vol'] < (0.8 * data['hist_vol'])
    data.loc[high_vol_condition, 'vol_regime'] = 'high'
    data.loc[low_vol_condition, 'vol_regime'] = 'low'
    
    # Core Reversal Signal
    data['overnight_gap'] = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
    data['intraday_return'] = (data['close'] - data['open']) / data['open']
    data['reversal_efficiency'] = np.sign(data['overnight_gap'] * data['intraday_return'])
    
    # Volume Confirmation
    data['volume_ma_20'] = data['volume'].rolling(window=20, min_periods=10).mean()
    data['volume_ratio'] = data['volume'] / data['volume_ma_20']
    data['volume_price_alignment'] = np.sign(data['volume_ratio']) * np.sign(data['intraday_return'])
    
    # Order Flow Analysis
    data['price_flow'] = (data['close'] - data['open']) / (data['high'] - data['low']).replace(0, np.nan)
    data['flow_momentum'] = data['price_flow'].pct_change(periods=3)
    
    # Raw signals for regime-specific processing
    reversal_signal = data['reversal_efficiency'] * data['volume_price_alignment']
    flow_signal = data['flow_momentum']
    
    # Regime-Specific Integration
    factor_values = pd.Series(index=data.index, dtype=float)
    
    for i in range(len(data)):
        if i < 10:  # Ensure enough data for calculations
            factor_values.iloc[i] = 0
            continue
            
        current_regime = data['vol_regime'].iloc[i]
        
        if current_regime == 'high':
            # High Volatility: Mean Reversion focus
            reversal_weight = 0.7
            flow_weight = 0.3
            # Raw signals for high volatility
            regime_signal = (reversal_weight * reversal_signal.iloc[i] + 
                           flow_weight * flow_signal.iloc[i])
            
        elif current_regime == 'low':
            # Low Volatility: Momentum Continuation focus
            reversal_weight = 0.4
            flow_weight = 0.6
            # 3-day smoothing for low volatility
            start_idx = max(0, i-2)
            smoothed_reversal = reversal_signal.iloc[start_idx:i+1].mean()
            smoothed_flow = flow_signal.iloc[start_idx:i+1].mean()
            regime_signal = (reversal_weight * smoothed_reversal + 
                           flow_weight * smoothed_flow)
            
        else:  # Normal regime
            # Balanced approach
            reversal_weight = 0.55
            flow_weight = 0.45
            regime_signal = (reversal_weight * reversal_signal.iloc[i] + 
                           flow_weight * flow_signal.iloc[i])
        
        # Volume confidence scaling
        volume_confidence = np.tanh(data['volume_ratio'].iloc[i])
        final_factor = regime_signal * volume_confidence
        
        factor_values.iloc[i] = final_factor
    
    # Clean up any remaining NaN values
    factor_values = factor_values.fillna(0)
    
    return factor_values
