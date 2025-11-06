import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # True Range Calculation
    data['prev_close'] = data['close'].shift(1)
    data['tr1'] = data['high'] - data['low']
    data['tr2'] = abs(data['high'] - data['prev_close'])
    data['tr3'] = abs(data['low'] - data['prev_close'])
    data['true_range'] = data[['tr1', 'tr2', 'tr3']].max(axis=1)
    
    # Rolling Volatility (10-day average of True Range)
    data['rolling_volatility'] = data['true_range'].rolling(window=10).mean()
    
    # Median True Range (60-day)
    data['median_true_range'] = data['true_range'].rolling(window=60).median()
    
    # Volatility Regime Classification
    conditions = [
        data['rolling_volatility'] > (2 * data['median_true_range']),
        (data['rolling_volatility'] >= data['median_true_range']) & 
        (data['rolling_volatility'] <= (2 * data['median_true_range'])),
        data['rolling_volatility'] < data['median_true_range']
    ]
    choices = ['high', 'normal', 'low']
    data['volatility_regime'] = np.select(conditions, choices, default='normal')
    
    # Volume-Weighted Components (5-day)
    data['vw_close'] = (data['close'] * data['volume']).rolling(window=5).sum() / data['volume'].rolling(window=5).sum()
    data['vw_range'] = ((data['high'] - data['low']) * data['volume']).rolling(window=5).sum() / data['volume'].rolling(window=5).sum()
    
    # Liquidity Flow Dynamics
    data['liquidity_range_expansion'] = data['vw_range'] / data['vw_range'].shift(5) - 1
    data['volume_momentum'] = data['volume'] / data['volume'].shift(5) - 1
    
    # Intraday Efficiency
    data['intraday_efficiency'] = abs(data['close'] - data['open']) / (data['high'] - data['low'])
    data['intraday_efficiency'] = data['intraday_efficiency'].replace([np.inf, -np.inf], np.nan)
    
    # Liquidity Efficiency Classification
    efficiency_conditions = [
        (data['intraday_efficiency'] > 0.6) & (data['liquidity_range_expansion'] > 0),
        (data['intraday_efficiency'] >= 0.3) & (data['intraday_efficiency'] <= 0.6) & (data['liquidity_range_expansion'] >= 0),
        (data['intraday_efficiency'] < 0.3) | (data['liquidity_range_expansion'] < 0)
    ]
    efficiency_choices = ['high', 'medium', 'low']
    data['liquidity_efficiency'] = np.select(efficiency_conditions, efficiency_choices, default='medium')
    
    # Price Momentum Components
    data['price_momentum'] = data['close'] / data['close'].shift(5) - 1
    data['acceleration_momentum'] = (data['close'] / data['close'].shift(5) - 1) - (data['close'].shift(5) / data['close'].shift(10) - 1)
    
    # Volume Momentum Components
    data['volume_momentum_5'] = data['volume'] / data['volume'].shift(5) - 1
    data['acceleration_volume_momentum'] = (data['volume'] / data['volume'].shift(5) - 1) - (data['volume'].shift(5) / data['volume'].shift(10) - 1)
    
    # Divergence Signal Generation
    data['price_volume_momentum_product'] = data['price_momentum'] * data['volume_momentum_5']
    data['acceleration_divergence'] = data['acceleration_momentum'] * data['acceleration_volume_momentum']
    data['combined_divergence'] = 0.7 * data['price_volume_momentum_product'] + 0.3 * data['acceleration_divergence']
    
    # Volume median for regime processing
    data['volume_median_20'] = data['volume'].rolling(window=20).median()
    
    # Amount density components
    data['amount_sum_5'] = data['amount'].rolling(window=5).sum()
    data['volume_sum_5'] = data['volume'].rolling(window=5).sum()
    data['amount_density_prev'] = (data['amount_sum_5'] / data['volume_sum_5']).shift(1)
    data['amount_density_current'] = data['amount'] / data['volume']
    
    # Regime-Adaptive Signal Enhancement
    enhanced_signals = []
    
    for i in range(len(data)):
        if i < 20:  # Ensure enough data for calculations
            enhanced_signals.append(np.nan)
            continue
            
        row = data.iloc[i]
        combined_div = row['combined_divergence']
        
        if row['volatility_regime'] == 'high':
            volume_confirmation = row['volume'] > (1.5 * row['volume_median_20'])
            amount_density_filter = row['amount_density_current'] > row['amount_density_prev']
            
            if volume_confirmation and amount_density_filter:
                enhanced_signal = combined_div * 1.5
            else:
                enhanced_signal = combined_div * 0.8
                
        elif row['volatility_regime'] == 'normal':
            volume_validation = row['volume'] > row['volume_median_20']
            consistency_check = abs(row['volume_momentum_5']) < 0.5
            
            if volume_validation and consistency_check:
                enhanced_signal = combined_div * 1.0
            else:
                enhanced_signal = combined_div * 0.9
                
        else:  # low volatility
            volume_sensitivity = row['volume'] > (0.8 * row['volume_median_20'])
            fine_detection = abs(row['price_momentum']) < 0.1
            
            if volume_sensitivity and fine_detection:
                enhanced_signal = combined_div * 0.7
            else:
                enhanced_signal = combined_div * 0.5
                
        enhanced_signals.append(enhanced_signal)
    
    data['enhanced_signal'] = enhanced_signals
    
    # Efficiency-Weighted Signal
    efficiency_factors = []
    for i in range(len(data)):
        if i < 20:
            efficiency_factors.append(np.nan)
            continue
            
        row = data.iloc[i]
        enhanced_signal = row['enhanced_signal']
        
        if row['liquidity_efficiency'] == 'high':
            weighted_signal = enhanced_signal * 1.3
        elif row['liquidity_efficiency'] == 'medium':
            weighted_signal = enhanced_signal * 1.0
        else:  # low efficiency
            weighted_signal = enhanced_signal * 0.5
            
        efficiency_factors.append(weighted_signal)
    
    data['efficiency_weighted_signal'] = efficiency_factors
    
    # Liquidity Flow Adjustment
    final_factors = []
    for i in range(len(data)):
        if i < 20:
            final_factors.append(np.nan)
            continue
            
        row = data.iloc[i]
        signal = row['efficiency_weighted_signal']
        vol_momentum = row['volume_momentum_5']
        
        if vol_momentum > 0:
            adjusted_signal = signal * (1 + vol_momentum)
        else:
            adjusted_signal = signal * (1 - abs(vol_momentum))
            
        final_factors.append(adjusted_signal)
    
    # Create final factor series
    factor_series = pd.Series(final_factors, index=data.index, name='volatility_regime_adaptive_pv_divergence')
    
    return factor_series
