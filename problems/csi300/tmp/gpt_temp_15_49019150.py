import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Multi-Scale Fractal Momentum with Volume-Pressure Alignment
    """
    data = df.copy()
    
    # Calculate True Range
    data['TR'] = np.maximum(
        data['high'] - data['low'],
        np.maximum(
            abs(data['high'] - data['close'].shift(1)),
            abs(data['low'] - data['close'].shift(1))
        )
    )
    
    # Volatility Regime Classification
    data['daily_range'] = data['high'] - data['low']
    data['range_5d_avg'] = data['daily_range'].rolling(window=5).mean()
    data['range_20d_median'] = data['daily_range'].rolling(window=20).median()
    
    # Volatility regime classification
    conditions = [
        data['range_5d_avg'] > (1.3 * data['range_20d_median']),
        data['range_5d_avg'] < (0.8 * data['range_20d_median'])
    ]
    choices = ['high', 'low']
    data['vol_regime'] = np.select(conditions, choices, default='normal')
    
    # Regime-specific parameters
    def get_regime_params(regime):
        if regime == 'high':
            return {'S': 3, 'M': 8, 'P': 3}
        elif regime == 'low':
            return {'S': 5, 'M': 15, 'P': 8}
        else:  # normal
            return {'S': 5, 'M': 10, 'P': 5}
    
    # Multi-Scale Fractal Efficiency Analysis
    efficiency_short = pd.Series(index=data.index, dtype=float)
    efficiency_medium = pd.Series(index=data.index, dtype=float)
    
    for i in range(len(data)):
        if i < 20:  # Need enough data for calculations
            continue
            
        regime = data['vol_regime'].iloc[i]
        params = get_regime_params(regime)
        S, M, P = params['S'], params['M'], params['P']
        
        # Short-term efficiency
        if i >= S:
            price_diff_short = abs(data['close'].iloc[i] - data['close'].iloc[i-S])
            tr_sum_short = data['TR'].iloc[i-S+1:i+1].sum()
            efficiency_short.iloc[i] = price_diff_short / tr_sum_short if tr_sum_short > 0 else 0
        
        # Medium-term efficiency
        if i >= M:
            price_diff_medium = abs(data['close'].iloc[i] - data['close'].iloc[i-M])
            tr_sum_medium = data['TR'].iloc[i-M+1:i+1].sum()
            efficiency_medium.iloc[i] = price_diff_medium / tr_sum_medium if tr_sum_medium > 0 else 0
    
    data['efficiency_short'] = efficiency_short
    data['efficiency_medium'] = efficiency_medium
    
    # Efficiency Gradient
    data['efficiency_diff'] = data['efficiency_medium'] - data['efficiency_short']
    
    # Directional scaling
    directional_scaling = pd.Series(index=data.index, dtype=float)
    for i in range(len(data)):
        if i < 5:  # Minimum window for regime parameters
            continue
        regime = data['vol_regime'].iloc[i]
        S = get_regime_params(regime)['S']
        if i >= S:
            price_change = data['close'].iloc[i] - data['close'].iloc[i-S]
            directional_scaling.iloc[i] = np.sign(price_change) if price_change != 0 else 0
    
    data['efficiency_gradient'] = data['efficiency_diff'] * directional_scaling
    
    # Intraday Fractal Confirmation
    data['intraday_efficiency'] = (data['close'] - data['open']) / (data['high'] - data['low']).replace(0, np.nan)
    data['volume_concentration'] = data['volume'] / (data['high'] - data['low']).replace(0, np.nan)
    data['intraday_confirmation'] = data['intraday_efficiency'] * data['volume_concentration']
    
    # Volume-Pressure Asymmetry Analysis
    bullish_pressure = pd.Series(index=data.index, dtype=float)
    bearish_pressure = pd.Series(index=data.index, dtype=float)
    
    for i in range(len(data)):
        if i < 20:
            continue
            
        regime = data['vol_regime'].iloc[i]
        P = get_regime_params(regime)['P']
        
        if i >= P:
            # Bullish pressure
            bullish_values = [(data['close'].iloc[j] - data['low'].iloc[j]) * data['volume'].iloc[j] 
                            for j in range(i-P+1, i+1)]
            bullish_pressure.iloc[i] = np.mean(bullish_values)
            
            # Bearish pressure
            bearish_values = [(data['high'].iloc[j] - data['close'].iloc[j]) * data['volume'].iloc[j] 
                            for j in range(i-P+1, i+1)]
            bearish_pressure.iloc[i] = np.mean(bearish_values)
    
    data['bullish_pressure'] = bullish_pressure
    data['bearish_pressure'] = bearish_pressure
    data['net_pressure_asymmetry'] = np.log1p(data['bullish_pressure']) - np.log1p(data['bearish_pressure'])
    
    # Pressure momentum
    pressure_momentum = pd.Series(index=data.index, dtype=float)
    for i in range(len(data)):
        if i < 20:
            continue
        regime = data['vol_regime'].iloc[i]
        S = get_regime_params(regime)['S']
        if i >= S:
            current_asymmetry = data['net_pressure_asymmetry'].iloc[i]
            lagged_asymmetry = data['net_pressure_asymmetry'].iloc[i-S]
            pressure_momentum.iloc[i] = current_asymmetry - lagged_asymmetry
    
    data['pressure_momentum'] = pressure_momentum
    
    # Turnover-Price Alignment
    data['daily_turnover'] = data['volume'] * data['close']
    
    turnover_momentum = pd.Series(index=data.index, dtype=float)
    for i in range(len(data)):
        if i < 20:
            continue
        regime = data['vol_regime'].iloc[i]
        S, M = get_regime_params(regime)['S'], get_regime_params(regime)['M']
        
        if i >= M:
            short_term_avg = data['daily_turnover'].iloc[i-S+1:i+1].mean()
            medium_term_avg = data['daily_turnover'].iloc[i-M+1:i+1].mean()
            turnover_momentum.iloc[i] = (short_term_avg / medium_term_avg) - 1 if medium_term_avg > 0 else 0
    
    data['turnover_momentum'] = turnover_momentum
    
    # Range Expansion Efficiency
    data['current_range'] = data['high'] - data['low']
    data['historical_range'] = data['current_range'].rolling(window=10).mean()
    data['range_expansion_ratio'] = data['current_range'] / data['historical_range'].replace(0, np.nan)
    
    # Range efficiency momentum
    range_efficiency_momentum = pd.Series(index=data.index, dtype=float)
    for i in range(1, len(data)):
        current_efficiency = data['range_expansion_ratio'].iloc[i]
        lagged_efficiency = data['range_expansion_ratio'].iloc[i-1]
        range_efficiency_momentum.iloc[i] = current_efficiency - lagged_efficiency
    
    data['range_efficiency_momentum'] = range_efficiency_momentum
    
    # Adaptive Signal Synthesis
    final_factor = pd.Series(index=data.index, dtype=float)
    
    for i in range(len(data)):
        if i < 20:
            continue
            
        regime = data['vol_regime'].iloc[i]
        
        # Get regime-specific weights
        if regime == 'high':
            w1, w2, w3, w4 = 0.4, 0.3, 0.2, 0.1  # Emphasize efficiency and pressure
        elif regime == 'low':
            w1, w2, w3, w4 = 0.2, 0.2, 0.4, 0.2  # Emphasize turnover and range
        else:  # normal
            w1, w2, w3, w4 = 0.25, 0.25, 0.25, 0.25  # Balanced
        
        # Combine components
        component1 = data['efficiency_gradient'].iloc[i] * data['pressure_momentum'].iloc[i]
        component2 = component1 * data['turnover_momentum'].iloc[i]
        component3 = component2 * data['range_efficiency_momentum'].iloc[i]
        component4 = component3 * data['intraday_confirmation'].iloc[i]
        
        # Weighted combination
        weighted_signal = (w1 * component1 + w2 * component2 + 
                          w3 * component3 + w4 * component4)
        
        final_factor.iloc[i] = weighted_signal
    
    # Apply regime-specific smoothing
    smoothed_factor = pd.Series(index=data.index, dtype=float)
    for i in range(len(data)):
        if i < 20:
            continue
            
        regime = data['vol_regime'].iloc[i]
        
        if regime == 'high':
            window = 3
        elif regime == 'low':
            window = 8
        else:  # normal
            window = 5
        
        if i >= window:
            smoothed_factor.iloc[i] = final_factor.iloc[i-window+1:i+1].mean()
        else:
            smoothed_factor.iloc[i] = final_factor.iloc[i]
    
    return smoothed_factor
