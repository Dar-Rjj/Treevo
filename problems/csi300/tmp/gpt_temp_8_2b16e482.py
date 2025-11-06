import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def hurst_exponent(series, max_lag=20):
    """Calculate Hurst exponent for persistence analysis"""
    if len(series) < max_lag * 2:
        return 0.5
    
    lags = range(2, min(max_lag, len(series)//2))
    tau = []
    for lag in lags:
        if len(series) >= lag * 2:
            diff = np.subtract(series[lag:], series[:-lag])
            tau.append(np.std(diff))
    
    if len(tau) < 2:
        return 0.5
    
    lags_arr = np.log(np.array(lags[:len(tau)]))
    tau_arr = np.log(np.array(tau))
    
    if len(lags_arr) > 1 and len(tau_arr) > 1:
        hurst = np.polyfit(lags_arr, tau_arr, 1)[0]
        return hurst
    return 0.5

def heuristics_v2(df):
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Fractal Liquidity Decomposition
    # Directional Fractal Flow
    data['Upside_Fractal_Flow'] = (data['high'] - data['open']) / data['volume'].replace(0, 1)
    data['Downside_Fractal_Flow'] = (data['open'] - data['low']) / data['volume'].replace(0, 1)
    data['Net_Fractal_Flow_Bias'] = (data['Upside_Fractal_Flow'] - data['Downside_Fractal_Flow']) / \
                                   (data['Upside_Fractal_Flow'] + data['Downside_Fractal_Flow'] + 1e-8)
    
    # Multi-scale Fractal Liquidity
    data['Short_term_Fractal_Liquidity'] = data['volume'] / (data['high'] - data['low'] + 1e-8)
    data['Medium_term_Fractal_Liquidity'] = data['volume'] / (data['high'].shift(3) - data['low'].shift(3) + 1e-8)
    
    # Fractal Absorption Dynamics
    data['Fractal_Absorption_Rate'] = abs(data['close'] - data['open']) / (data['high'] - data['low'] + 1e-8)
    data['Fractal_Absorption_Momentum'] = data['Fractal_Absorption_Rate'] / (data['Fractal_Absorption_Rate'].shift(3) + 1e-8)
    data['Fractal_Absorption_Consistency'] = np.sign(data['close'] - data['open']) * np.sign(data['close'].shift(1) - data['open'].shift(1))
    
    # Fractal Flow Momentum Efficiency
    # Flow Momentum Quality Assessment
    data['Clean_Flow_Momentum'] = data['Net_Fractal_Flow_Bias'] / (data['Net_Fractal_Flow_Bias'].shift(1) + 1e-8) - 1
    data['Noisy_Flow_Momentum'] = (data['Upside_Fractal_Flow'] + data['Downside_Fractal_Flow']) / data['volume'].replace(0, 1)
    data['Fractal_Flow_Efficiency'] = data['Clean_Flow_Momentum'] / (data['Noisy_Flow_Momentum'] + 1e-8)
    
    # Multi-scale Flow Consistency
    data['Short_term_Flow_Consistency'] = np.sign(data['Net_Fractal_Flow_Bias']) * np.sign(data['Net_Fractal_Flow_Bias'].shift(1))
    data['Medium_term_Flow_Alignment'] = np.sign(data['Net_Fractal_Flow_Bias']) * np.sign(data['Net_Fractal_Flow_Bias'].shift(3))
    
    # Volume-Fractal Absorption Interaction
    data['Absolute_Fractal_Absorption'] = data['volume'] * abs(data['close'] - data['open']) / (data['high'] - data['low'] + 1e-8)
    data['Relative_Fractal_Absorption'] = (data['volume'] / (data['volume'].shift(1) + 1e-8)) * data['Fractal_Absorption_Rate']
    data['High_Fractal_Absorption'] = data['amount'] / (data['high'] - data['low'] + 1e-8)
    data['Low_Fractal_Absorption'] = data['volume'] / (data['amount'] + 1e-8)
    
    # Asymmetric Fractal Breakout Absorption
    data['Upside_Fractal_Breakout'] = (data['high'] / (data['high'].shift(1) + 1e-8) - 1) * data['Upside_Fractal_Flow']
    data['Downside_Fractal_Breakout'] = (data['low'] / (data['low'].shift(1) + 1e-8) - 1) * data['Downside_Fractal_Flow']
    data['Fractal_Breakout_Asymmetry'] = data['Upside_Fractal_Breakout'] - data['Downside_Fractal_Breakout']
    
    # Fractal Liquidity Regime Classification
    data['Strong_Fractal_Absorption'] = (data['Fractal_Absorption_Rate'] > 0.7).astype(int)
    data['Weak_Fractal_Absorption'] = (data['Fractal_Absorption_Rate'] < 0.3).astype(int)
    data['Neutral_Fractal_Absorption'] = ((data['Fractal_Absorption_Rate'] >= 0.3) & (data['Fractal_Absorption_Rate'] <= 0.7)).astype(int)
    
    # Fractal Flow Momentum Regime
    data['Accelerating_Fractal_Flow'] = ((data['Fractal_Flow_Efficiency'] > 0) & 
                                        (data['Fractal_Flow_Efficiency'] > data['Fractal_Flow_Efficiency'].shift(1))).astype(int)
    data['Decelerating_Fractal_Flow'] = ((data['Fractal_Flow_Efficiency'] < 0) & 
                                        (data['Fractal_Flow_Efficiency'] < data['Fractal_Flow_Efficiency'].shift(1))).astype(int)
    data['Stable_Fractal_Flow'] = (abs(data['Fractal_Flow_Efficiency']) < 0.1).astype(int)
    
    # Fractal Liquidity Concentration Regime
    rolling_liquidity = data['Short_term_Fractal_Liquidity'].rolling(window=3, min_periods=1).mean()
    data['High_Fractal_Liquidity'] = (data['Short_term_Fractal_Liquidity'] > rolling_liquidity).astype(int)
    data['Low_Fractal_Liquidity'] = (data['Short_term_Fractal_Liquidity'] < rolling_liquidity).astype(int)
    data['Normal_Fractal_Liquidity'] = ((data['Short_term_Fractal_Liquidity'] <= rolling_liquidity) & 
                                       (data['Short_term_Fractal_Liquidity'] >= rolling_liquidity)).astype(int)
    
    # Calculate Hurst-based persistence measures
    window_size = 20
    data['Fractal_Liquidity_Persistence'] = 0.5
    data['Flow_Persistence_Score'] = 0.5
    
    for i in range(window_size, len(data)):
        if i >= window_size:
            liquidity_series = data['Short_term_Fractal_Liquidity'].iloc[i-window_size:i]
            flow_series = data['Net_Fractal_Flow_Bias'].iloc[i-window_size:i]
            
            data.loc[data.index[i], 'Fractal_Liquidity_Persistence'] = hurst_exponent(liquidity_series.values)
            data.loc[data.index[i], 'Flow_Persistence_Score'] = hurst_exponent(flow_series.values)
    
    # Composite Fractal Liquidity Alpha - Regime-Adaptive Factor Construction
    alpha_signal = pd.Series(index=data.index, dtype=float)
    
    for i, row in data.iterrows():
        # Regime classification
        strong_absorption = row['Strong_Fractal_Absorption']
        weak_absorption = row['Weak_Fractal_Absorption']
        neutral_absorption = row['Neutral_Fractal_Absorption']
        
        accelerating_flow = row['Accelerating_Fractal_Flow']
        stable_flow = row['Stable_Fractal_Flow']
        
        high_liquidity = row['High_Fractal_Liquidity']
        low_liquidity = row['Low_Fractal_Liquidity']
        normal_liquidity = row['Normal_Fractal_Liquidity']
        
        # Regime-adaptive factor selection
        if strong_absorption and accelerating_flow:
            # Strong Absorption, Accelerating Flow Regime
            base_factor = row['Net_Fractal_Flow_Bias'] * row['Fractal_Flow_Efficiency']
            factor = base_factor * row['Fractal_Absorption_Rate'] * row['Short_term_Fractal_Liquidity']
            
        elif weak_absorption and high_liquidity:
            # Weak Absorption, High Liquidity Regime
            base_factor = row['Fractal_Breakout_Asymmetry'] * row['High_Fractal_Absorption']
            factor = base_factor * row['Fractal_Flow_Efficiency'] * row['Low_Fractal_Absorption']
            
        elif neutral_absorption and stable_flow:
            # Neutral Absorption, Stable Flow Regime
            base_factor = row['Fractal_Absorption_Consistency'] * row['Fractal_Liquidity_Persistence']
            factor = base_factor * row['Net_Fractal_Flow_Bias'] * row['Absolute_Fractal_Absorption']
            
        else:
            # Transition Flow Detection
            base_factor = row['Fractal_Flow_Efficiency'] * row['Fractal_Absorption_Momentum']
            factor = base_factor * row['Fractal_Breakout_Asymmetry'] * row['Relative_Fractal_Absorption']
        
        alpha_signal.loc[i] = factor
    
    # Core Fractal Liquidity Components integration
    data['Fractal_Adjusted_Flow_Momentum'] = data['Clean_Flow_Momentum'] * (1 / (data['Noisy_Flow_Momentum'] + 1e-8))
    data['Directionally_Biased_Fractal_Absorption'] = data['Net_Fractal_Flow_Bias'] * data['Fractal_Flow_Efficiency']
    
    # Final signal refinement with smoothing
    final_alpha = alpha_signal.rolling(window=3, min_periods=1).mean()
    
    # Fractal validation - apply consistency checks
    for i in range(2, len(final_alpha)):
        if i >= 2:
            current_val = final_alpha.iloc[i]
            prev_val = final_alpha.iloc[i-1]
            prev_prev_val = final_alpha.iloc[i-2]
            
            # Check for extreme outliers and smooth them
            if abs(current_val - prev_val) > 2 * abs(prev_val - prev_prev_val):
                final_alpha.iloc[i] = (prev_val + prev_prev_val) / 2
    
    return final_alpha
