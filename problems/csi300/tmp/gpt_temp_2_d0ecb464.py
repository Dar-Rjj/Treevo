import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Price-Volume Asymmetry with Regime Transition Detection alpha factor
    """
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate Directional Price Impact Asymmetry
    # Upside Price Impact
    up_days = data['close'] > data['open']
    upside_volume_efficiency = []
    for i in range(len(data)):
        if i >= 5:
            window_data = data.iloc[i-5:i+1]
            up_mask = (window_data['close'] > window_data['open'])
            if up_mask.any():
                upside_eff = (window_data.loc[up_mask, 'volume'] * 
                            (window_data.loc[up_mask, 'close'] - window_data.loc[up_mask, 'open']) / 
                            window_data.loc[up_mask, 'open']).sum()
            else:
                upside_eff = 0
            upside_volume_efficiency.append(upside_eff)
        else:
            upside_volume_efficiency.append(np.nan)
    
    data['upside_volume_efficiency'] = upside_volume_efficiency
    data['upside_impact_change'] = data['upside_volume_efficiency'] / data['upside_volume_efficiency'].shift(6) - 1
    
    # Downside Price Impact
    downside_volume_efficiency = []
    for i in range(len(data)):
        if i >= 5:
            window_data = data.iloc[i-5:i+1]
            down_mask = (window_data['close'] < window_data['open'])
            if down_mask.any():
                downside_eff = (window_data.loc[down_mask, 'volume'] * 
                              (window_data.loc[down_mask, 'open'] - window_data.loc[down_mask, 'close']) / 
                              window_data.loc[down_mask, 'open']).sum()
            else:
                downside_eff = 0
            downside_volume_efficiency.append(downside_eff)
        else:
            downside_volume_efficiency.append(np.nan)
    
    data['downside_volume_efficiency'] = downside_volume_efficiency
    data['downside_impact_change'] = data['downside_volume_efficiency'] / data['downside_volume_efficiency'].shift(6) - 1
    
    # Impact Asymmetry Ratio
    data['asymmetry_strength'] = data['upside_impact_change'] - data['downside_impact_change']
    
    # Calculate Intraday Pressure Dynamics
    # Opening Pressure Structure (using high/low as proxy for intraday extremes)
    data['opening_pressure'] = (data['high'] - data['open']) / (data['open'] - data['low']).replace(0, np.nan)
    # Assuming first hour volume is 25% of daily volume as proxy
    data['opening_volume_bias'] = 0.25
    data['opening_pressure_strength'] = data['opening_pressure'] * data['opening_volume_bias']
    
    # Closing Pressure Structure
    data['closing_pressure'] = (data['close'] - data['low']) / (data['high'] - data['close']).replace(0, np.nan)
    # Assuming last hour volume is 25% of daily volume as proxy
    data['closing_volume_bias'] = 0.25
    data['closing_pressure_strength'] = data['closing_pressure'] * data['closing_volume_bias']
    
    # Pressure Divergence Pattern
    data['pressure_shift'] = data['opening_pressure_strength'] - data['closing_pressure_strength']
    
    # Combine Asymmetry and Pressure Signals
    # Core Asymmetry Factor
    data['core_asymmetry_factor'] = data['asymmetry_strength'] * data['pressure_shift']
    
    # Volume-Price Coherence Analysis
    data['pressure_coherence'] = np.sign(data['opening_pressure_strength']) * np.sign(data['closing_pressure_strength'])
    data['asymmetry_pressure_alignment'] = np.sign(data['asymmetry_strength']) * np.sign(data['pressure_shift'])
    
    # Enhanced factor with coherence
    data['enhanced_factor'] = data['core_asymmetry_factor'] * (1 + 0.5 * data['pressure_coherence'])
    
    # Incorporate Range-Adaptive Signals
    # Daily Range Structure
    data['range_expansion'] = (data['high'] - data['low']) / (data['high'].shift(1) - data['low'].shift(1)).replace(0, np.nan)
    data['pressure_range_ratio'] = data['opening_pressure_strength'] / data['closing_pressure_strength'].replace(0, np.nan)
    
    # Range-adaptive weighting
    expanding_range = data['range_expansion'] > 1.0
    contracting_range = data['range_expansion'] < 0.8
    transitioning_range = ~expanding_range & ~contracting_range
    
    # Final alpha factor with range adaptation
    alpha_factor = pd.Series(index=data.index, dtype=float)
    
    for i in range(len(data)):
        if expanding_range.iloc[i]:
            # Favor pressure divergence signals in expanding ranges
            alpha_factor.iloc[i] = data['pressure_shift'].iloc[i] * 0.7 + data['enhanced_factor'].iloc[i] * 0.3
        elif contracting_range.iloc[i]:
            # Favor impact asymmetry signals in contracting ranges
            alpha_factor.iloc[i] = data['asymmetry_strength'].iloc[i] * 0.7 + data['enhanced_factor'].iloc[i] * 0.3
        else:
            # Transitioning range - weight by range expansion
            range_weight = min(1.0, max(0.0, data['range_expansion'].iloc[i]))
            alpha_factor.iloc[i] = (data['pressure_shift'].iloc[i] * range_weight + 
                                  data['asymmetry_strength'].iloc[i] * (1 - range_weight))
    
    # Apply smoothing and normalization
    alpha_factor = alpha_factor.rolling(window=3, min_periods=1).mean()
    alpha_factor = (alpha_factor - alpha_factor.rolling(window=20, min_periods=1).mean()) / alpha_factor.rolling(window=20, min_periods=1).std()
    
    return alpha_factor
