import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Chaotic-Entropic Microstructure Resonance Dynamics alpha factor
    Combines multiple chaotic entropy measures and microstructure dynamics
    """
    # Create a copy to avoid modifying original dataframe
    data = df.copy()
    
    # Basic chaotic entropy components
    data['microstructure_entropy'] = (data['high'] - data['low']) * data['volume'] / data['amount'] * (data['close'] - data['open'])
    
    # Chaotic flow divergence with safe handling
    data['close_prev'] = data['close'].shift(1)
    data['volume_prev'] = data['volume'].shift(1)
    data['chaotic_flow_divergence'] = np.abs(data['close'] - data['close_prev']) * (data['volume'] - data['volume_prev']) / data['amount']
    
    # Entropic resonance with safe handling
    data['amount_prev'] = data['amount'].shift(1)
    data['entropic_resonance'] = (data['high'] + data['low'] - 2 * data['close']) * data['volume'] / (data['amount'] - data['amount_prev']).replace(0, np.nan)
    
    # Entropic density
    data['entropic_density'] = data['volume'] * (data['high'] - data['low']) / (data['close'] - data['open']).replace(0, np.nan)
    
    # Chaotic persistence - 5-day correlation
    data['close_open_diff'] = data['close'] - data['open']
    chaotic_persistence = []
    for i in range(len(data)):
        if i >= 4:
            window_close_open = data['close_open_diff'].iloc[i-4:i+1]
            window_volume = data['volume'].iloc[i-4:i+1]
            if len(window_close_open) == 5 and len(window_volume) == 5:
                corr = np.corrcoef(window_close_open, window_volume)[0,1]
                if not np.isnan(corr):
                    chaotic_persistence.append(corr * (data['high'].iloc[i] - data['low'].iloc[i]))
                else:
                    chaotic_persistence.append(0)
            else:
                chaotic_persistence.append(0)
        else:
            chaotic_persistence.append(0)
    data['chaotic_persistence'] = chaotic_persistence
    
    # Entropic probability
    data['entropic_probability'] = (data['close'] - data['open']) / (data['high'] - data['low']).replace(0, np.nan) * data['volume']
    
    # Volatility-chaotic flow
    data['volatility_chaotic_flow'] = (data['high'] - data['low']) * data['volume'] / data['amount']
    
    # Chaotic flow wave
    data['chaotic_flow_wave'] = (data['close'] - data['low']) / (data['high'] - data['low']).replace(0, np.nan) * data['volume']
    
    # Entropic interference with safe handling
    data['amount_prev'] = data['amount'].shift(1)
    data['volume_prev'] = data['volume'].shift(1)
    data['close_prev'] = data['close'].shift(1)
    data['entropic_interference'] = (data['amount'] - data['amount_prev']) / (data['volume'] - data['volume_prev']).replace(0, np.nan) * (data['close']/data['close_prev'] - 1)
    
    # Chaotic uncertainty
    data['chaotic_uncertainty'] = (data['high'] - data['low']) * data['volume']
    
    # Chaotic measurement with safe handling
    data['close_prev'] = data['close'].shift(1)
    data['amount_prev2'] = data['amount'].shift(2)
    data['chaotic_measurement'] = np.abs(data['close'] - data['close_prev']) * data['volume'] / data['amount'] * (data['amount']/data['amount_prev2'] - 1)
    
    # Entropic compression
    data['entropic_compression'] = data['volume'] / (data['high'] - data['low']).replace(0, np.nan)
    
    # Chaotic expansion
    data['amount_prev'] = data['amount'].shift(1)
    data['chaotic_expansion'] = (data['close'] - data['open']) * data['volume'] / data['amount'] * (data['amount']/data['amount_prev'] - 1)
    
    # Entropic critical point with safe handling
    data['volume_prev2'] = data['volume'].shift(2)
    data['volume_prev4'] = data['volume'].shift(4)
    data['entropic_critical_point'] = (data['volume'] - data['volume_prev2']) / (data['volume_prev2'] - data['volume_prev4']).replace(0, np.nan) * (data['close'] - data['open'])
    
    # Construct final alpha factor using chaotic-entropic integration
    # Entangled Chaotic Momentum Factor
    entangled_momentum = (data['microstructure_entropy'] * 
                         data['entropic_compression'] * 
                         data['entropic_probability'])
    
    # Chaotic Information Flow Alpha
    chaotic_info_flow = (data['entropic_compression'] * 
                        data['entropic_interference'] * 
                        data['entropic_critical_point'])
    
    # Microstructure Chaotic Factor
    microstructure_chaotic = (data['chaotic_uncertainty'] * 
                             data['chaotic_flow_divergence'] * 
                             data['chaotic_measurement'])
    
    # Adaptive weighting based on chaotic state
    chaotic_state = data['chaotic_uncertainty'].rolling(window=5, min_periods=1).mean()
    high_chaotic_threshold = chaotic_state.quantile(0.7)
    low_chaotic_threshold = chaotic_state.quantile(0.3)
    
    # Final alpha with adaptive superposition
    final_alpha = np.where(
        chaotic_state > high_chaotic_threshold,
        entangled_momentum * 0.6 + chaotic_info_flow * 0.4,  # High chaotic state
        np.where(
            chaotic_state < low_chaotic_threshold,
            microstructure_chaotic * 0.8 + entangled_momentum * 0.2,  # Low chaotic state
            entangled_momentum * 0.4 + chaotic_info_flow * 0.3 + microstructure_chaotic * 0.3  # Superposition state
        )
    )
    
    # Clean up and return
    result = pd.Series(final_alpha, index=data.index)
    result = result.replace([np.inf, -np.inf], np.nan).fillna(method='ffill')
    
    return result
