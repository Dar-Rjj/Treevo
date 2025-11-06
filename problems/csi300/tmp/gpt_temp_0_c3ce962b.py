import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate True Range
    data['prev_close'] = data['close'].shift(1)
    data['tr1'] = data['high'] - data['low']
    data['tr2'] = abs(data['high'] - data['prev_close'])
    data['tr3'] = abs(data['low'] - data['prev_close'])
    data['true_range'] = data[['tr1', 'tr2', 'tr3']].max(axis=1)
    
    # Calculate Volatility-Adjusted Return
    data['price_change'] = data['close'] - data['prev_close']
    data['vol_adj_return'] = data['price_change'] / data['true_range']
    data['vol_adj_return'] = data['vol_adj_return'].replace([np.inf, -np.inf], np.nan)
    
    # Calculate Volume-Weighted Price Trend
    data['volume_flow'] = data['price_change'] * data['volume']
    
    # Calculate Volume Flow Persistence Score
    data['flow_sign'] = np.sign(data['volume_flow'])
    data['sign_change'] = data['flow_sign'] != data['flow_sign'].shift(1)
    data['persistence'] = 1
    
    # Calculate consecutive persistence streak
    for i in range(1, len(data)):
        if data['sign_change'].iloc[i]:
            data.loc[data.index[i], 'persistence'] = 1
        else:
            data.loc[data.index[i], 'persistence'] = data['persistence'].iloc[i-1] + 1
    
    # Calculate Convergence Signal
    data['convergence_score'] = data['vol_adj_return'] * data['persistence']
    
    # Apply Momentum Confirmation
    data['prev_convergence_sign'] = np.sign(data['convergence_score'].shift(1))
    data['current_convergence_sign'] = np.sign(data['convergence_score'])
    data['momentum_confirmation'] = np.where(
        data['prev_convergence_sign'] == data['current_convergence_sign'], 
        1, -1
    )
    data['confirmed_convergence'] = data['convergence_score'] * data['momentum_confirmation']
    
    # Apply Convergence Filter (keep only positive convergence)
    data['filtered_convergence'] = np.where(
        data['confirmed_convergence'] > 0, 
        data['confirmed_convergence'], 
        0
    )
    
    # Calculate Intensity Adjustment using intraday price range
    data['relative_range'] = (data['high'] - data['low']) / data['close']
    
    # Final Alpha Factor
    alpha_factor = data['filtered_convergence'] * data['relative_range']
    
    # Clean up and return
    alpha_factor = alpha_factor.replace([np.inf, -np.inf], np.nan)
    return alpha_factor
