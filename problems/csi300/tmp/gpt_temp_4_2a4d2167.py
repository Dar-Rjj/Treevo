import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Price Momentum Analysis
    data['price_change_5d'] = data['close'].shift(1) - data['close'].shift(5)
    data['price_change_20d'] = data['close'].shift(1) - data['close'].shift(20)
    
    # Movement Efficiency Analysis
    data['prev_close'] = data['close'].shift(1)
    data['true_range'] = np.maximum(data['high'], data['prev_close']) - np.minimum(data['low'], data['prev_close'])
    data['efficiency_ratio'] = np.abs(data['close'] - data['prev_close']) / data['true_range']
    data['efficiency_ratio'] = data['efficiency_ratio'].replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Volume-Price Analysis
    data['volume_change_5d'] = data['volume'].shift(1) - data['volume'].shift(5)
    
    # Calculate rolling correlation between price returns and volume changes
    data['price_returns'] = data['close'].pct_change()
    data['volume_changes'] = data['volume'].pct_change()
    
    # 10-day rolling correlation using only past data
    corr_values = []
    for i in range(len(data)):
        if i < 10:
            corr_values.append(np.nan)
        else:
            start_idx = i - 10
            end_idx = i - 1  # Use t-10 to t-1 (past 10 days excluding current)
            price_window = data['price_returns'].iloc[start_idx:end_idx]
            volume_window = data['volume_changes'].iloc[start_idx:end_idx]
            if len(price_window) >= 2 and len(volume_window) >= 2:
                corr = price_window.corr(volume_window)
                corr_values.append(corr if not pd.isna(corr) else 0)
            else:
                corr_values.append(0)
    
    data['price_volume_corr_10d'] = corr_values
    
    # Signal Integration
    # Momentum divergence
    data['momentum_divergence'] = (np.sign(data['price_change_5d']) != np.sign(data['price_change_20d'])).astype(int)
    
    # Efficiency-volume alignment
    data['efficiency_volume_alignment'] = data['efficiency_ratio'] * np.sign(data['volume_change_5d'])
    
    # Regime detection
    data['regime_detection'] = data['price_volume_corr_10d'] * data['efficiency_ratio']
    
    # Combine signals into final factor
    # Positive when momentum is consistent, efficiency aligns with volume trend, and regime is favorable
    data['factor'] = (
        (1 - data['momentum_divergence']) *  # Penalize momentum divergence
        data['efficiency_volume_alignment'] *  # Efficiency-volume alignment
        data['regime_detection']  # Regime quality
    )
    
    # Return the factor series with proper indexing
    return data['factor']
