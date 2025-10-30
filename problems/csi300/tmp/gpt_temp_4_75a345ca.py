import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Price Momentum Component
    # Short-term momentum (5-day)
    data['short_momentum'] = data['close'].shift(1) / data['close'].shift(5) - 1
    
    # Medium-term momentum (20-day)
    data['medium_momentum'] = data['close'].shift(1) / data['close'].shift(20) - 1
    
    # Movement Efficiency Component
    # True Range calculation
    data['prev_close'] = data['close'].shift(1)
    data['tr1'] = data['high'] - data['low']
    data['tr2'] = abs(data['high'] - data['prev_close'])
    data['tr3'] = abs(data['low'] - data['prev_close'])
    data['true_range'] = data[['tr1', 'tr2', 'tr3']].max(axis=1)
    
    # Price Movement Efficiency
    data['abs_price_change'] = abs(data['close'] - data['prev_close'])
    data['movement_efficiency'] = data['abs_price_change'] / data['true_range']
    data['movement_efficiency'] = data['movement_efficiency'].replace([np.inf, -np.inf], np.nan)
    
    # Volume-Price Alignment Component
    # Volume momentum (5-day)
    data['volume_momentum'] = data['volume'].shift(1) / data['volume'].shift(5) - 1
    
    # Price-Volume Correlation (10-day rolling)
    data['price_returns'] = data['close'].pct_change().shift(1)
    data['volume_returns'] = data['volume'].pct_change().shift(1)
    
    # Calculate rolling correlation
    price_vol_corr = []
    for i in range(len(data)):
        if i < 10:
            price_vol_corr.append(np.nan)
            continue
        
        start_idx = i - 9
        end_idx = i
        
        price_window = data['price_returns'].iloc[start_idx:end_idx+1]
        volume_window = data['volume_returns'].iloc[start_idx:end_idx+1]
        
        if len(price_window.dropna()) >= 5 and len(volume_window.dropna()) >= 5:
            corr = price_window.corr(volume_window)
            price_vol_corr.append(corr if not pd.isna(corr) else 0)
        else:
            price_vol_corr.append(0)
    
    data['price_volume_correlation'] = price_vol_corr
    
    # Combined Signal Generation
    # Momentum Divergence Analysis
    data['momentum_divergence'] = np.sign(data['short_momentum']) != np.sign(data['medium_momentum'])
    
    # Efficiency-Volume Alignment
    data['efficiency_volume_alignment'] = (
        (data['movement_efficiency'] > data['movement_efficiency'].rolling(20, min_periods=10).mean()) &
        (data['price_volume_correlation'] > 0.1)
    )
    
    # Volume-Price Regime Detection
    data['trend_continuation'] = (
        (data['price_volume_correlation'] > 0.2) & 
        (data['movement_efficiency'] > data['movement_efficiency'].rolling(20, min_periods=10).mean())
    )
    
    data['distribution_phase'] = (
        (data['price_volume_correlation'] < -0.1) & 
        (data['movement_efficiency'] < data['movement_efficiency'].rolling(20, min_periods=10).mean())
    )
    
    data['early_reversal'] = (
        data['momentum_divergence'] & 
        (abs(data['price_volume_correlation']) > 0.15)
    )
    
    # Final factor calculation
    data['factor'] = (
        # Base momentum strength
        data['short_momentum'] * 0.3 +
        data['medium_momentum'] * 0.2 +
        
        # Efficiency component
        data['movement_efficiency'] * 0.15 +
        
        # Volume alignment
        data['price_volume_correlation'] * 0.1 +
        data['volume_momentum'] * 0.05 +
        
        # Regime signals
        data['trend_continuation'].astype(float) * 0.1 -
        data['distribution_phase'].astype(float) * 0.1 -
        data['early_reversal'].astype(float) * 0.1
    )
    
    # Clean up intermediate columns
    cols_to_drop = ['prev_close', 'tr1', 'tr2', 'tr3', 'true_range', 'abs_price_change', 
                   'price_returns', 'volume_returns', 'momentum_divergence', 
                   'efficiency_volume_alignment', 'trend_continuation', 
                   'distribution_phase', 'early_reversal']
    data = data.drop(columns=[col for col in cols_to_drop if col in data.columns])
    
    return data['factor']
