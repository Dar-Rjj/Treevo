import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Multi-timeframe Volatility Assessment
    data['vol_5d'] = data['close'].rolling(window=5).std()
    data['vol_10d'] = data['close'].rolling(window=10).std()
    data['vol_ratio'] = data['vol_5d'] / data['vol_10d']
    
    # Volatility Regime Detection
    conditions = [
        data['vol_ratio'] > 1.2,
        data['vol_ratio'] < 0.8,
        (data['vol_ratio'] >= 0.8) & (data['vol_ratio'] <= 1.2)
    ]
    choices = ['high', 'low', 'normal']
    data['vol_regime'] = np.select(conditions, choices, default='normal')
    
    # Price Movement Efficiency
    data['daily_efficiency'] = np.abs(data['close'] - data['open']) / (data['high'] - data['low'])
    data['cumulative_efficiency_5d'] = data['daily_efficiency'].rolling(window=5).sum()
    data['cumulative_efficiency_10d'] = data['daily_efficiency'].rolling(window=10).sum()
    data['efficiency_momentum'] = data['daily_efficiency'] / data['daily_efficiency'].shift(1)
    
    # Volume-Weighted Price Impact
    data['price_range'] = np.abs(data['close'] - data['open'])
    
    # Calculate rolling correlation between volume and price range
    vol_price_corr = []
    for i in range(len(data)):
        if i >= 4:
            window_vol = data['volume'].iloc[i-4:i+1]
            window_price = data['price_range'].iloc[i-4:i+1]
            corr_val = window_vol.corr(window_price) if len(window_vol) > 1 and window_vol.std() > 0 and window_price.std() > 0 else 0
            vol_price_corr.append(corr_val)
        else:
            vol_price_corr.append(0)
    data['volume_price_correlation'] = vol_price_corr
    
    # Volume efficiency components
    data['volume_price_product'] = data['volume'] * data['price_range']
    data['volume_price_5d_avg'] = data['volume_price_product'].rolling(window=5).mean()
    data['abnormal_volume_impact'] = data['volume_price_product'] / data['volume_price_5d_avg']
    data['volume_efficiency'] = (data['price_range'] * data['volume']) / (data['high'] - data['low'])
    
    # Multi-timeframe Efficiency Divergence
    data['efficiency_divergence'] = data['cumulative_efficiency_5d'] / data['cumulative_efficiency_10d']
    data['volume_price_alignment'] = np.sign(data['volume_price_correlation']) * np.abs(data['volume_efficiency'])
    data['efficiency_regime_consistency'] = data['efficiency_momentum'] * data['vol_ratio']
    
    # Volatility-Adjusted Efficiency Scoring
    high_vol_score = np.where(data['vol_regime'] == 'high', 
                             data['volume_efficiency'] * (1 + data['vol_ratio']), 0)
    low_vol_score = np.where(data['vol_regime'] == 'low', 
                            data['daily_efficiency'] * data['volume_price_correlation'], 0)
    normal_vol_score = np.where(data['vol_regime'] == 'normal', 
                               data['cumulative_efficiency_5d'] * data['volume_price_alignment'], 0)
    
    data['regime_weighted_signal'] = high_vol_score + low_vol_score + normal_vol_score
    
    # Regime Transition Signals
    data['volatility_breakout'] = data['vol_ratio'] * data['efficiency_momentum']
    data['efficiency_regime_shift'] = data['efficiency_divergence'] * data['vol_ratio']
    data['transition_momentum'] = data['volatility_breakout'] * data['efficiency_regime_shift']
    
    # Composite Alpha Factor
    data['transition_enhancement'] = data['regime_weighted_signal'] * (1 + data['transition_momentum'])
    data['final_factor'] = data['transition_enhancement'] * np.sign(data['volume_price_correlation'])
    
    # Handle NaN values
    data['final_factor'] = data['final_factor'].fillna(0)
    
    return data['final_factor']
