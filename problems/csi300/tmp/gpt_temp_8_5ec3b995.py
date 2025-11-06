import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate basic efficiency components
    data['intraday_efficiency'] = (data['close'] - data['open']) * data['volume'] / (data['high'] - data['low']).replace(0, np.nan)
    data['gap_efficiency'] = (data['open'] - data['close'].shift(1)) * data['volume'] / (data['high'] - data['low']).replace(0, np.nan)
    data['efficiency'] = data['intraday_efficiency'] + data['gap_efficiency']
    data['efficiency_momentum'] = data['efficiency'] - data['efficiency'].shift(1)
    
    # Ultra-Short Term (1-day) components
    data['efficiency_velocity'] = (data['close'] - data['open']) / (data['high'] - data['low']).replace(0, np.nan)
    data['gap_impact'] = abs(data['open'] - data['close'].shift(1)) / (data['high'] - data['low']).replace(0, np.nan)
    
    # Short-Term (5-day) components
    data['efficiency_trend'] = data['efficiency'] / data['efficiency'].shift(5)
    data['price_momentum_5d'] = data['close'] / data['close'].shift(5)
    data['regime_signal'] = np.sign(data['efficiency_trend']) * np.sign(data['price_momentum_5d'])
    
    # Medium-Term (20-day) components
    # Calculate efficiency persistence (count of efficiency increases over 20 days)
    efficiency_increases = pd.Series(index=data.index, dtype=float)
    for i in range(20, len(data)):
        window = data['efficiency'].iloc[i-20:i]
        efficiency_increases.iloc[i] = (window.diff().dropna() > 0).sum()
    data['efficiency_persistence'] = efficiency_increases
    
    data['momentum_consistency'] = data['close'] / data['close'].shift(20)
    data['regime_strength'] = data['efficiency_persistence'] * data['momentum_consistency']
    
    # Volume-Flow Divergence components
    data['volume_acceleration'] = data['volume'] / data['volume'].shift(5)
    data['flow_imbalance'] = (data['amount'] - data['amount'].shift(1)) / (data['high'] - data['low']).replace(0, np.nan)
    data['efficiency_flow_divergence'] = data['efficiency_momentum'] * data['flow_imbalance']
    data['volume_efficiency_alignment'] = data['volume_acceleration'] * data['efficiency_momentum']
    
    # Volatility-Adjusted Signals
    data['realized_volatility_ratio'] = (data['high'] - data['low']) / (data['high'].shift(5) - data['low'].shift(5)).replace(0, np.nan)
    data['efficiency_volatility'] = abs(data['efficiency'] - data['efficiency'].shift(1)) / (data['high'].shift(1) - data['low'].shift(1)).replace(0, np.nan)
    data['volatility_regime'] = data['realized_volatility_ratio'] * data['efficiency_volatility']
    
    # Composite Alpha Construction
    data['regime_score'] = data['regime_signal'] * data['regime_strength']
    data['divergence_score'] = data['efficiency_flow_divergence'] * data['volume_efficiency_alignment']
    data['volatility_score'] = 1 / data['volatility_regime'].replace(0, np.nan)
    
    # Final Alpha
    data['alpha'] = data['regime_score'] * data['divergence_score'] * data['volatility_score'] * data['flow_imbalance']
    
    return data['alpha']
