import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Microstructure Divergence
    # Intraday Divergence
    data['intraday_divergence'] = (data['close'] - data['open']) * (data['volume'] - data['volume'].shift(1))
    
    # Gap Divergence
    data['volume_ma_5'] = data['volume'].shift(1).rolling(window=5).mean()
    data['gap_divergence'] = (data['open'] - data['close'].shift(1)) * (data['volume'] - data['volume_ma_5'])
    
    # Range Divergence
    data['range_divergence'] = (data['high'] - data['low']) * (data['amount'] - data['amount'].shift(1))
    
    # Multi-Scale Momentum
    # Ultra-Short Term
    data['opening_momentum'] = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
    data['closing_momentum'] = (data['close'] - data['open']) / data['open']
    data['intraday_acceleration'] = data['closing_momentum'] - data['opening_momentum']
    
    # Short-Term
    data['price_range_momentum'] = (data['high'] - data['low']) / (data['high'].shift(3) - data['low'].shift(3))
    
    # Efficiency Momentum
    current_efficiency = (data['close'] - data['open']) / (data['high'] - data['low'])
    past_efficiency = (data['close'].shift(3) - data['open'].shift(3)) / (data['high'].shift(3) - data['low'].shift(3))
    data['efficiency_momentum'] = current_efficiency - past_efficiency
    
    # Medium-Term
    # Volatility Momentum
    data['vol_short'] = data['close'].rolling(window=10).std()
    data['vol_long'] = data['close'].shift(10).rolling(window=10).std()
    data['volatility_momentum'] = data['vol_short'] / data['vol_long']
    
    # Amount Persistence
    amount_increases = pd.Series(index=data.index, dtype=float)
    for i in range(len(data)):
        if i >= 10:
            window_data = data.iloc[i-9:i+1]
            count = sum(window_data['amount'].iloc[j] > window_data['amount'].iloc[j-1] for j in range(1, 10))
            amount_increases.iloc[i] = count / 10
        else:
            amount_increases.iloc[i] = np.nan
    data['amount_persistence'] = amount_increases
    
    # Regime-Sensitive Factors
    # High Activity
    data['activity_level'] = data['volume'] * (data['high'] - data['low']) / data['close'].shift(1)
    
    high_low_range = data['high'] - data['low']
    data['volatility_breakout'] = high_low_range / high_low_range.shift(1).rolling(window=5).mean()
    
    # Low Activity
    data['quiet_period_detection'] = 1 / (data['volume'] * (data['high'] - data['low']) / data['close'].shift(1))
    data['price_efficiency'] = abs(data['close'] - data['open']) / (data['high'] - data['low'])
    
    # Regime Transition
    data['volume_spike'] = data['volume'] / data['volume'].shift(6).rolling(window=5).mean()
    data['range_expansion'] = high_low_range / high_low_range.shift(6).rolling(window=5).mean()
    
    # Cross-Sectional Dynamics
    # Relative Strength
    data['price_relative'] = data['close'] / data['close'].shift(1).rolling(window=5).mean()
    data['volume_relative'] = data['volume'] / data['volume'].shift(1).rolling(window=5).mean()
    
    # Momentum Divergence
    data['price_volume_momentum_divergence'] = ((data['close']/data['close'].shift(1) - 1) - 
                                               (data['volume']/data['volume'].shift(1) - 1))
    
    data['open_close_momentum_divergence'] = ((data['close']/data['open'] - 1) - 
                                             (data['open']/data['close'].shift(1) - 1))
    
    # Alpha Synthesis
    # Core Factors
    data['divergence_core'] = data['intraday_divergence'] * data['price_volume_momentum_divergence']
    data['momentum_core'] = data['intraday_acceleration'] * data['price_range_momentum']
    data['regime_core'] = data['activity_level'] * data['price_efficiency']
    
    # Final Alpha
    alpha = (data['divergence_core'] * data['volatility_momentum'] + 
             data['momentum_core'] * data['volume_relative'] + 
             data['regime_core'] * data['volume_spike'])
    
    return alpha
