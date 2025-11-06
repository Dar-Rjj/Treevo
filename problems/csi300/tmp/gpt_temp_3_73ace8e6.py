import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Generate novel alpha factor combining momentum, range expansion, and volatility-adaptive signals
    """
    data = df.copy()
    
    # Price Momentum with Volume Confirmation
    # Multi-timeframe momentum
    data['momentum_short'] = (data['close'] - data['close'].shift(3)) / data['close'].shift(3)
    data['momentum_medium'] = (data['close'] - data['close'].shift(8)) / data['close'].shift(8)
    data['momentum_long'] = (data['close'] - data['close'].shift(21)) / data['close'].shift(21)
    
    # Momentum divergence patterns
    data['bullish_divergence'] = ((data['momentum_short'] > data['momentum_medium']) & 
                                 (data['momentum_medium'] > data['momentum_long'])).astype(int)
    data['bearish_divergence'] = ((data['momentum_short'] < data['momentum_medium']) & 
                                 (data['momentum_medium'] < data['momentum_long'])).astype(int)
    
    # Volume confirmation
    data['volume_positive_momentum'] = data['volume'].rolling(window=5).apply(
        lambda x: x[x.index[-1]] if (data.loc[x.index[-1], 'momentum_short'] > 0) else 0
    )
    data['volume_negative_momentum'] = data['volume'].rolling(window=5).apply(
        lambda x: x[x.index[-1]] if (data.loc[x.index[-1], 'momentum_short'] < 0) else 0
    )
    data['volume_confirmation_ratio'] = (data['volume_positive_momentum'] - 
                                       data['volume_negative_momentum']) / data['volume'].rolling(window=5).mean()
    
    # Range Expansion Efficiency
    # Range expansion patterns
    data['daily_range'] = data['high'] - data['low']
    data['range_change'] = data['daily_range'] - data['daily_range'].shift(1)
    
    # Directional efficiency
    data['position_in_range'] = (data['close'] - data['low']) / (data['high'] - data['low'] + 1e-8)
    data['efficiency'] = abs(data['close'] - data['open']) / (data['high'] - data['low'] + 1e-8)
    
    # Order flow alignment
    data['directional_amount_flow'] = data['amount'] * np.sign(data['close'] - data['close'].shift(1))
    data['amount_flow_ma'] = data['directional_amount_flow'].rolling(window=5).mean()
    
    # Volatility-Regime Adaptive Momentum
    # Volatility regime shifts
    data['daily_volatility'] = (data['high'] - data['low']) / data['close'].shift(1)
    data['volatility_regime'] = data['daily_volatility'].rolling(window=10).std()
    
    # Adaptive momentum
    data['price_momentum'] = data['close'] / data['close'].shift(1) - 1
    data['volatility_adjusted_momentum'] = data['price_momentum'] / (data['volatility_regime'] + 1e-8)
    
    # Price impact efficiency
    data['price_impact'] = abs(data['close'] - data['close'].shift(1)) / (data['volume'] + 1e-8)
    data['price_impact_ma'] = data['price_impact'].rolling(window=5).mean()
    
    # Factor Integration
    # Combine momentum signals
    momentum_component = (data['momentum_short'] * 0.4 + 
                         data['momentum_medium'] * 0.3 + 
                         data['momentum_long'] * 0.3)
    
    # Range expansion component
    range_component = (data['range_change'] * data['efficiency'] * 
                      data['position_in_range'] * data['amount_flow_ma'])
    
    # Volatility-adaptive component
    volatility_component = (data['volatility_adjusted_momentum'] * 
                           (1 - data['price_impact_ma']))
    
    # Final factor integration
    data['alpha_factor'] = (
        momentum_component * 0.4 +
        range_component * 0.35 +
        volatility_component * 0.25 +
        data['volume_confirmation_ratio'] * 0.1 +
        (data['bullish_divergence'] - data['bearish_divergence']) * 0.05
    )
    
    # Normalize the factor
    data['alpha_factor'] = (data['alpha_factor'] - data['alpha_factor'].rolling(window=20).mean()) / data['alpha_factor'].rolling(window=20).std()
    
    return data['alpha_factor']
