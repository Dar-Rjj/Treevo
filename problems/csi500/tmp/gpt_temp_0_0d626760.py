import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Adaptive Momentum with Volume Confirmation factor
    Combines price momentum with volume validation across multiple timeframes
    """
    data = df.copy()
    
    # Core Momentum Calculation
    # Price Momentum Components
    data['short_term_momentum'] = (data['close'] - data['close'].shift(2)) / data['close'].shift(2)
    data['medium_term_momentum'] = (data['close'] - data['close'].shift(5)) / data['close'].shift(5)
    data['momentum_convergence'] = data['short_term_momentum'] * data['medium_term_momentum']
    
    # Volume Momentum Components
    data['volume_change'] = data['volume'] / data['volume'].shift(5)
    data['volume_acceleration'] = (data['volume'] - data['volume'].shift(5)) / data['volume'].shift(5)
    
    # Volume-Based Momentum Validation
    # Volume-Price Alignment Check
    data['direction_match'] = np.sign(data['short_term_momentum']) * np.sign(data['volume_change'])
    data['strength_match'] = np.abs(data['short_term_momentum']) * np.abs(data['volume_change'])
    
    # Persistence Validation
    # Calculate consecutive alignment days
    alignment_mask = data['direction_match'] > 0
    data['alignment_streak'] = alignment_mask.groupby((alignment_mask != alignment_mask.shift()).cumsum()).cumsum()
    
    # Generate Adaptive Factor
    # Signal Strength Calculation
    data['base_signal'] = data['momentum_convergence'] * data['direction_match']
    data['persistence_multiplier'] = 1 + (data['alignment_streak'] * 0.1)
    data['raw_signal'] = data['base_signal'] * data['persistence_multiplier']
    
    # Market Regime Adjustment
    # Volatility Scaling
    returns = data['close'].pct_change()
    volatility_20d = returns.rolling(window=20).std()
    volatility_60d = returns.rolling(window=60).std()
    data['volatility_regime'] = volatility_20d / volatility_60d
    
    # Trend Condition Filter
    price_trend = data['close'].rolling(window=10).apply(lambda x: 1 if x.iloc[-1] > x.iloc[0] else -1)
    volume_trend = data['volume'].rolling(window=10).apply(lambda x: 1 if x.iloc[-1] > x.iloc[0] else -1)
    data['trend_alignment'] = price_trend * volume_trend
    
    # Final Adaptive Signal
    # Apply volatility scaling (reduce signal in high volatility)
    volatility_weight = 1 / (1 + data['volatility_regime'])
    # Apply trend filter (enhance signal when trends align)
    trend_weight = 1 + (0.2 * data['trend_alignment'])
    
    data['adaptive_factor'] = data['raw_signal'] * volatility_weight * trend_weight
    
    # Clean and return the factor
    factor = data['adaptive_factor'].replace([np.inf, -np.inf], np.nan).fillna(0)
    
    return factor
