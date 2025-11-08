import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Volatility-Adjusted Momentum Divergence with Volume Confirmation alpha factor
    """
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate momentum components
    data['momentum_5d'] = (data['close'].shift(1) - data['close'].shift(5)) / data['close'].shift(5)
    data['momentum_10d'] = (data['close'].shift(1) - data['close'].shift(10)) / data['close'].shift(10)
    data['momentum_20d'] = (data['close'].shift(1) - data['close'].shift(20)) / data['close'].shift(20)
    
    # Calculate daily returns for volatility
    data['returns'] = data['close'].pct_change()
    
    # Calculate historical volatility (20-day rolling std of returns)
    data['volatility_20d'] = data['returns'].rolling(window=20).std()
    
    # Calculate volatility regime thresholds using 60-day lookback
    data['volatility_percentile'] = data['volatility_20d'].rolling(window=60).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) == 60 else np.nan, raw=False
    )
    
    # Assign volatility regime weights
    conditions = [
        data['volatility_percentile'] < 0.4,
        data['volatility_percentile'] > 0.6
    ]
    choices = [
        'low',    # Low volatility regime
        'high'    # High volatility regime
    ]
    data['vol_regime'] = np.select(conditions, choices, default='normal')
    
    # Calculate volume ratio
    data['volume_20d_avg'] = data['volume'].rolling(window=20).mean()
    data['volume_ratio'] = data['volume'] / data['volume_20d_avg']
    
    # Assign volume confirmation levels
    volume_conditions = [
        data['volume_ratio'] > 2.0,
        (data['volume_ratio'] > 1.0) & (data['volume_ratio'] <= 2.0),
        (data['volume_ratio'] > 0.5) & (data['volume_ratio'] <= 1.0),
        data['volume_ratio'] <= 0.5
    ]
    volume_choices = ['strong', 'moderate', 'weak', 'contradiction']
    data['volume_confirmation'] = np.select(volume_conditions, volume_choices, default='weak')
    
    # Initialize weighted momentum score
    data['weighted_momentum'] = 0.0
    
    # Calculate weighted momentum based on volatility regime
    for regime in ['low', 'normal', 'high']:
        mask = data['vol_regime'] == regime
        
        if regime == 'low':
            weights = [0.6, 0.3, 0.1]  # short, medium, long
        elif regime == 'high':
            weights = [0.2, 0.3, 0.5]  # short, medium, long
        else:  # normal
            weights = [0.4, 0.4, 0.2]  # short, medium, long
            
        data.loc[mask, 'weighted_momentum'] = (
            data.loc[mask, 'momentum_5d'] * weights[0] +
            data.loc[mask, 'momentum_10d'] * weights[1] +
            data.loc[mask, 'momentum_20d'] * weights[2]
        )
    
    # Apply volume multipliers
    volume_multipliers = {
        'strong': 1.5,
        'moderate': 1.2,
        'weak': 0.8,
        'contradiction': 0.5
    }
    
    data['volume_multiplier'] = data['volume_confirmation'].map(volume_multipliers)
    data['alpha_factor'] = data['weighted_momentum'] * data['volume_multiplier']
    
    return data['alpha_factor']
