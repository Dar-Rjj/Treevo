import pandas as pd
import numpy as np
import pandas as pd

def heuristics_v2(data):
    """
    Multi-Timeframe Trend Confirmation Factor
    Combines 40-day primary trend with 60-day confirmation trend
    """
    df = data.copy()
    
    # Calculate 40-day trends
    df['price_return_40'] = df['close'] / df['close'].shift(40) - 1
    df['volume_return_40'] = df['volume'] / df['volume'].shift(40) - 1
    
    # Calculate 60-day trends
    df['price_return_60'] = df['close'] / df['close'].shift(60) - 1
    df['volume_return_60'] = df['volume'] / df['volume'].shift(60) - 1
    
    # Base signal: 40-day price return × 40-day volume return
    df['base_signal'] = df['price_return_40'] * df['volume_return_40']
    
    # Determine primary trend direction
    conditions_primary = [
        (df['price_return_40'] > 0) & (df['volume_return_40'] > 0),  # Bullish primary
        (df['price_return_40'] < 0) & (df['volume_return_40'] < 0),  # Bearish primary
        True  # No clear primary trend
    ]
    choices_primary = ['bullish', 'bearish', 'none']
    df['primary_trend'] = pd.Categorical(pd.np.select(conditions_primary, choices_primary, default='none'))
    
    # Determine confirmation status
    conditions_confirmation = [
        (df['price_return_60'] > 0) & (df['volume_return_60'] > 0),  # Confirmed bullish
        (df['price_return_60'] < 0) & (df['volume_return_60'] < 0),  # Confirmed bearish
        True  # Unconfirmed trend
    ]
    choices_confirmation = ['confirmed_bullish', 'confirmed_bearish', 'unconfirmed']
    df['confirmation_status'] = pd.Categorical(pd.np.select(conditions_confirmation, choices_confirmation, default='unconfirmed'))
    
    # Calculate confirmation multiplier
    conditions_multiplier = [
        (df['primary_trend'] == 'bullish') & (df['confirmation_status'] == 'confirmed_bullish'),
        (df['primary_trend'] == 'bearish') & (df['confirmation_status'] == 'confirmed_bearish'),
        (df['primary_trend'] == 'bullish') & (df['confirmation_status'] == 'unconfirmed'),
        (df['primary_trend'] == 'bearish') & (df['confirmation_status'] == 'unconfirmed'),
        (df['primary_trend'] == 'none') & (df['confirmation_status'].isin(['confirmed_bullish', 'confirmed_bearish'])),
        (df['primary_trend'] == 'none') & (df['confirmation_status'] == 'unconfirmed')
    ]
    choices_multiplier = [2.0, 2.0, 1.0, 1.0, 0.1, 0.0]
    df['confirmation_multiplier'] = pd.np.select(conditions_multiplier, choices_multiplier, default=0.0)
    
    # Final factor
    factor = df['base_signal'] * df['confirmation_multiplier']
    
    return factor
