import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df, n=10, atr_period=14):
    # Calculate Simple Momentum
    df['simple_momentum'] = df['close'] / df['close'].shift(n) - 1
    
    # Volume Adjusted Component
    df['volume_change'] = df['volume'] / df['volume'].shift(1)
    df['volume_adjusted_momentum'] = df['simple_momentum'] * df['volume_change']
    
    # Average True Range (ATR) Component
    df['true_range'] = df[['high', 'low']].apply(lambda x: max(x['high'] - x['low'], 
                                                               abs(x['high'] - df['close'].shift(1)), 
                                                               abs(x['low'] - df['close'].shift(1))), axis=1)
    df['atr'] = df['true_range'].rolling(window=atr_period).mean()
    df['atr_adjusted_momentum'] = df['simple_momentum'] / df['atr']
    
    # Price Reversal Sensitivity
    df['high_low_spread'] = df['high'] - df['low']
    df['weighted_high_low_spread'] = df['high_low_spread'] * df['volume']
    
    # Final Alpha Factor
    df['combined_alpha_factor'] = df['volume_adjusted_momentum'] + df['atr_adjusted_momentum']
    df['final_alpha_factor'] = df['combined_alpha_factor'] - df['weighted_high_low_spread']
    
    # Introduce Volume Adjustment
    df['volume_ma'] = df['volume'].rolling(window=n).mean()
    df['final_alpha_factor'] = df['final_alpha_factor'] * df['volume_ma']
    
    return df['final_alpha_factor']
