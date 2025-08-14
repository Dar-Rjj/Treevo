import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df, n=10):
    # Thought 2: Simple Price Movement over N Days
    price_movement = (df['close'] / df['close'].shift(n)) - 1
    
    # Thought 3: High-Low Range Expansion
    range_expansion = (df['high'] - df['low']) / df['close']
    
    # Thought 4: Volume-Weighted Price
    vwp = (df['close'] * df['volume']).rolling(window=n).sum() / df['volume'].rolling(window=n).sum()
    
    # Thought 5: Change in Volume-Weighted Price
    change_in_vwp = (vwp / vwp.shift(1)) - 1
    
    # Thought 7: Money Flow Index (MFI)
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    raw_money_flow = typical_price * df['volume']
    
    positive_flow = np.where(typical_price > typical_price.shift(1), raw_money_flow, 0)
    negative_flow = np.where(typical_price < typical_price.shift(1), raw_money_flow, 0)
    
    positive_mf = pd.Series(positive_flow).rolling(window=n).sum()
    negative_mf = pd.Series(negative_flow).rolling(window=n).sum()
    
    mfi = 100 - (100 / (1 + (positive_mf / negative_mf)))
    
    # Thought 8: Chaikin Oscillator
    money_flow_multiplier = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
    ad_line = money_flow_multiplier * df['volume']
    
    ad_3d_ema = ad_line.ewm(span=3, adjust=False).mean()
    ad_10d_ema = ad_line.ewm(span=10, adjust=False).mean()
    
    chaikin_oscillator = ad_3d_ema - ad_10d_ema
    
    # Thought 10: Momentum-Adjusted Volume-Weighted Price
    momentum_adjusted_vwp = price_movement * change_in_vwp
    
    # Thought 11: Volume-Weighted Momentum
    volume_weighted_momentum = change_in_vwp * (df['close'] / df['close'].shift(n)) - 1
    
    # Combine all factors into a single DataFrame
    factors = pd.DataFrame({
        'price_movement': price_movement,
        'range_expansion': range_expansion,
        'change_in_vwp': change_in_vwp,
        'mfi': mfi,
        'chaikin_oscillator': chaikin_oscillator,
        'momentum_adjusted_vwp': momentum_adjusted_vwp,
        'volume_weighted_momentum': volume_weighted_momentum
    })
    
    # Return the combined factor as the average of all individual factors
    combined_factor = factors.mean(axis=1)
    
    return combined_factor
