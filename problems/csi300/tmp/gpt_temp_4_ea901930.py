import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(data):
    """
    Generate a novel alpha factor combining price-volume interaction, market efficiency,
    capital flow analysis, and volatility context.
    """
    df = data.copy()
    
    # Price-Volume Interaction
    # Momentum Divergence
    price_momentum = (df['close'] / df['close'].shift(5)) - 1
    volume_momentum = df['volume'] / df['volume'].shift(5)
    divergence_factor = price_momentum / volume_momentum
    
    # Reversal Signals
    price_reversal = -np.sign(df['close'] - df['close'].shift(1)) * np.abs(df['close'] / df['close'].shift(1) - 1)
    volume_confirmation = df['volume'] / df['volume'].shift(1)
    reversal_strength = price_reversal * volume_confirmation
    
    # Market Efficiency
    # Single Period Efficiency
    price_movement = np.abs(df['close'] - df['close'].shift(1))
    trading_range = df['high'] - df['low']
    efficiency = price_movement / trading_range
    
    # Multi-period Efficiency
    net_movement = np.abs(df['close'] - df['close'].shift(3))
    total_range = (df['high'] - df['low']) + (df['high'].shift(1) - df['low'].shift(1)) + (df['high'].shift(2) - df['low'].shift(2))
    efficiency_persistence = net_movement / total_range
    
    # Capital Flow Analysis
    # Directional Flow
    price_direction = np.sign(df['close'] - df['close'].shift(1))
    flow_magnitude = df['amount'] * np.abs(df['close'] - df['close'].shift(1)) / df['close'].shift(1)
    net_flow = df['amount'] * price_direction
    
    # Flow Quality
    direction_consistency = pd.Series(index=df.index, dtype=float)
    for i in range(len(df)):
        if i >= 5:
            current_sign = np.sign(df['close'].iloc[i] - df['close'].iloc[i-1])
            count = 0
            for j in range(5):
                if i-j-1 >= 0:
                    prev_sign = np.sign(df['close'].iloc[i-j] - df['close'].iloc[i-j-1])
                    if prev_sign == current_sign:
                        count += 1
            direction_consistency.iloc[i] = count
        else:
            direction_consistency.iloc[i] = np.nan
    
    flow_persistence = net_flow + net_flow.shift(1) + net_flow.shift(2)
    quality_score = flow_persistence * direction_consistency
    
    # Volatility Context
    # Volatility Regimes
    short_term_range = df['high'].rolling(window=6).max() - df['low'].rolling(window=6).min()
    long_term_range = df['high'].rolling(window=11).max() - df['low'].rolling(window=11).min()
    volatility_state = short_term_range / long_term_range
    
    # Volume Context
    volume_level = df['volume'] / df['volume'].rolling(window=11).median()
    
    volume_trend = pd.Series(index=df.index, dtype=float)
    for i in range(len(df)):
        if i >= 5:
            count = 0
            for j in range(5):
                if i-j-1 >= 0:
                    if df['volume'].iloc[i-j] > df['volume'].iloc[i-j-1]:
                        count += 1
            volume_trend.iloc[i] = count
        else:
            volume_trend.iloc[i] = np.nan
    
    context_factor = volatility_state * volume_level * volume_trend
    
    # Combine all components
    momentum_component = divergence_factor.rank(pct=True) + reversal_strength.rank(pct=True)
    efficiency_component = efficiency.rank(pct=True) + efficiency_persistence.rank(pct=True)
    flow_component = net_flow.rank(pct=True) + quality_score.rank(pct=True)
    volatility_component = context_factor.rank(pct=True)
    
    # Final alpha factor
    alpha_factor = (
        momentum_component * 0.25 +
        efficiency_component * 0.25 +
        flow_component * 0.25 +
        volatility_component * 0.25
    )
    
    return alpha_factor
