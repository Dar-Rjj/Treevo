import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Generate alpha factor combining multiple technical indicators:
    - Volume-confirmed momentum
    - Intraday efficiency
    - Volume-weighted breakout strength
    - Gap filling efficiency
    - Volume-adjusted price trend
    - Amount-based price impact
    - Support level confirmation
    """
    result = pd.Series(index=df.index, dtype=float)
    
    for i in range(5, len(df)):
        # Volume-Confirmed Momentum
        price_momentum = (df['close'].iloc[i] - df['close'].iloc[i-5]) / df['close'].iloc[i-5]
        volume_momentum = (df['volume'].iloc[i] - df['volume'].iloc[i-5]) / df['volume'].iloc[i-5]
        volume_confirmed_momentum = price_momentum * volume_momentum
        
        # Intraday Efficiency Signal
        daily_range = df['high'].iloc[i] - df['low'].iloc[i]
        net_movement = abs(df['close'].iloc[i] - df['open'].iloc[i])
        efficiency_ratio = net_movement / daily_range if daily_range > 0 else 0
        
        # Volume-Weighted Breakout Strength
        resistance_level = max(df['high'].iloc[i-4:i+1])
        breakout_pct = (df['close'].iloc[i] - resistance_level) / resistance_level if resistance_level > 0 else 0
        volume_weighted_breakout = breakout_pct * df['volume'].iloc[i]
        
        # Gap Filling Efficiency
        opening_gap = df['open'].iloc[i] - df['close'].iloc[i-1]
        intraday_gap_fill = df['close'].iloc[i] - df['open'].iloc[i]
        gap_interaction = opening_gap * intraday_gap_fill
        
        # Volume-Adjusted Price Trend
        cumulative_price_change = df['close'].iloc[i] - df['close'].iloc[i-3]
        cumulative_volume = sum(df['volume'].iloc[i-2:i+1])
        volume_normalized_trend = cumulative_price_change / cumulative_volume if cumulative_volume > 0 else 0
        
        # Amount-Based Price Impact
        daily_price_movement = df['close'].iloc[i] - df['close'].iloc[i-1]
        trading_impact = daily_price_movement / df['amount'].iloc[i] if df['amount'].iloc[i] > 0 else 0
        
        # Support Level Confirmation
        support_level = min(df['low'].iloc[i-4:i+1])
        support_bounce = df['close'].iloc[i] - support_level
        volume_confirmed_support = support_bounce * df['volume'].iloc[i]
        
        # Combine all signals with equal weights
        combined_signal = (
            volume_confirmed_momentum +
            efficiency_ratio +
            volume_weighted_breakout +
            gap_interaction +
            volume_normalized_trend +
            trading_impact +
            volume_confirmed_support
        )
        
        result.iloc[i] = combined_signal
    
    return result
