import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Generate composite alpha factor combining price momentum, volume confirmation, 
    and volatility-adjusted efficiency.
    """
    # Price-based components
    # Short-term momentum (5-day)
    momentum_5d = df['close'] / df['close'].shift(5) - 1
    
    # Medium-term reversal (20-day)
    reversal_20d = df['close'] / df['close'].shift(20) - 1
    
    # Volatility-adjusted efficiency ratio
    price_change_20d = df['close'] - df['close'].shift(20)
    volatility_20d = df['close'].rolling(window=20).std()
    efficiency_ratio = price_change_20d / (volatility_20d + 1e-8)
    
    # Volume-based components
    # Volume-price trend (5-day)
    vpt = (df['volume'] * (df['close'] - df['close'].shift(1))).rolling(window=5).sum()
    
    # Abnormal volume (current vs 20-day average)
    avg_volume_20d = df['volume'].rolling(window=20).mean().shift(1)
    abnormal_volume = df['volume'] / (avg_volume_20d + 1e-8)
    
    # Amount-based components
    # Large order activity (amount per trade)
    amount_per_trade = df['amount'] / (df['volume'] + 1e-8)
    
    # Amount momentum (current vs 5-day average)
    avg_amount_5d = df['amount'].rolling(window=5).mean().shift(1)
    amount_momentum = df['amount'] / (avg_amount_5d + 1e-8)
    
    # Technical patterns
    # Body-to-range ratio
    body_range_ratio = (df['close'] - df['open']) / (df['high'] - df['low'] + 1e-8)
    
    # Trend consistency (5-day sign consistency)
    price_changes = df['close'].diff()
    trend_signs = np.sign(price_changes.rolling(window=5).apply(lambda x: x[x != 0].iloc[-1] if len(x[x != 0]) > 0 else 0))
    trend_consistency = (price_changes.rolling(window=5).apply(lambda x: np.mean(np.sign(x) == trend_signs.iloc[-1]) 
                                                              if trend_signs.iloc[-1] != 0 else 0) 
                        if len(trend_signs) > 0 else 0)
    
    # Composite factor construction
    # Weight short-term momentum more heavily with volume confirmation
    momentum_component = momentum_5d * (1 + 0.5 * np.tanh(abnormal_volume - 1))
    
    # Combine reversal with efficiency for mean-reversion signal
    reversal_component = reversal_20d * efficiency_ratio
    
    # Incorporate large order activity with price patterns
    order_flow_component = amount_momentum * body_range_ratio
    
    # Final composite factor
    alpha_factor = (
        0.4 * momentum_component +
        0.3 * reversal_component +
        0.2 * vpt / (df['volume'].rolling(window=5).sum() + 1e-8) +
        0.1 * order_flow_component * trend_consistency
    )
    
    return alpha_factor
