import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Generate alpha factor combining volatility-adjusted momentum, volume-price confirmation,
    gap and intraday momentum, and multi-timeframe breakout signals.
    """
    # Volatility-Adjusted Momentum
    # Multi-Timeframe Momentum
    momentum_3d = (df['close'] - df['close'].shift(3)) / df['close'].shift(3)
    momentum_5d = (df['close'] - df['close'].shift(5)) / df['close'].shift(5)
    momentum_10d = (df['close'] - df['close'].shift(10)) / df['close'].shift(10)
    
    # Volatility Scaling
    volatility_5d = ((df['high'] - df['low']) / df['close']).rolling(window=5).mean()
    volatility_10d = ((df['high'] - df['low']) / df['close']).rolling(window=10).mean()
    
    # Combined Signal
    momentum_avg = (momentum_3d + momentum_5d + momentum_10d) / 3
    volatility_adjusted_momentum = momentum_avg / volatility_5d
    
    # Volume-Price Confirmation
    # Price Strength
    intraday_strength = (df['close'] - df['open']) / (df['high'] - df['low']).replace(0, np.nan)
    close_position = (df['close'] - df['low']) / (df['high'] - df['low']).replace(0, np.nan)
    price_change_efficiency = (df['close'] - df['close'].shift(1)) / (df['high'] - df['low']).replace(0, np.nan)
    
    # Volume Confirmation
    volume_ratio = df['volume'] / df['volume'].rolling(window=5).mean()
    volume_trend = df['volume'] / df['volume'].shift(1)
    
    # Combined Signal
    price_strength_avg = (intraday_strength + close_position + price_change_efficiency) / 3
    volume_confirmed = price_strength_avg * volume_ratio
    
    # Gap and Intraday Momentum
    # Gap Analysis
    overnight_gap = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
    gap_strength = np.abs(overnight_gap) / ((df['high'] - df['low']) / df['close']).replace(0, np.nan)
    
    # Intraday Momentum
    morning_momentum = (df['high'] - df['open']) / df['open'].replace(0, np.nan)
    afternoon_momentum = (df['close'] - df['low']) / df['low'].replace(0, np.nan)
    intraday_consistency = (df['close'] - df['open']) / (df['high'] - df['low']).replace(0, np.nan)
    
    # Combined Signal
    gap_adjusted_momentum = (morning_momentum + afternoon_momentum) / 2
    gap_intraday_signal = gap_adjusted_momentum * intraday_consistency
    
    # Multi-Timeframe Breakout
    # Price Breakout
    short_term_breakout = df['close'] / df['high'].rolling(window=5).apply(lambda x: x[:-1].max() if len(x) > 1 else np.nan)
    medium_term_breakout = df['close'] / df['high'].rolling(window=10).apply(lambda x: x[:-1].max() if len(x) > 1 else np.nan)
    breakout_strength = (short_term_breakout + medium_term_breakout) / 2
    
    # Volume Breakout
    volume_surge = df['volume'] / df['volume'].rolling(window=5).mean()
    volume_persistence = (df['volume'].rolling(window=3).mean() / 
                         df['volume'].rolling(window=3).apply(lambda x: x.mean() if len(x) == 3 else np.nan).shift(3))
    
    # Combined Signal
    price_volume_breakout = breakout_strength * volume_surge
    confirmed_breakout = price_volume_breakout * volume_persistence
    
    # Final alpha factor - combine all components with equal weighting
    alpha_factor = (
        volatility_adjusted_momentum + 
        volume_confirmed + 
        gap_intraday_signal + 
        confirmed_breakout
    ) / 4
    
    return alpha_factor
