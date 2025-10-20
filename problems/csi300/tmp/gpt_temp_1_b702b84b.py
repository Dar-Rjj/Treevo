import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    data = df.copy()
    
    # Multi-Scale Momentum Analysis
    # Calculate 5/10/21-day close returns
    ret_5 = data['close'].pct_change(5)
    ret_10 = data['close'].pct_change(10)
    ret_21 = data['close'].pct_change(21)
    
    # Assess momentum convergence (std of returns)
    momentum_std = pd.concat([ret_5, ret_10, ret_21], axis=1).std(axis=1)
    momentum_convergence = 1 / (1 + momentum_std.rolling(10).mean())
    
    # Volume-Price Dynamics
    # Volume momentum (5/10-day change)
    vol_ma_5 = data['volume'].rolling(5).mean()
    vol_ma_10 = data['volume'].rolling(10).mean()
    volume_momentum = (vol_ma_5 / vol_ma_10 - 1)
    
    # Volume-price confirmation (direction alignment)
    price_trend_5 = data['close'].rolling(5).apply(lambda x: 1 if x.iloc[-1] > x.iloc[0] else -1, raw=False)
    volume_trend_5 = data['volume'].rolling(5).apply(lambda x: 1 if x.iloc[-1] > x.iloc[0] else -1, raw=False)
    volume_price_confirmation = (price_trend_5 * volume_trend_5).fillna(0)
    
    # Volatility Context
    # Range volatility (ATR, daily range)
    daily_range = (data['high'] - data['low']) / data['close']
    atr = daily_range.rolling(14).mean()
    
    # Volatility regime (high/low relative to median)
    vol_regime = (atr > atr.rolling(63).median()).astype(int)
    
    # Convergence Pattern Detection
    # Momentum-volume alignment strength
    momentum_strength = (ret_5.rolling(5).std().fillna(0) + 1e-6)
    volume_strength = (volume_momentum.rolling(5).std().fillna(0) + 1e-6)
    alignment_strength = (ret_5 * volume_momentum) / (momentum_strength * volume_strength)
    
    # Multi-timeframe momentum convergence
    momentum_ranks = pd.concat([ret_5.rank(), ret_10.rank(), ret_21.rank()], axis=1)
    multi_timeframe_convergence = momentum_ranks.std(axis=1).rolling(5).mean()
    multi_timeframe_convergence = 1 / (1 + multi_timeframe_convergence)
    
    # Adaptive Signal Generation
    # Strong trend: convergence + volume confirmation + high volatility
    strong_trend_signal = (momentum_convergence * 
                          volume_price_confirmation * 
                          vol_regime * 
                          alignment_strength.rolling(5).mean())
    
    # Mean reversion: divergence + volume contradiction + low volatility
    mean_reversion_signal = ((1 - momentum_convergence) * 
                            (-volume_price_confirmation) * 
                            (1 - vol_regime) * 
                            (-alignment_strength.rolling(5).mean()))
    
    # Volatility-adjusted composite factor
    volatility_weight = atr.rolling(21).rank(pct=True)
    composite_factor = (strong_trend_signal * volatility_weight + 
                       mean_reversion_signal * (1 - volatility_weight))
    
    # Final factor normalization
    factor = composite_factor.rolling(63).apply(
        lambda x: (x.iloc[-1] - x.mean()) / (x.std() + 1e-6), raw=False
    )
    
    return factor
