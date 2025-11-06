import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    # Ultra-short momentum signals (1-3 days)
    mom_1d = (df['close'] - df['close'].shift(1)) / df['close'].shift(1)
    mom_2d = (df['close'] - df['close'].shift(2)) / df['close'].shift(2)
    mom_3d = (df['close'] - df['close'].shift(3)) / df['close'].shift(3)
    
    # Conditional momentum convergence - only when all ultra-short signals align
    momentum_convergence = pd.Series(index=df.index, dtype=float)
    for i in range(len(df)):
        if i >= 3:
            current_moms = [mom_1d.iloc[i], mom_2d.iloc[i], mom_3d.iloc[i]]
            if all(x > 0 for x in current_moms):
                # Bullish convergence: geometric mean of positive momentum
                momentum_convergence.iloc[i] = (mom_1d.iloc[i] * mom_2d.iloc[i] * mom_3d.iloc[i]) ** (1/3)
            elif all(x < 0 for x in current_moms):
                # Bearish convergence: geometric mean of negative momentum
                momentum_convergence.iloc[i] = -((abs(mom_1d.iloc[i]) * abs(mom_2d.iloc[i]) * abs(mom_3d.iloc[i])) ** (1/3))
            else:
                # No convergence - zero signal
                momentum_convergence.iloc[i] = 0
    
    # Volume-amount directional alignment
    volume_trend = df['volume'] / df['volume'].rolling(window=5).mean()
    amount_trend = df['amount'] / df['amount'].rolling(window=5).mean()
    
    # Conditional volume-amount synergy - only amplify when both trending together
    volume_amount_synergy = pd.Series(index=df.index, dtype=float)
    for i in range(len(df)):
        if i >= 5:
            vol_dir = np.sign(volume_trend.iloc[i] - 1)
            amt_dir = np.sign(amount_trend.iloc[i] - 1)
            if vol_dir == amt_dir and vol_dir != 0:
                # Both trending in same direction - use geometric mean
                volume_amount_synergy.iloc[i] = (volume_trend.iloc[i] * amount_trend.iloc[i]) ** 0.5
            else:
                # No synergy - neutral impact
                volume_amount_synergy.iloc[i] = 1
    
    # Dynamic volatility scaling with regime detection
    intraday_vol = (df['high'] - df['low']) / df['close']
    price_vol = df['close'].pct_change().abs()
    
    # Short-term volatility regime (3-day)
    short_term_regime = intraday_vol.rolling(window=3).mean()
    # Medium-term volatility regime (6-day)  
    medium_term_regime = price_vol.rolling(window=6).mean()
    
    # Dynamic volatility blend - adapts to current market conditions
    volatility_ratio = short_term_regime / (medium_term_regime + 1e-7)
    dynamic_volatility = np.where(
        volatility_ratio > 1.5, 
        short_term_regime,  # High volatility regime - use short-term
        np.where(
            volatility_ratio < 0.67,
            medium_term_regime,  # Low volatility regime - use medium-term
            (short_term_regime * medium_term_regime) ** 0.5  # Normal regime - blend
        )
    )
    
    # Intraday strength with persistence filter
    intraday_strength = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-7)
    # Only consider persistent intraday strength (2 out of 3 days > 0.5 for bullish, < 0.5 for bearish)
    intraday_persistence = pd.Series(index=df.index, dtype=float)
    for i in range(len(df)):
        if i >= 3:
            recent_strength = intraday_strength.iloc[i-2:i+1]
            bullish_days = sum(recent_strength > 0.5)
            bearish_days = sum(recent_strength < 0.5)
            
            if bullish_days >= 2:
                intraday_persistence.iloc[i] = 1.2  # Amplify bullish signals
            elif bearish_days >= 2:
                intraday_persistence.iloc[i] = 0.8  # Dampen bearish signals
            else:
                intraday_persistence.iloc[i] = 1.0  # Neutral
    
    # Final factor: conditional momentum amplified by volume-amount synergy and intraday persistence,
    # dynamically scaled by adaptive volatility
    raw_factor = momentum_convergence * volume_amount_synergy * intraday_persistence
    factor = raw_factor / (dynamic_volatility + 1e-7)
    
    return factor
