import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    # Multi-timeframe momentum convergence with regime-aware volatility scaling and volume confirmation
    
    # Multi-timeframe momentum signals
    # Ultra-short (1-day) - immediate price reaction
    mom_1d = (df['close'] - df['close'].shift(1)) / df['close'].shift(1)
    
    # Short-term (3-day) - recent trend strength
    mom_3d = (df['close'] - df['close'].shift(3)) / df['close'].shift(3)
    
    # Medium-term (6-day) - intermediate trend
    mom_6d = (df['close'] - df['close'].shift(6)) / df['close'].shift(6)
    
    # Long-term (12-day) - primary trend direction
    mom_12d = (df['close'] - df['close'].shift(12)) / df['close'].shift(12)
    
    # Momentum convergence factor: measures alignment across timeframes
    momentum_convergence = pd.Series(index=df.index, dtype=float)
    for i in range(len(df)):
        if i >= 12:
            moms = [mom_1d.iloc[i], mom_3d.iloc[i], mom_6d.iloc[i], mom_12d.iloc[i]]
            
            # Calculate convergence strength based on directional alignment
            positive_alignment = sum(m for m in moms if m > 0)
            negative_alignment = sum(abs(m) for m in moms if m < 0)
            
            if positive_alignment > negative_alignment:
                # Bullish convergence: strength increases with more timeframes aligned
                convergence_strength = positive_alignment / (len([m for m in moms if m > 0]) + 1e-7)
                momentum_convergence.iloc[i] = convergence_strength
            elif negative_alignment > positive_alignment:
                # Bearish convergence: strength increases with more timeframes aligned
                convergence_strength = negative_alignment / (len([m for m in moms if m < 0]) + 1e-7)
                momentum_convergence.iloc[i] = -convergence_strength
            else:
                # Neutral: use weighted average favoring shorter timeframes
                weights = [0.4, 0.3, 0.2, 0.1]
                momentum_convergence.iloc[i] = sum(w * m for w, m in zip(weights, moms))
    
    # Volume confirmation: measure if volume supports price movement
    volume_confirmation = pd.Series(index=df.index, dtype=float)
    for i in range(len(df)):
        if i >= 6:
            current_volume = df['volume'].iloc[i]
            avg_volume = df['volume'].iloc[i-5:i+1].mean()
            volume_ratio = current_volume / (avg_volume + 1e-7)
            
            # Volume confirms momentum if both move in same direction with above-average volume
            if momentum_convergence.iloc[i] > 0 and volume_ratio > 1:
                volume_confirmation.iloc[i] = volume_ratio
            elif momentum_convergence.iloc[i] < 0 and volume_ratio > 1:
                volume_confirmation.iloc[i] = -volume_ratio
            else:
                volume_confirmation.iloc[i] = 0
    
    # Regime-aware volatility scaling
    # Short-term volatility (intraday)
    intraday_vol = (df['high'] - df['low']).rolling(window=3).mean() / df['close']
    
    # Medium-term volatility (close-to-close)
    close_vol = df['close'].pct_change().rolling(window=6).std()
    
    # Volatility regime detection
    vol_regime = (intraday_vol > intraday_vol.rolling(window=12).mean()).astype(int)
    
    # Adaptive volatility scaling based on regime
    regime_volatility = pd.Series(index=df.index, dtype=float)
    for i in range(len(df)):
        if i >= 12:
            if vol_regime.iloc[i] == 1:  # High volatility regime
                # In high volatility, emphasize intraday movements and use conservative scaling
                regime_volatility.iloc[i] = (intraday_vol.iloc[i] * 0.8 + close_vol.iloc[i] * 0.2)
            else:  # Low volatility regime
                # In low volatility, emphasize trend persistence with moderate scaling
                regime_volatility.iloc[i] = (intraday_vol.iloc[i] * 0.4 + close_vol.iloc[i] * 0.6)
    
    # Final factor: momentum convergence amplified by volume confirmation,
    # dynamically scaled by regime-aware volatility
    factor = momentum_convergence * (1 + abs(volume_confirmation)) / (regime_volatility + 1e-7)
    
    return factor
