import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    # Multi-timeframe price-volume convergence with regime-based signal amplification
    
    # Price momentum across multiple timeframes (1, 3, 6, 9 days)
    mom_1d = (df['close'] - df['close'].shift(1)) / df['close'].shift(1)
    mom_3d = (df['close'] - df['close'].shift(3)) / df['close'].shift(3)
    mom_6d = (df['close'] - df['close'].shift(6)) / df['close'].shift(6)
    mom_9d = (df['close'] - df['close'].shift(9)) / df['close'].shift(9)
    
    # Volume momentum across same timeframes
    vol_mom_1d = (df['volume'] - df['volume'].shift(1)) / df['volume'].shift(1)
    vol_mom_3d = (df['volume'] - df['volume'].shift(3)) / df['volume'].shift(3)
    vol_mom_6d = (df['volume'] - df['volume'].shift(6)) / df['volume'].shift(6)
    vol_mom_9d = (df['volume'] - df['volume'].shift(9)) / df['volume'].shift(9)
    
    # Price-volume convergence score
    price_volume_convergence = pd.Series(index=df.index, dtype=float)
    
    for i in range(len(df)):
        if i >= 9:
            price_moms = [mom_1d.iloc[i], mom_3d.iloc[i], mom_6d.iloc[i], mom_9d.iloc[i]]
            volume_moms = [vol_mom_1d.iloc[i], vol_mom_3d.iloc[i], vol_mom_6d.iloc[i], vol_mom_9d.iloc[i]]
            
            # Count directional alignment between price and volume
            aligned_signals = sum(1 for p, v in zip(price_moms, volume_moms) if p * v > 0)
            
            if aligned_signals >= 3:  # Strong convergence
                # Geometric mean of absolute momentum values for strong signals
                price_strength = (abs(mom_1d.iloc[i]) * abs(mom_3d.iloc[i]) * 
                                abs(mom_6d.iloc[i]) * abs(mom_9d.iloc[i])) ** 0.25
                volume_strength = (abs(vol_mom_1d.iloc[i]) * abs(vol_mom_3d.iloc[i]) * 
                                 abs(vol_mom_6d.iloc[i]) * abs(vol_mom_9d.iloc[i])) ** 0.25
                
                # Direction determined by price momentum majority
                positive_price = sum(1 for p in price_moms if p > 0)
                direction = 1 if positive_price >= 2 else -1
                
                price_volume_convergence.iloc[i] = direction * price_strength * volume_strength
                
            else:  # Weak or mixed convergence
                # Weighted average favoring recent timeframes
                weights = [0.4, 0.3, 0.2, 0.1]
                price_weighted = sum(w * p for w, p in zip(weights, price_moms))
                volume_weighted = sum(w * v for w, v in zip(weights, volume_moms))
                price_volume_convergence.iloc[i] = price_weighted * volume_weighted
    
    # Regime detection: Price, Volume, and Volatility regimes
    # Price regime: Bullish vs Bearish (using 5-day vs 15-day moving averages)
    price_ma_short = df['close'].rolling(window=5).mean()
    price_ma_long = df['close'].rolling(window=15).mean()
    price_regime = (price_ma_short > price_ma_long).astype(int)  # 1 = bullish, 0 = bearish
    
    # Volume regime: High vs Low volume (relative to 10-day average)
    volume_ma = df['volume'].rolling(window=10).mean()
    volume_regime = (df['volume'] > volume_ma).astype(int)  # 1 = high volume, 0 = low volume
    
    # Volatility regime: High vs Low volatility (using intraday range)
    intraday_range = (df['high'] - df['low']) / df['close']
    volatility_ma = intraday_range.rolling(window=10).mean()
    volatility_regime = (intraday_range > volatility_ma).astype(int)  # 1 = high vol, 0 = low vol
    
    # Regime-based amplification factor
    regime_amplification = pd.Series(index=df.index, dtype=float)
    
    for i in range(len(df)):
        if i >= 15:
            regime_score = 0
            
            # Bullish price regime with high volume: strong amplification
            if price_regime.iloc[i] == 1 and volume_regime.iloc[i] == 1:
                regime_score += 2
            
            # Bearish price regime with high volume: moderate amplification (contrarian)
            if price_regime.iloc[i] == 0 and volume_regime.iloc[i] == 1:
                regime_score += 1
            
            # High volatility regime: dampen signals (more noise)
            if volatility_regime.iloc[i] == 1:
                regime_score -= 1
            
            # Convert regime score to amplification factor
            if regime_score >= 2:
                regime_amplification.iloc[i] = 2.0  # Strong bullish with volume
            elif regime_score == 1:
                regime_amplification.iloc[i] = 1.5  # Moderate signal
            elif regime_score == 0:
                regime_amplification.iloc[i] = 1.0  # Neutral
            elif regime_score == -1:
                regime_amplification.iloc[i] = 0.7  # Dampened in high volatility
            else:
                regime_amplification.iloc[i] = 0.5  # Strongly dampened
    
    # Final factor: Price-volume convergence amplified by regime conditions
    factor = price_volume_convergence * regime_amplification
    
    return factor
