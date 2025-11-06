import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Extract price and volume data
    high = df['high']
    low = df['low']
    close = df['close']
    volume = df['volume']
    
    # Initialize result series
    alpha_signal = pd.Series(index=df.index, dtype=float)
    
    # Calculate rolling windows
    for i in range(15, len(df)):
        # Volatility Regime Detection
        current_range = (high.iloc[i] - low.iloc[i]) / close.iloc[i]
        
        # Short-term volatility baseline (4-day window)
        vol_window = [(high.iloc[j] - low.iloc[j]) / close.iloc[j] for j in range(i-4, i)]
        short_term_vol = np.mean(vol_window)
        
        # Regime classification
        if current_range > 1.5 * short_term_vol:
            regime = 'high'
            momentum_lookback = 5
        elif current_range < 0.7 * short_term_vol:
            regime = 'low'
            momentum_lookback = 15
        else:
            regime = 'normal'
            momentum_lookback = 8
        
        # Regime-optimized momentum calculation
        if i >= momentum_lookback:
            base_signal = (close.iloc[i] / close.iloc[i - momentum_lookback]) - 1
        else:
            base_signal = 0
        
        # Volume Breakout Confirmation
        # Volume surge detection
        volume_window = [volume.iloc[j] for j in range(i-4, i)]
        avg_volume = np.mean(volume_window)
        volume_intensity = volume.iloc[i] / avg_volume
        
        # Volume breakout and alignment
        volume_breakout = volume_intensity > 2.0
        
        if i >= 3:
            volume_momentum = (volume.iloc[i] / volume.iloc[i-3]) - 1
            price_momentum_3d = (close.iloc[i] / close.iloc[i-3]) - 1
            alignment_strength = min(abs(price_momentum_3d), abs(volume_momentum))
        else:
            volume_breakout = False
            alignment_strength = 0
        
        # Momentum Consistency Framework
        if i >= 10:
            # Multi-timeframe momentum calculation
            vs_momentum = (close.iloc[i] / close.iloc[i-2]) - 1 if i >= 2 else 0
            s_momentum = (close.iloc[i] / close.iloc[i-5]) - 1 if i >= 5 else 0
            m_momentum = (close.iloc[i] / close.iloc[i-10]) - 1 if i >= 10 else 0
            
            # Count aligned signs
            signs = [np.sign(vs_momentum), np.sign(s_momentum), np.sign(m_momentum)]
            positive_count = sum(1 for sign in signs if sign > 0)
            negative_count = sum(1 for sign in signs if sign < 0)
            
            if positive_count == 3 or negative_count == 3:
                consistency_multiplier = 1.3
            elif positive_count == 2 or negative_count == 2:
                consistency_multiplier = 1.1
            else:
                consistency_multiplier = 0.7
        else:
            consistency_multiplier = 1.0
        
        # Alpha Signal Integration
        final_signal = base_signal
        
        # Volume breakout enhancement
        if volume_breakout:
            if alignment_strength > 0:
                final_signal = base_signal * (1 + alignment_strength)
            else:
                final_signal = base_signal * 1.2
        
        # Momentum consistency adjustment
        final_signal = final_signal * consistency_multiplier
        
        alpha_signal.iloc[i] = final_signal
    
    return alpha_signal
