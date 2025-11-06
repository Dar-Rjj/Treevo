import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import pearsonr

def heuristics_v2(data):
    """
    Multi-Timeframe Price-Volume Convergence with Adaptive Weighting
    """
    df = data.copy()
    
    # Ultra-Short Term Component (3-day window)
    # Price Acceleration
    returns_3d = df['close'].pct_change(periods=1)
    latest_return = returns_3d
    prev_return = returns_3d.shift(1)
    oldest_return = returns_3d.shift(2)
    price_acceleration = (latest_return - prev_return) * (prev_return - oldest_return)
    
    # Volume Surge Detection
    volume_ma_3d = df['volume'].rolling(window=3, min_periods=3).mean().shift(1)
    volume_ratio = df['volume'] / volume_ma_3d
    volume_surge = volume_ratio - 1
    
    # Combined Ultra-Short Signal
    ultra_short_convergence = price_acceleration * volume_surge
    
    # Short-Term Component (8-day window)
    # Price-Volume Correlation
    price_changes = []
    volume_changes = []
    
    for i in range(8):
        price_change = df['close'].pct_change(periods=i+1)
        volume_change = df['volume'].pct_change(periods=i+1)
        price_changes.append(price_change)
        volume_changes.append(volume_change)
    
    # Calculate rolling correlation
    correlation_signal = pd.Series(index=df.index, dtype=float)
    for i in range(8, len(df)):
        if i >= 8:
            price_window = [pc.iloc[i] for pc in price_changes if not pd.isna(pc.iloc[i])]
            volume_window = [vc.iloc[i] for vc in volume_changes if not pd.isna(vc.iloc[i])]
            if len(price_window) >= 3 and len(volume_window) >= 3:
                try:
                    corr, _ = pearsonr(price_window, volume_window)
                    correlation_signal.iloc[i] = corr if not np.isnan(corr) else 0
                except:
                    correlation_signal.iloc[i] = 0
            else:
                correlation_signal.iloc[i] = 0
    
    # Trend Consistency
    price_direction = np.sign(df['close'] - df['close'].shift(7))
    volume_direction = np.sign(df['volume'] - df['volume'].shift(7))
    direction_alignment = price_direction * volume_direction
    
    # Combined Short-Term Signal
    short_term_convergence = correlation_signal * direction_alignment
    
    # Medium-Term Component (21-day window)
    # Price Reversal Strength
    early_trend = (df['close'].shift(14) - df['close'].shift(20)) / df['close'].shift(20)
    late_trend = (df['close'] - df['close'].shift(14)) / df['close'].shift(14)
    reversal_strength = late_trend - early_trend
    
    # Volume Confirmation Pattern
    early_volume = df['volume'].rolling(window=7, min_periods=7).mean().shift(14)
    late_volume = df['volume'].rolling(window=7, min_periods=7).mean()
    volume_shift = (late_volume - early_volume) / early_volume
    
    # Combined Medium-Term Signal
    medium_term_convergence = reversal_strength * volume_shift
    
    # Signal Enhancement
    ultra_short_enhanced = ultra_short_convergence * (1 + abs(ultra_short_convergence))
    short_term_enhanced = np.tanh(short_term_convergence * 2)
    medium_term_enhanced = medium_term_convergence / (1 + abs(medium_term_convergence))
    
    # Adaptive Weighting
    # Market Regime Detection
    recent_volatility = df['close'].pct_change().rolling(window=5, min_periods=5).std()
    vol_20d = df['close'].pct_change().rolling(window=20, min_periods=20).std()
    vol_median = vol_20d.rolling(window=20, min_periods=20).median()
    
    volatility_regime = (recent_volatility > vol_median).astype(int)
    regime_adjustment = np.where(volatility_regime == 1, 1.2, 0.8)
    
    # Weight Calculation
    ultra_short_weight = 0.3 * regime_adjustment
    short_term_weight = 0.4 * regime_adjustment
    medium_term_weight = 0.3 * regime_adjustment
    
    # Normalized Weights
    total_weight = ultra_short_weight + short_term_weight + medium_term_weight
    norm_ultra_short_weight = ultra_short_weight / total_weight
    norm_short_term_weight = short_term_weight / total_weight
    norm_medium_term_weight = medium_term_weight / total_weight
    
    # Final Alpha Factor
    alpha_factor = (norm_ultra_short_weight * ultra_short_enhanced + 
                   norm_short_term_weight * short_term_enhanced + 
                   norm_medium_term_weight * medium_term_enhanced)
    
    return alpha_factor
