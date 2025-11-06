import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Adaptive Momentum Persistence with Volume-Price Divergence Detection
    """
    # Multi-Timeframe Momentum Acceleration
    # Short-term momentum (2-day)
    short_momentum = (df['close'] - df['close'].shift(2)) / df['close'].shift(2)
    
    # Medium-term momentum (5-day)
    medium_momentum = (df['close'] - df['close'].shift(5)) / df['close'].shift(5)
    
    # Momentum acceleration factor
    acceleration = short_momentum - medium_momentum
    bounded_acceleration = np.tanh(acceleration * 10)
    
    # Volume Persistence Analysis
    # Volume trend strength
    volume_ma_3 = df['volume'].rolling(window=3).mean()
    volume_trend = df['volume'] / volume_ma_3
    
    # Volume direction persistence
    volume_above_ma = (df['volume'] > volume_ma_3).astype(int)
    
    # Calculate consecutive days with volume above MA with exponential decay
    volume_persistence = pd.Series(index=df.index, dtype=float)
    current_streak = 0
    for i in range(len(df)):
        if volume_above_ma.iloc[i] == 1:
            current_streak += 1
            # Exponential decay: sum(0.8^i for i=0 to n-1)
            persistence_value = sum(0.8 ** j for j in range(current_streak))
            volume_persistence.iloc[i] = persistence_value
        else:
            current_streak = 0
            volume_persistence.iloc[i] = 0
    
    # Volume-Price Divergence Detection
    divergence_score = pd.Series(0, index=df.index)
    bullish_divergence = (medium_momentum < 0) & (volume_trend > 1)
    bearish_divergence = (medium_momentum > 0) & (volume_trend < 1)
    divergence_score[bullish_divergence] = 1
    divergence_score[bearish_divergence] = -1
    
    # Volatility Regime Detection
    intraday_volatility = (df['high'] - df['low']) / df['close']
    high_vol_regime = (intraday_volatility > 0.02).astype(int)
    
    # Momentum stability assessment
    momentum_std = short_momentum.rolling(window=4).std()
    stability_ratio = 1 / (momentum_std + 0.0001)
    
    # Bounded Transformation Framework
    bounded_momentum = np.tanh(medium_momentum * 5)
    volume_persistence_scaling = 1 - np.exp(-volume_persistence)
    divergence_adjustment = divergence_score * 0.3
    
    # Regime-Adaptive Signal Components
    core_signal = pd.Series(index=df.index, dtype=float)
    
    for i in range(len(df)):
        if high_vol_regime.iloc[i]:
            # High volatility regime weights
            momentum_component = bounded_momentum.iloc[i] * stability_ratio.iloc[i]
            acceleration_component = bounded_acceleration.iloc[i] * 0.3
            volume_component = volume_persistence.iloc[i] * 0.2
            core_signal.iloc[i] = momentum_component + acceleration_component + volume_component
        else:
            # Low volatility regime weights
            momentum_component = bounded_momentum.iloc[i] * 0.5
            acceleration_component = bounded_acceleration.iloc[i] * 0.4
            volume_component = volume_persistence.iloc[i] * 0.1
            core_signal.iloc[i] = momentum_component + acceleration_component + volume_component
    
    # Divergence Adjustment
    divergence_adjusted_signal = core_signal * (1 + divergence_adjustment)
    
    # Persistence Enhancement
    persistence_enhanced = divergence_adjusted_signal * volume_persistence_scaling
    
    # Volatility Regime Confirmation
    final_signal = pd.Series(index=df.index, dtype=float)
    for i in range(len(df)):
        if high_vol_regime.iloc[i]:
            final_signal.iloc[i] = persistence_enhanced.iloc[i] * 0.8
        else:
            final_signal.iloc[i] = persistence_enhanced.iloc[i] * 1.2
    
    return final_signal
