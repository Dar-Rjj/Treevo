import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Volatility-Regime Adaptive Alpha Factor
    Combines volatility regime detection with regime-specific price patterns and volume confirmation
    """
    data = df.copy()
    
    # Volatility Regime Identification
    # Short-term vs long-term volatility ratio (5-day vs 20-day)
    short_term_vol = data['close'].pct_change().rolling(window=5).std()
    long_term_vol = data['close'].pct_change().rolling(window=20).std()
    vol_ratio = short_term_vol / long_term_vol
    
    # Volatility regime classification
    high_vol_threshold = long_term_vol.rolling(window=60).quantile(0.7)
    low_vol_threshold = long_term_vol.rolling(window=60).quantile(0.3)
    
    # Regime persistence (how long current regime has lasted)
    regime_persistence = pd.Series(index=data.index, dtype=float)
    current_regime = 0
    persistence_count = 0
    
    for i in range(len(data)):
        if i < 60:
            regime_persistence.iloc[i] = 0
            continue
            
        if long_term_vol.iloc[i] > high_vol_threshold.iloc[i]:
            if current_regime == 1:
                persistence_count += 1
            else:
                current_regime = 1
                persistence_count = 1
        elif long_term_vol.iloc[i] < low_vol_threshold.iloc[i]:
            if current_regime == -1:
                persistence_count += 1
            else:
                current_regime = -1
                persistence_count = 1
        else:
            current_regime = 0
            persistence_count = 0
            
        regime_persistence.iloc[i] = persistence_count
    
    # Price Pattern Effectiveness by Regime
    # High volatility: breakout momentum (5-day high/low breakout)
    high_breakout = (data['close'] - data['high'].rolling(window=5).max()) / data['close']
    low_breakout = (data['close'] - data['low'].rolling(window=5).min()) / data['close']
    
    # Low volatility: mean reversion (RSI-based mean reversion)
    returns = data['close'].pct_change()
    rsi_period = 14
    gains = returns.where(returns > 0, 0).rolling(window=rsi_period).mean()
    losses = (-returns).where(returns < 0, 0).rolling(window=rsi_period).mean()
    rsi = 100 - (100 / (1 + gains / (losses + 1e-8)))
    mean_reversion_signal = (rsi - 50) / 50  # Normalized around 50
    
    # Volume-Volatility Relationship
    # Volume concentration during high volatility
    volume_ma = data['volume'].rolling(window=20).mean()
    volume_spike = data['volume'] / volume_ma
    
    # Volume efficiency (price movement per unit volume)
    price_range = (data['high'] - data['low']) / data['close']
    volume_efficiency = price_range / (data['volume'] + 1e-8)
    normalized_volume_efficiency = volume_efficiency / volume_efficiency.rolling(window=20).mean()
    
    # Volume divergence (price vs volume direction)
    price_trend = data['close'].pct_change(3)
    volume_trend = data['volume'].pct_change(3)
    volume_divergence = np.sign(price_trend) * np.sign(volume_trend) * np.abs(price_trend)
    
    # Intraday Price Dynamics
    # Opening session volatility concentration
    open_gap = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
    morning_range = (data['high'].rolling(window=3).apply(lambda x: x[0] if len(x) == 3 else np.nan) - 
                    data['low'].rolling(window=3).apply(lambda x: x[0] if len(x) == 3 else np.nan)) / data['close']
    
    # Closing session price positioning
    close_position = (data['close'] - data['low']) / (data['high'] - data['low'] + 1e-8)
    
    # Intraday range efficiency
    intraday_range = (data['high'] - data['low']) / data['close']
    intraday_efficiency = (data['close'] - data['open']) / (intraday_range + 1e-8)
    
    # Regime-Adaptive Factor Construction
    # Volatility-weighted signals
    regime_weight = np.where(long_term_vol > high_vol_threshold, 
                           vol_ratio,  # Emphasize momentum in high vol
                           np.where(long_term_vol < low_vol_threshold,
                                  1/vol_ratio,  # Emphasize mean reversion in low vol
                                  1.0))  # Neutral in transition
    
    # High volatility regime: breakout signals with volume confirmation
    high_vol_signal = (high_breakout - low_breakout) * volume_spike * regime_persistence
    
    # Low volatility regime: mean reversion with volume efficiency
    low_vol_signal = mean_reversion_signal * normalized_volume_efficiency * (1 + regime_persistence/10)
    
    # Transition regime: balanced approach
    transition_signal = (intraday_efficiency * close_position * volume_divergence)
    
    # Combine signals based on regime
    factor = pd.Series(index=data.index, dtype=float)
    
    for i in range(len(data)):
        if i < 60:
            factor.iloc[i] = 0
            continue
            
        if long_term_vol.iloc[i] > high_vol_threshold.iloc[i]:
            # High volatility regime - momentum focus
            signal = high_vol_signal.iloc[i]
        elif long_term_vol.iloc[i] < low_vol_threshold.iloc[i]:
            # Low volatility regime - mean reversion focus
            signal = low_vol_signal.iloc[i]
        else:
            # Transition regime - balanced approach
            signal = transition_signal.iloc[i]
        
        # Apply regime persistence smoothing
        if regime_persistence.iloc[i] > 0:
            smoothing_factor = min(1.0, regime_persistence.iloc[i] / 10)
            factor.iloc[i] = signal * smoothing_factor
        else:
            factor.iloc[i] = signal
    
    # Final normalization and risk adjustment
    factor = factor / (factor.rolling(window=20).std() + 1e-8)
    factor = factor.fillna(0)
    
    return factor
