import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Momentum Regime Transition with Volume Confirmation Alpha Factor
    Detects transitions between bullish and bearish regimes using price momentum
    and volume confirmation patterns to predict future returns.
    """
    
    # Initialize result series
    alpha = pd.Series(index=df.index, dtype=float)
    
    # Calculate basic price and volume features
    df = df.copy()
    df['returns'] = df['close'].pct_change()
    df['range'] = (df['high'] - df['low']) / (df['close'].shift(1) + 1e-8)
    df['close_position'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-8)
    df['volume_ma'] = df['volume'].rolling(window=20, min_periods=10).mean()
    df['volume_ratio'] = df['volume'] / (df['volume_ma'] + 1e-8)
    
    # Price momentum features
    df['price_ma_short'] = df['close'].rolling(window=10, min_periods=5).mean()
    df['price_ma_long'] = df['close'].rolling(window=30, min_periods=15).mean()
    df['momentum_5'] = df['close'] / df['close'].shift(5) - 1
    df['momentum_10'] = df['close'] / df['close'].shift(10) - 1
    
    # Regime detection features
    df['higher_highs'] = ((df['high'] > df['high'].shift(1)) & 
                         (df['high'].shift(1) > df['high'].shift(2))).astype(int)
    df['lower_lows'] = ((df['low'] < df['low'].shift(1)) & 
                       (df['low'].shift(1) < df['low'].shift(2))).astype(int)
    
    # Volume confirmation patterns
    df['bull_volume_conf'] = ((df['returns'] > 0) & (df['volume_ratio'] > 1.2) | 
                             (df['returns'] < 0) & (df['volume_ratio'] < 0.8)).astype(int)
    df['bear_volume_conf'] = ((df['returns'] < 0) & (df['volume_ratio'] > 1.2) | 
                             (df['returns'] > 0) & (df['volume_ratio'] < 0.8)).astype(int)
    
    # Regime strength indicators
    df['bull_regime_strength'] = (
        df['higher_highs'].rolling(window=5, min_periods=3).mean() +
        df['bull_volume_conf'].rolling(window=5, min_periods=3).mean() +
        (df['close_position'] > 0.6).rolling(window=5, min_periods=3).mean()
    ) / 3
    
    df['bear_regime_strength'] = (
        df['lower_lows'].rolling(window=5, min_periods=3).mean() +
        df['bear_volume_conf'].rolling(window=5, min_periods=3).mean() +
        (df['close_position'] < 0.4).rolling(window=5, min_periods=3).mean()
    ) / 3
    
    # Transition signals
    df['volume_divergence'] = (
        (df['returns'] > 0) & (df['volume_ratio'] < 0.8) |  # Bullish divergence
        (df['returns'] < 0) & (df['volume_ratio'] < 0.8)    # Bearish divergence
    ).astype(int)
    
    # Intraday reversal patterns
    df['intraday_reversal'] = (
        ((df['high'] - df['close']) / (df['high'] - df['low'] + 1e-8) > 0.6) |  # Strong selling pressure
        ((df['close'] - df['low']) / (df['high'] - df['low'] + 1e-8) > 0.6)     # Strong buying pressure
    ).astype(int)
    
    # Range compression/expansion
    df['range_change'] = df['range'] / df['range'].rolling(window=10, min_periods=5).mean()
    
    # Accumulation/distribution patterns
    df['vwap'] = (df['amount'] / df['volume']).replace([np.inf, -np.inf], np.nan)
    df['price_vwap_ratio'] = df['close'] / (df['vwap'] + 1e-8)
    
    # Transition probability calculation
    for i in range(len(df)):
        if i < 30:  # Need sufficient history
            alpha.iloc[i] = 0
            continue
            
        current_data = df.iloc[:i+1]
        
        # Current regime assessment
        bull_strength = current_data['bull_regime_strength'].iloc[-1]
        bear_strength = current_data['bear_regime_strength'].iloc[-1]
        
        # Early warning signals
        recent_divergence = current_data['volume_divergence'].iloc[-5:].mean()
        recent_reversals = current_data['intraday_reversal'].iloc[-5:].mean()
        range_compression = current_data['range_change'].iloc[-1]
        
        # Momentum building indicators
        momentum_accel = (current_data['momentum_5'].iloc[-1] - 
                         current_data['momentum_10'].iloc[-1])
        vwap_position = current_data['price_vwap_ratio'].iloc[-1]
        
        # Bullish to bearish transition signals
        bull_to_bear = (
            (bull_strength > 0.7) &  # Strong bullish regime
            (recent_divergence > 0.4) &  # Volume divergence
            (range_compression > 1.2) &  # Range expansion
            (vwap_position > 1.02) &  # Price above VWAP
            (momentum_accel < 0)  # Momentum deceleration
        )
        
        # Bearish to bullish transition signals  
        bear_to_bull = (
            (bear_strength > 0.7) &  # Strong bearish regime
            (recent_divergence > 0.4) &  # Volume divergence
            (range_compression > 1.2) &  # Range expansion
            (vwap_position < 0.98) &  # Price below VWAP
            (momentum_accel > 0)  # Momentum acceleration
        )
        
        # Calculate transition alpha
        if bull_to_bear:
            # Negative signal for bullish to bearish transition
            transition_strength = (
                bull_strength * 0.4 +
                recent_divergence * 0.3 +
                range_compression * 0.2 +
                abs(momentum_accel) * 0.1
            )
            alpha.iloc[i] = -transition_strength
            
        elif bear_to_bull:
            # Positive signal for bearish to bullish transition
            transition_strength = (
                bear_strength * 0.4 +
                recent_divergence * 0.3 +
                range_compression * 0.2 +
                abs(momentum_accel) * 0.1
            )
            alpha.iloc[i] = transition_strength
            
        else:
            # No clear transition signal
            alpha.iloc[i] = 0
    
    # Normalize the alpha values
    alpha = (alpha - alpha.rolling(window=60, min_periods=30).mean()) / (alpha.rolling(window=60, min_periods=30).std() + 1e-8)
    
    return alpha
