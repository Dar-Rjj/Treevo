import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(data):
    """
    Generate alpha factor based on price-volume divergence analysis with volatility regime
    classification and key price level context.
    """
    df = data.copy()
    
    # Multi-Timeframe Momentum Calculation
    for period in [5, 10, 20]:
        df[f'price_mom_{period}'] = (df['close'] - df['close'].shift(period)) / df['close'].shift(period)
        df[f'volume_mom_{period}'] = (df['volume'] - df['volume'].shift(period)) / df['volume'].shift(period)
    
    # Divergence Signal Generation
    divergence_signals = []
    timeframe_weights = {'5': 0.3, '10': 0.4, '20': 0.3}
    
    for period in [5, 10, 20]:
        price_mom = df[f'price_mom_{period}']
        volume_mom = df[f'volume_mom_{period}']
        
        # Bullish divergence (price down, volume up)
        bullish_mask = (price_mom < 0) & (volume_mom > 0)
        bullish_strength = abs(price_mom) * volume_mom
        
        # Bearish divergence (price up, volume down)
        bearish_mask = (price_mom > 0) & (volume_mom < 0)
        bearish_strength = price_mom * abs(volume_mom)
        
        # Combine signals
        period_signal = pd.Series(0.0, index=df.index)
        period_signal[bullish_mask] = bullish_strength[bullish_mask]
        period_signal[bearish_mask] = -bearish_strength[bearish_mask]
        
        divergence_signals.append(period_signal * timeframe_weights[str(period)])
    
    # Timeframe Consensus
    base_divergence = sum(divergence_signals)
    
    # Volatility Regime Classification
    # True Range Calculation
    tr1 = df['high'] - df['low']
    tr2 = abs(df['high'] - df['close'].shift(1))
    tr3 = abs(df['low'] - df['close'].shift(1))
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    normalized_tr = true_range / df['close']
    
    # Rolling volatility assessment
    rolling_vol = normalized_tr.rolling(window=15).mean()
    vol_percentile = rolling_vol.rolling(window=60).apply(
        lambda x: (x.iloc[-1] > x).mean(), raw=False
    )
    
    # Volatility regime multipliers
    volatility_multiplier = pd.Series(1.0, index=df.index)
    timeframe_adjustment = pd.Series(1.0, index=df.index)
    
    # Low volatility regime (bottom 30%)
    low_vol_mask = vol_percentile <= 0.3
    volatility_multiplier[low_vol_mask] = 1.6
    # Focus on medium and long-term in low volatility
    if low_vol_mask.any():
        med_long_divergence = sum([divergence_signals[1], divergence_signals[2]])
        base_divergence[low_vol_mask] = med_long_divergence[low_vol_mask]
    
    # High volatility regime (top 30%)
    high_vol_mask = vol_percentile >= 0.7
    volatility_multiplier[high_vol_mask] = 0.6
    # Focus on short-term only in high volatility
    if high_vol_mask.any():
        base_divergence[high_vol_mask] = divergence_signals[0][high_vol_mask]
    
    # Key Price Level Context
    # Support and resistance identification
    recent_high = df['high'].rolling(window=20).max()
    recent_low = df['low'].rolling(window=20).min()
    
    # Position-based multipliers
    price_level_multiplier = pd.Series(1.0, index=df.index)
    
    # Near resistance
    near_resistance = df['close'] >= 0.98 * recent_high
    bearish_signals = base_divergence < 0
    bullish_signals = base_divergence > 0
    
    # Amplify bearish divergence near resistance
    resistance_bearish = near_resistance & bearish_signals
    price_level_multiplier[resistance_bearish] = 1.8
    
    # Suppress bullish divergence near resistance
    resistance_bullish = near_resistance & bullish_signals
    price_level_multiplier[resistance_bullish] = 0.3
    
    # Near support
    near_support = df['close'] <= 1.02 * recent_low
    
    # Amplify bullish divergence near support
    support_bullish = near_support & bullish_signals
    price_level_multiplier[support_bullish] = 1.8
    
    # Suppress bearish divergence near support
    support_bearish = near_support & bearish_signals
    price_level_multiplier[support_bearish] = 0.3
    
    # Composite Alpha Factor
    alpha_factor = base_divergence * volatility_multiplier * price_level_multiplier
    
    # Timeframe consensus reinforcement
    direction_counts = []
    for signal in divergence_signals:
        direction = (signal > 0).astype(int) - (signal < 0).astype(int)
        direction_counts.append(direction)
    
    consensus = sum(direction_counts)
    strong_consensus = abs(consensus) >= 2  # At least 2 out of 3 timeframes agree
    
    # Reinforce strong consensus signals
    alpha_factor[strong_consensus & (consensus > 0)] *= 1.5  # Strong bullish
    alpha_factor[strong_consensus & (consensus < 0)] *= 1.5  # Strong bearish
    
    return alpha_factor
