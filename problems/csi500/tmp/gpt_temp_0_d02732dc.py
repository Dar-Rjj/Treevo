import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(data):
    """
    Generate alpha factor using multi-timeframe price-volume divergence with dynamic regime detection
    and volume confirmation.
    """
    df = data.copy()
    
    # Calculate returns for volatility computation
    df['returns'] = df['close'] / df['close'].shift(1) - 1
    
    # Multi-Timeframe Price-Volume Divergence
    timeframes = [1, 3, 8, 21, 55]
    divergence_columns = []
    
    for tf in timeframes:
        price_momentum = df['close'] / df['close'].shift(tf)
        volume_momentum = df['volume'] / df['volume'].shift(tf)
        divergence = price_momentum - volume_momentum
        col_name = f'divergence_{tf}d'
        df[col_name] = divergence
        divergence_columns.append(col_name)
    
    # Dynamic Volatility-Momentum Regime Detection
    # Volatility Structure Analysis
    df['ultra_short_vol'] = df['returns'].rolling(window=2).std()
    df['short_vol'] = df['returns'].rolling(window=7).std()
    df['medium_vol'] = df['returns'].rolling(window=20).std()
    df['vol_acceleration'] = df['ultra_short_vol'] / df['short_vol']
    df['vol_persistence'] = df['short_vol'] / df['medium_vol']
    
    # Momentum Regime Assessment
    df['short_momentum'] = df['close'] / df['close'].shift(5)
    df['medium_momentum'] = df['close'] / df['close'].shift(20)
    df['momentum_consistency'] = np.sign(df['short_momentum'] - 1) * np.sign(df['medium_momentum'] - 1)
    
    # Adaptive Timeframe Weighting
    weights = pd.DataFrame(index=df.index, columns=['ultra_short', 'very_short', 'short', 'medium', 'long'])
    
    for i in range(len(df)):
        if i < 55:  # Skip initial period with insufficient data
            continue
            
        vol_acc = df['vol_acceleration'].iloc[i]
        vol_pers = df['vol_persistence'].iloc[i]
        mom_cons = df['momentum_consistency'].iloc[i]
        
        if vol_acc > 1.5 and mom_cons == 1:
            weights.iloc[i] = [0.4, 0.3, 0.2, 0.1, 0.0]
        elif vol_acc > 1.5 and mom_cons == -1:
            weights.iloc[i] = [0.3, 0.3, 0.2, 0.1, 0.1]
        elif 0.8 <= vol_acc <= 1.5 and vol_pers > 1.1:
            weights.iloc[i] = [0.1, 0.2, 0.3, 0.3, 0.1]
        elif 0.8 <= vol_acc <= 1.5 and vol_pers <= 1.1:
            weights.iloc[i] = [0.2, 0.2, 0.2, 0.2, 0.2]
        elif vol_acc < 0.8:
            weights.iloc[i] = [0.1, 0.1, 0.2, 0.3, 0.3]
        else:
            weights.iloc[i] = [0.2, 0.2, 0.2, 0.2, 0.2]
    
    # Multi-dimensional Volume Analysis
    df['volume_ma_10'] = df['volume'].rolling(window=10).mean()
    df['volume_std_10'] = df['volume'].rolling(window=10).std()
    df['volume_zscore'] = (df['volume'] - df['volume_ma_10']) / df['volume_std_10']
    
    # Volume Spike Classification
    volume_multiplier = np.ones(len(df))
    volume_multiplier[df['volume_zscore'] > 3] = 2.5
    volume_multiplier[(df['volume_zscore'] > 2) & (df['volume_zscore'] <= 3)] = 1.8
    volume_multiplier[(df['volume_zscore'] > 1) & (df['volume_zscore'] <= 2)] = 1.3
    
    # Volume-Price Confirmation
    up_volume_days = np.zeros(len(df))
    down_volume_days = np.zeros(len(df))
    
    for i in range(5, len(df)):
        window = df.iloc[i-4:i+1]
        up_count = ((window['close'] > window['close'].shift(1)) & 
                   (window['volume'] > window['volume_ma_10'])).sum()
        down_count = ((window['close'] < window['close'].shift(1)) & 
                     (window['volume'] > window['volume_ma_10'])).sum()
        up_volume_days[i] = up_count
        down_volume_days[i] = down_count
    
    df['volume_bias'] = (up_volume_days - down_volume_days) / 5
    
    # Price Level and Trend Context
    df['short_high'] = df['high'].rolling(window=10).max()
    df['short_low'] = df['low'].rolling(window=10).min()
    df['medium_high'] = df['high'].rolling(window=30).max()
    df['medium_low'] = df['low'].rolling(window=30).min()
    
    df['short_position'] = (df['close'] - df['short_low']) / (df['short_high'] - df['short_low'])
    df['medium_position'] = (df['close'] - df['medium_low']) / (df['medium_high'] - df['medium_low'])
    df['trend_alignment'] = df['short_position'] - df['medium_position']
    
    # Contextual Multipliers
    contextual_multiplier = np.ones(len(df))
    breakout_cond = (df['short_position'] > 0.9) & (df['trend_alignment'] > 0.1)
    resistance_cond = (df['short_position'] > 0.8) & (df['trend_alignment'] < 0)
    support_cond = (df['short_position'] < 0.2) & (df['trend_alignment'] > 0)
    breakdown_cond = (df['short_position'] < 0.1) & (df['trend_alignment'] < -0.1)
    
    contextual_multiplier[breakout_cond] = 1.4
    contextual_multiplier[resistance_cond] = 0.8
    contextual_multiplier[support_cond] = 1.2
    contextual_multiplier[breakdown_cond] = 0.6
    
    # Final Alpha Construction
    # Multi-timeframe Divergence Blend
    divergence_blend = (
        df['divergence_1d'] * weights['ultra_short'] +
        df['divergence_3d'] * weights['very_short'] +
        df['divergence_8d'] * weights['short'] +
        df['divergence_21d'] * weights['medium'] +
        df['divergence_55d'] * weights['long']
    )
    
    # Volume Enhanced Score
    volume_enhanced_score = divergence_blend * volume_multiplier
    
    # Intraday Pattern Score (simplified - using daily data only)
    # For intraday patterns, we use simplified proxies since we only have daily data
    early_momentum = (df['high'] - df['open']) / df['open']
    final_hour_strength = (df['close'] - df['low']) / df['low']
    intraday_pattern_score = volume_enhanced_score * (1 + early_momentum * final_hour_strength)
    
    # Volume Confirmation Score
    volume_confirmation_score = intraday_pattern_score * (1 + df['volume_bias'] * 0.2)
    
    # Context Final Alpha
    final_alpha = volume_confirmation_score * contextual_multiplier
    
    return final_alpha
