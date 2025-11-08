import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(data):
    """
    Multi-Timeframe Volatility-Adaptive Momentum with Volume-Price Synergy factor
    """
    df = data.copy()
    
    # Multi-Timeframe Momentum Framework
    # Ultra-Short Momentum (1-3 days)
    df['ret_1d'] = df['close'] / df['close'].shift(1) - 1
    df['ret_3d'] = df['close'] / df['close'].shift(3) - 1
    df['hl_range_3d'] = (df['high'] - df['low']) / df['close']
    
    # Short-Term Momentum (5-10 days)
    df['ret_5d'] = df['close'] / df['close'].shift(5) - 1
    df['ret_10d'] = df['close'] / df['close'].shift(10) - 1
    
    # True Range calculation for 10-day efficiency
    df['tr'] = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            abs(df['high'] - df['close'].shift(1)),
            abs(df['low'] - df['close'].shift(1))
        )
    )
    df['sum_tr_10d'] = df['tr'].rolling(window=10).sum()
    df['tr_efficiency_10d'] = (df['close'] - df['close'].shift(10)) / df['sum_tr_10d']
    
    # Medium-Term Momentum (20-30 days)
    df['ret_20d'] = df['close'] / df['close'].shift(20) - 1
    df['ret_30d'] = df['close'] / df['close'].shift(30) - 1
    
    # Momentum persistence
    df['ret_direction'] = np.sign(df['ret_1d'])
    df['momentum_persistence'] = 0
    for i in range(1, len(df)):
        if df['ret_direction'].iloc[i] == df['ret_direction'].iloc[i-1]:
            df['momentum_persistence'].iloc[i] = df['momentum_persistence'].iloc[i-1] + 1
        else:
            df['momentum_persistence'].iloc[i] = 0
    
    # Volatility Regime Adaptive System
    # Volatility Spectrum Calculation
    df['vol_ultra_short'] = df['ret_1d'].rolling(window=5).std()
    df['vol_short'] = df['ret_1d'].rolling(window=10).std()
    df['vol_medium'] = df['ret_1d'].rolling(window=20).std()
    df['vol_ratio'] = df['vol_short'] / df['vol_medium']
    
    # Regime Classification
    conditions = [
        df['vol_ultra_short'] > (2 * df['vol_medium']),  # Explosive
        df['vol_ratio'] > 1.5,  # High
        (df['vol_ratio'] >= 0.8) & (df['vol_ratio'] <= 1.2),  # Normal
        df['vol_ratio'] < 0.8,  # Low
        df['vol_ultra_short'] < (0.5 * df['vol_medium'])  # Compressed
    ]
    choices = ['explosive', 'high', 'normal', 'low', 'compressed']
    df['vol_regime'] = np.select(conditions, choices, default='normal')
    
    # Volume-Price Synergy Analysis
    # Volume Momentum Structure
    df['volume_acceleration'] = df['volume'] / df['volume'].shift(5)
    df['volume_trend_strength'] = df['volume'] / df['volume'].rolling(window=20).mean()
    df['volume_consistency'] = df['volume'] / df['volume'].rolling(window=5).mean()
    
    # Price-Volume Alignment
    df['volume_change'] = df['volume'].pct_change()
    
    # Calculate rolling correlation
    corr_values = []
    for i in range(len(df)):
        if i >= 10:
            window_ret = df['ret_1d'].iloc[i-9:i+1]
            window_vol = df['volume_change'].iloc[i-9:i+1]
            corr = window_ret.corr(window_vol)
            corr_values.append(corr if not np.isnan(corr) else 0)
        else:
            corr_values.append(0)
    df['return_volume_corr'] = corr_values
    
    # Volume concentration
    df['up_day'] = df['ret_1d'] > 0
    df['down_day'] = df['ret_1d'] < 0
    
    up_volume = []
    down_volume = []
    for i in range(len(df)):
        if i >= 10:
            up_mask = df['up_day'].iloc[i-9:i+1]
            down_mask = df['down_day'].iloc[i-9:i+1]
            up_vol_sum = df['volume'].iloc[i-9:i+1][up_mask].sum()
            down_vol_sum = df['volume'].iloc[i-9:i+1][down_mask].sum()
            up_volume.append(up_vol_sum if up_vol_sum > 0 else 1)
            down_volume.append(down_vol_sum if down_vol_sum > 0 else 1)
        else:
            up_volume.append(1)
            down_volume.append(1)
    
    df['volume_concentration'] = np.array(up_volume) / np.array(down_volume)
    df['volume_efficiency'] = abs(df['ret_5d']) / df['volume'].rolling(window=5).mean()
    
    # Amount-Based Confirmation
    df['amount_per_volume'] = df['amount'] / df['volume']
    df['amount_momentum'] = df['amount'] / df['amount'].shift(5)
    df['large_trade_concentration'] = df['amount'] / df['amount'].rolling(window=20).mean()
    
    # Integrated Factor Construction
    # Multi-Timeframe Momentum Score
    momentum_scores = []
    for i in range(len(df)):
        regime = df['vol_regime'].iloc[i]
        
        if regime == 'explosive':
            momentum = 0.7 * df['ret_1d'].iloc[i] + 0.3 * df['ret_30d'].iloc[i]
        elif regime == 'high':
            momentum = 0.6 * df['ret_10d'].iloc[i] + 0.4 * df['ret_30d'].iloc[i]
        elif regime == 'normal':
            momentum = (df['ret_1d'].iloc[i] + df['ret_10d'].iloc[i] + df['ret_30d'].iloc[i]) / 3
        elif regime == 'low':
            momentum = 0.7 * df['ret_30d'].iloc[i] + 0.3 * df['ret_10d'].iloc[i]
        else:  # compressed
            momentum = 0.8 * df['ret_30d'].iloc[i] + 0.2 * df['ret_1d'].iloc[i]
        
        # Apply momentum persistence multiplier (1.0 to 1.5)
        persistence_multiplier = 1.0 + min(0.5, df['momentum_persistence'].iloc[i] * 0.1)
        momentum *= persistence_multiplier
        
        # Adjust for high-low range efficiency
        range_efficiency = 1.0 + (df['hl_range_3d'].iloc[i] * 0.5)
        momentum *= range_efficiency
        
        momentum_scores.append(momentum)
    
    df['momentum_score'] = momentum_scores
    
    # Volume-Price Confirmation Score
    volume_confirmation_scores = []
    for i in range(len(df)):
        # Volume momentum alignment (0.5 to 2.0 multiplier)
        vol_alignment = 1.0
        if df['volume_acceleration'].iloc[i] > 1.2 and df['ret_1d'].iloc[i] > 0:
            vol_alignment = min(2.0, 1.0 + df['volume_acceleration'].iloc[i] * 0.5)
        elif df['volume_acceleration'].iloc[i] < 0.8 and df['ret_1d'].iloc[i] < 0:
            vol_alignment = max(0.5, 1.0 - abs(df['volume_acceleration'].iloc[i] - 1.0))
        
        # Price-volume correlation adjustment (-1.0 to 1.0)
        corr_adjustment = df['return_volume_corr'].iloc[i]
        
        # Amount quality factor (0.8 to 1.2 multiplier)
        amount_quality = 1.0 + (df['large_trade_concentration'].iloc[i] - 1.0) * 0.2
        amount_quality = max(0.8, min(1.2, amount_quality))
        
        volume_confirmation = vol_alignment * (1.0 + corr_adjustment) * amount_quality
        volume_confirmation_scores.append(volume_confirmation)
    
    df['volume_confirmation_score'] = volume_confirmation_scores
    
    # Final Adaptive Factor
    final_factors = []
    for i in range(len(df)):
        # Multiply momentum score by volume-price confirmation
        factor = df['momentum_score'].iloc[i] * df['volume_confirmation_score'].iloc[i]
        
        # Apply volatility regime scaling (0.7 to 1.3)
        regime = df['vol_regime'].iloc[i]
        if regime == 'explosive':
            factor *= 1.3
        elif regime == 'high':
            factor *= 1.1
        elif regime == 'low':
            factor *= 0.9
        elif regime == 'compressed':
            factor *= 0.7
        # Normal regime gets 1.0 multiplier
        
        # Preserve directional consistency with ultra-short momentum
        if np.sign(factor) != np.sign(df['ret_1d'].iloc[i]):
            factor *= 0.5  # Reduce magnitude if direction inconsistent
        
        final_factors.append(factor)
    
    result = pd.Series(final_factors, index=df.index, name='adaptive_momentum_factor')
    return result
