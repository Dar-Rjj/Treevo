import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Volatility Regime Detection
    data['daily_range'] = (data['high'] - data['low']) / data['close']
    data['avg_5day_range'] = data['daily_range'].rolling(window=5).mean()
    data['high_vol_regime'] = data['daily_range'] > data['avg_5day_range']
    
    # Multi-Timeframe Momentum Analysis - Price
    data['price_momentum_5d'] = data['close'] / data['close'].shift(5) - 1
    data['price_momentum_10d'] = data['close'] / data['close'].shift(10) - 1
    data['price_momentum_20d'] = data['close'] / data['close'].shift(20) - 1
    
    # Multi-Timeframe Momentum Analysis - Volume
    data['volume_change_5d'] = data['volume'] / data['volume'].shift(5) - 1
    data['volume_change_10d'] = data['volume'] / data['volume'].shift(10) - 1
    data['volume_change_20d'] = data['volume'] / data['volume'].shift(20) - 1
    
    # Convergence Quality Assessment - Price Momentum Consistency
    price_signs = pd.DataFrame({
        'short': np.sign(data['price_momentum_5d']),
        'medium': np.sign(data['price_momentum_10d']),
        'long': np.sign(data['price_momentum_20d'])
    })
    data['price_consistency'] = price_signs.apply(
        lambda x: sum(x == x.iloc[0]) if not x.isna().any() else 0, axis=1
    )
    
    # Convergence Quality Assessment - Volume Momentum Consistency
    volume_signs = pd.DataFrame({
        'short': np.sign(data['volume_change_5d']),
        'medium': np.sign(data['volume_change_10d']),
        'long': np.sign(data['volume_change_20d'])
    })
    data['volume_consistency'] = volume_signs.apply(
        lambda x: sum(x == x.iloc[0]) if not x.isna().any() else 0, axis=1
    )
    
    # Momentum-Volume Alignment
    alignment_scores = []
    for i in range(len(data)):
        if pd.isna(data['price_momentum_5d'].iloc[i]) or pd.isna(data['volume_change_5d'].iloc[i]):
            alignment_scores.append(0)
            continue
            
        price_signs = [
            np.sign(data['price_momentum_5d'].iloc[i]),
            np.sign(data['price_momentum_10d'].iloc[i]),
            np.sign(data['price_momentum_20d'].iloc[i])
        ]
        volume_signs = [
            np.sign(data['volume_change_5d'].iloc[i]),
            np.sign(data['volume_change_10d'].iloc[i]),
            np.sign(data['volume_change_20d'].iloc[i])
        ]
        
        alignment_count = sum(p == v for p, v in zip(price_signs, volume_signs) 
                            if not (pd.isna(p) or pd.isna(v)))
        alignment_scores.append(alignment_count / 3.0)
    
    data['momentum_volume_alignment'] = alignment_scores
    
    # Intraday Momentum Integration
    data['intraday_move'] = data['close'] - data['open']
    data['intraday_range'] = data['high'] - data['low']
    data['intraday_momentum_quality'] = np.where(
        data['intraday_range'] != 0,
        data['intraday_move'] / data['intraday_range'],
        0
    )
    
    # Base Convergence Factor
    data['price_convergence'] = (
        data['price_momentum_5d'] * 
        data['price_momentum_10d'] * 
        data['price_momentum_20d']
    )
    data['volume_convergence'] = (
        data['volume_change_5d'] * 
        data['volume_change_10d'] * 
        data['volume_change_20d']
    )
    
    # Direction Alignment Multiplier
    price_converging = (data['price_consistency'] >= 2)
    volume_converging = (data['volume_consistency'] >= 2)
    
    alignment_multiplier = np.where(
        price_converging & volume_converging, 2.0,
        np.where(price_converging | volume_converging, 1.0, 0.3)
    )
    
    # Quality-Adjusted Enhancement
    price_consistency_multiplier = data['price_consistency'] / 3.0
    volume_consistency_multiplier = data['volume_consistency'] / 3.0
    
    # Volatility-Regime Optimization
    base_factor = data['price_convergence'] * alignment_multiplier
    
    # High Volatility Mode
    high_vol_mask = data['high_vol_regime']
    high_vol_factor = (
        base_factor * 
        (1 + 0.5 * data['volume_convergence']) *  # Emphasize volume confirmation
        (1 + 0.3 * abs(data['intraday_momentum_quality'])) *  # Weight intraday momentum
        (1 + 0.2 * data['momentum_volume_alignment'])  # Stronger convergence
    )
    
    # Low Volatility Mode
    low_vol_mask = ~data['high_vol_regime']
    low_vol_factor = (
        base_factor * 
        (1 + 0.2 * data['price_consistency']) *  # Focus on price momentum consistency
        (1 + 0.1 * data['momentum_volume_alignment']) *  # Moderate convergence
        (1 + 0.3 * price_consistency_multiplier)  # Emphasize multi-timeframe alignment
    )
    
    # Final Factor Generation
    final_factor = np.where(
        high_vol_mask,
        high_vol_factor,
        low_vol_factor
    )
    
    # Apply consistency multipliers
    final_factor = (
        final_factor * 
        (1 + 0.2 * price_consistency_multiplier) *
        (1 + 0.15 * volume_consistency_multiplier)
    )
    
    # Signal Classification - Quality Adjustment
    high_quality_mask = (
        (data['price_consistency'] >= 2) &
        (data['volume_consistency'] >= 2) &
        (data['momentum_volume_alignment'] >= 0.67) &
        (abs(data['intraday_momentum_quality']) >= 0.3)
    )
    
    low_quality_mask = (
        (data['price_consistency'] <= 1) |
        (data['volume_consistency'] <= 1) |
        (data['momentum_volume_alignment'] <= 0.33)
    )
    
    # Apply quality adjustments
    final_factor = np.where(
        high_quality_mask,
        final_factor * 1.5,  # Boost high-quality signals
        np.where(
            low_quality_mask,
            final_factor * 0.5,  # Reduce low-quality signals
            final_factor  # Keep moderate quality signals unchanged
        )
    )
    
    return pd.Series(final_factor, index=data.index)
