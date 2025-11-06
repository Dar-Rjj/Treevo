import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Dynamic Momentum Regime Alpha factor that combines intraday momentum structure,
    volume confirmation, regime detection, and momentum quality scoring.
    """
    
    # Initialize result series
    result = pd.Series(index=df.index, dtype=float)
    
    # Calculate basic price components
    df['prev_close'] = df['close'].shift(1)
    df['gap_size'] = df['open'] / df['prev_close'] - 1
    df['intraday_range'] = df['high'] - df['low']
    df['open_to_close'] = df['close'] - df['open']
    df['price_change'] = df['close'] / df['prev_close'] - 1
    
    # Volume components
    df['volume_ma_5'] = df['volume'].rolling(window=5).mean()
    df['relative_volume'] = df['volume'] / df['volume_ma_5']
    df['volume_acceleration'] = df['volume'] / df['volume'].shift(1) - 1
    
    # 1. Intraday Momentum Structure
    # Opening Gap Momentum
    df['gap_fill_percentage'] = np.where(
        df['gap_size'] > 0,
        (df['high'] - df['open']) / (df['high'] - df['low']),
        (df['open'] - df['low']) / (df['high'] - df['low'])
    )
    
    # Gap persistence (1 if gap maintained, 0 if filled)
    df['gap_persistence'] = np.where(
        (df['gap_size'] > 0) & (df['close'] > df['open']), 1,
        np.where((df['gap_size'] < 0) & (df['close'] < df['open']), 1, 0)
    )
    
    # Intraday Range Efficiency
    df['range_efficiency'] = np.where(
        df['open_to_close'] != 0,
        abs(df['open_to_close']) / df['intraday_range'],
        0.5  # Neutral value when no movement
    )
    
    # Directional consistency
    df['directional_consistency'] = np.sign(df['open_to_close']) * df['range_efficiency']
    
    # Closing momentum strength (last hour proxy)
    df['closing_strength'] = (df['close'] - (df['high'] + df['low']) / 2) / df['intraday_range']
    
    # 2. Volume Confirmation Framework
    df['volume_price_sync'] = np.sign(df['price_change']) * df['relative_volume']
    df['volume_efficiency'] = df['volume'] / (abs(df['price_change']) * df['close'] + 1e-6)
    
    # 3. Regime Detection & Adaptation
    # Volatility regime
    df['range_volatility'] = df['intraday_range'] / df['close']
    df['volatility_ma_10'] = df['range_volatility'].rolling(window=10).mean()
    df['volatility_regime'] = df['range_volatility'] / df['volatility_ma_10']
    
    # Trend regime
    df['short_trend'] = df['close'].rolling(window=5).mean() / df['close'].rolling(window=10).mean() - 1
    df['medium_trend'] = df['close'].rolling(window=10).mean() / df['close'].rolling(window=20).mean() - 1
    df['trend_alignment'] = np.sign(df['short_trend']) * np.sign(df['medium_trend'])
    df['trend_strength'] = (abs(df['short_trend']) + abs(df['medium_trend'])) / 2
    
    # 4. Momentum Quality Scoring
    # Momentum persistence
    df['momentum_direction'] = np.sign(df['price_change'])
    df['consecutive_days'] = 0
    for i in range(1, len(df)):
        if df['momentum_direction'].iloc[i] == df['momentum_direction'].iloc[i-1]:
            df.loc[df.index[i], 'consecutive_days'] = df['consecutive_days'].iloc[i-1] + 1
    
    # Momentum decay detection
    df['momentum_acceleration'] = df['price_change'] - df['price_change'].shift(1)
    
    # 5. Signal Confirmation Hierarchy
    # Primary confirmation
    df['volume_price_alignment'] = np.sign(df['price_change']) * np.sign(df['volume_price_sync'])
    df['intraday_consistency'] = np.sign(df['open_to_close']) * np.sign(df['directional_consistency'])
    
    # 6. Dynamic Weight Assignment and Signal Generation
    for i in range(10, len(df)):
        current_data = df.iloc[i]
        
        # Regime-based weights
        volatility_weight = 1.0 / (1.0 + abs(current_data['volatility_regime'] - 1))
        trend_weight = 1.0 if current_data['trend_alignment'] > 0 else 0.7
        
        # Momentum quality weights
        persistence_weight = min(current_data['consecutive_days'] / 5.0, 2.0)  # Cap at 2x
        confirmation_weight = (current_data['volume_price_alignment'] + 
                             current_data['intraday_consistency'] + 2) / 4  # Scale 0-1
        
        # Component scores
        gap_momentum_score = current_data['gap_size'] * current_data['gap_persistence']
        range_efficiency_score = current_data['directional_consistency']
        closing_momentum_score = current_data['closing_strength']
        volume_confirmation_score = current_data['volume_price_sync']
        
        # Weighted combination
        base_signal = (
            gap_momentum_score * 0.2 +
            range_efficiency_score * 0.3 +
            closing_momentum_score * 0.25 +
            volume_confirmation_score * 0.25
        )
        
        # Apply dynamic weights
        regime_adjusted = base_signal * volatility_weight * trend_weight
        quality_adjusted = regime_adjusted * persistence_weight * confirmation_weight
        
        # Volatility scaling
        volatility_scaled = quality_adjusted / (current_data['volatility_ma_10'] + 0.01)
        
        result.iloc[i] = volatility_scaled
    
    # Fill NaN values
    result = result.fillna(0)
    
    return result
