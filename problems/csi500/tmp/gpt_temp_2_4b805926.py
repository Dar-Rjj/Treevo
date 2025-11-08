import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(data):
    """
    Multi-Timeframe Volume-Persistence Momentum with Volatility Adjustment
    """
    df = data.copy()
    
    # Multi-Timeframe Momentum Calculation
    # Short-Term (1-3 days)
    df['st_price_change'] = df['close'] - df['close'].shift(2)
    df['st_volatility'] = df['high'].rolling(window=3).max() - df['low'].rolling(window=3).min()
    df['st_momentum'] = df['st_price_change'] / (df['st_volatility'] + 1e-8)
    
    # Medium-Term (5-10 days)
    df['mt_price_change'] = df['close'] - df['close'].shift(5)
    df['mt_volatility'] = df['high'].rolling(window=6).max() - df['low'].rolling(window=6).min()
    df['mt_momentum'] = df['mt_price_change'] / (df['mt_volatility'] + 1e-8)
    
    # Long-Term (20-30 days)
    df['lt_price_change'] = df['close'] - df['close'].shift(20)
    df['lt_volatility'] = df['high'].rolling(window=11).max() - df['low'].rolling(window=11).min()
    df['lt_momentum'] = df['lt_price_change'] / (df['lt_volatility'] + 1e-8)
    
    # Momentum Persistence Analysis
    # Direction persistence tracking
    df['st_direction'] = np.sign(df['st_price_change'])
    df['mt_direction'] = np.sign(df['mt_price_change'])
    df['lt_direction'] = np.sign(df['lt_price_change'])
    
    # Count consecutive days with same direction
    def count_consecutive_direction(series):
        count = pd.Series(index=series.index, dtype=float)
        current_count = 0
        for i in range(len(series)):
            if i == 0 or series.iloc[i] == series.iloc[i-1]:
                current_count += 1
            else:
                current_count = 1
            count.iloc[i] = current_count
        return count
    
    df['st_persistence'] = count_consecutive_direction(df['st_direction'])
    df['mt_persistence'] = count_consecutive_direction(df['mt_direction'])
    df['lt_persistence'] = count_consecutive_direction(df['lt_direction'])
    
    # Persistence-weighted momentum
    df['st_weighted_momentum'] = df['st_momentum'] * df['st_persistence']
    df['mt_weighted_momentum'] = df['mt_momentum'] * df['mt_persistence']
    df['lt_weighted_momentum'] = df['lt_momentum'] * df['lt_persistence']
    
    # Volume Persistence Confirmation
    df['volume_direction'] = np.sign(df['volume'] - df['volume'].shift(1))
    df['volume_persistence'] = count_consecutive_direction(df['volume_direction'])
    df['volume_momentum_strength'] = abs(df['volume'] - df['volume'].shift(1)) / (df['volume'].shift(1) + 1e-8)
    
    # Volume-price alignment
    df['st_alignment'] = np.sign(df['st_momentum']) * df['volume_direction']
    df['mt_alignment'] = np.sign(df['mt_momentum']) * df['volume_direction']
    df['lt_alignment'] = np.sign(df['lt_momentum']) * df['volume_direction']
    df['alignment_strength'] = df['volume_momentum_strength'] * df['volume_persistence']
    
    # Volatility-Adjusted Weighting
    # Recent volatility proxy (5-day ATR)
    df['recent_atr'] = (df['high'] - df['low']).rolling(window=5).mean()
    df['long_term_atr'] = (df['high'] - df['low']).rolling(window=20).mean()
    
    # Volatility regime detection
    def get_regime_weights(recent_atr, long_term_atr):
        if recent_atr > long_term_atr * 1.1:  # High volatility
            return 0.3, 0.4, 0.3
        elif recent_atr < long_term_atr * 0.9:  # Low volatility
            return 0.5, 0.3, 0.2
        else:  # Normal volatility
            return 0.4, 0.35, 0.25
    
    regime_weights = df.apply(
        lambda row: get_regime_weights(row['recent_atr'], row['long_term_atr']), 
        axis=1, result_type='expand'
    )
    regime_weights.columns = ['st_weight', 'mt_weight', 'lt_weight']
    df = pd.concat([df, regime_weights], axis=1)
    
    # Final Factor Integration
    # Weighted momentum combination
    df['st_component'] = df['st_weighted_momentum'] * df['st_alignment'] * df['st_weight']
    df['mt_component'] = df['mt_weighted_momentum'] * df['mt_alignment'] * df['mt_weight']
    df['lt_component'] = df['lt_weighted_momentum'] * df['lt_alignment'] * df['lt_weight']
    df['combined_momentum'] = df['st_component'] + df['mt_component'] + df['lt_component']
    
    # Volume persistence enhancement
    df['volume_confirmation'] = 1 + df['alignment_strength']
    df['enhanced_signal'] = df['combined_momentum'] * df['volume_confirmation']
    
    # Volatility normalization
    df['final_alpha'] = df['enhanced_signal'] / (df['recent_atr'] + 1e-8)
    
    return df['final_alpha']
