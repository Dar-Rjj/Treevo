import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Core Price Momentum Components
    df['intraday_momentum'] = df['close'] - df['open']
    df['price_range'] = df['high'] - df['low']
    df['price_position'] = (df['close'] - df['low']) / (df['price_range'] + 1e-8)
    
    # Volume-Price Integration
    df['volume_adj_momentum'] = df['intraday_momentum'] * df['volume']
    df['volume_direction'] = np.sign(df['volume'] - df['volume'].shift(1))
    df['volume_price_alignment'] = np.sign(df['intraday_momentum']) * df['volume_direction']
    
    # Multi-Timeframe Regime Classification
    df['short_term_vol'] = df['price_range'].rolling(window=5).mean()
    df['medium_term_vol'] = df['price_range'].rolling(window=10).mean()
    df['volatility_ratio'] = df['short_term_vol'] / (df['medium_term_vol'] + 1e-8)
    
    df['short_term_volume'] = df['volume'].rolling(window=5).mean()
    df['medium_term_volume'] = df['volume'].rolling(window=10).mean()
    df['volume_ratio'] = df['short_term_volume'] / (df['medium_term_volume'] + 1e-8)
    
    df['short_term_momentum'] = df['intraday_momentum'].rolling(window=5).sum()
    df['medium_term_momentum'] = df['intraday_momentum'].rolling(window=10).sum()
    df['momentum_ratio'] = df['short_term_momentum'] / (df['medium_term_momentum'] + 1e-8)
    
    # Persistence Tracking
    def calculate_persistence(series, condition_func):
        persistence = pd.Series(index=series.index, dtype=float)
        current_count = 0
        for i in range(len(series)):
            if condition_func(series.iloc[i]):
                current_count += 1
            else:
                current_count = 0
            persistence.iloc[i] = current_count
        return persistence
    
    # Momentum Persistence
    df['momentum_persistence'] = calculate_persistence(
        df['intraday_momentum'], 
        lambda x: np.sign(x) == np.sign(df['intraday_momentum'].shift(1).iloc[-1]) if not pd.isna(df['intraday_momentum'].shift(1).iloc[-1]) else False
    )
    
    # Volume-Price Alignment Persistence
    df['alignment_persistence'] = calculate_persistence(
        df['volume_price_alignment'],
        lambda x: x > 0
    )
    
    # Range Persistence
    df['range_persistence'] = calculate_persistence(
        df['price_range'],
        lambda x: x > df['price_range'].rolling(window=10).mean().iloc[-1] if not pd.isna(df['price_range'].rolling(window=10).mean().iloc[-1]) else False
    )
    
    # Adaptive Factor Construction
    # Base Signal Generation
    df['core_momentum'] = df['volume_adj_momentum'] * df['momentum_persistence']
    df['alignment_enhancement'] = df['core_momentum'] * (1 + df['alignment_persistence'] / 5)
    df['range_adjustment'] = df['alignment_enhancement'] * (1 + df['range_persistence'] / 10)
    
    # Multi-Regime Weighting
    df['volatility_weight'] = np.where(df['volatility_ratio'] < 0.8, 1.3, 
                                     np.where(df['volatility_ratio'] > 1.2, 0.7, 1.0))
    df['volume_weight'] = np.where(df['volume_ratio'] > 1.1, 1.2,
                                 np.where(df['volume_ratio'] < 0.9, 0.8, 1.0))
    df['momentum_weight'] = np.where(df['momentum_ratio'] > 1.2, 1.4,
                                   np.where(df['momentum_ratio'] < 0.8, 0.6, 1.0))
    
    # Final Factor Integration
    df['regime_weighted_signal'] = df['range_adjustment'] * df['volatility_weight'] * df['volume_weight'] * df['momentum_weight']
    
    # Recent Momentum Confirmation
    df['recent_momentum'] = df['intraday_momentum'].rolling(window=3).sum()
    df['final_factor'] = df['regime_weighted_signal'] * np.sign(df['recent_momentum'])
    
    return df['final_factor']
