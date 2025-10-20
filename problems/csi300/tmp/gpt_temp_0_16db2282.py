import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Price-Volume Asymmetry with Regime-Dependent Persistence alpha factor
    """
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Initialize result series
    alpha = pd.Series(index=df.index, dtype=float)
    
    # Calculate basic technical indicators
    df['returns'] = df['close'].pct_change()
    df['price_range'] = df['high'] - df['low']
    df['price_efficiency'] = (df['close'] - df['open']) / (df['price_range'].replace(0, np.nan))
    df['price_efficiency'] = df['price_efficiency'].fillna(0)
    
    # ATR calculation
    df['tr'] = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            abs(df['high'] - df['close'].shift(1)),
            abs(df['low'] - df['close'].shift(1))
        )
    )
    df['atr_5'] = df['tr'].rolling(window=5).mean()
    
    # Directional Volume-Price Efficiency Analysis
    up_days = df['close'] > df['open']
    down_days = df['close'] < df['open']
    
    # Up-day efficiency
    df['up_efficiency'] = np.where(
        up_days & (df['volume'] > 0),
        (df['close'] - df['open']) / df['volume'],
        0
    )
    
    # Down-day efficiency
    df['down_efficiency'] = np.where(
        down_days & (df['volume'] > 0),
        (df['open'] - df['close']) / df['volume'],
        0
    )
    
    # Directional efficiency ratio
    df['efficiency_ratio'] = df['up_efficiency'].rolling(window=10).mean() / \
                            (df['down_efficiency'].rolling(window=10).mean() + 1e-8)
    
    # Volume concentration in high-impact periods
    high_impact = abs(df['close'] - df['open']) > 0.5 * df['atr_5']
    df['high_impact_volume_ratio'] = np.where(
        high_impact,
        df['volume'],
        0
    ).rolling(window=10).sum() / (df['volume'].rolling(window=10).sum() + 1e-8)
    
    # Volume acceleration divergence
    df['up_volume_momentum'] = np.where(
        up_days,
        df['volume'].pct_change(periods=5),
        0
    ).rolling(window=5).mean()
    
    df['down_volume_momentum'] = np.where(
        down_days,
        df['volume'].pct_change(periods=5),
        0
    ).rolling(window=5).mean()
    
    df['volume_acceleration_div'] = df['up_volume_momentum'] - df['down_volume_momentum']
    
    # Price-Level Dependent Volume Behavior
    # Recent highs and lows
    df['recent_high'] = df['high'].rolling(window=20).max()
    df['recent_low'] = df['low'].rolling(window=20).min()
    
    # Volume intensity near highs/lows
    near_high = (df['close'] > 0.95 * df['recent_high']) & (df['close'] <= df['recent_high'])
    near_low = (df['close'] < 1.05 * df['recent_low']) & (df['close'] >= df['recent_low'])
    
    df['volume_intensity_high'] = np.where(
        near_high,
        df['volume'] / df['volume'].rolling(window=20).mean(),
        1
    )
    
    df['volume_intensity_low'] = np.where(
        near_low,
        df['volume'] / df['volume'].rolling(window=20).mean(),
        1
    )
    
    # Multi-Scale Regime Classification
    # Volatility-Liquidity Regime
    df['volatility_volume_ratio'] = df['returns'].rolling(window=10).std() / \
                                   (df['volume'].pct_change().rolling(window=10).std() + 1e-8)
    
    df['volatility_regime'] = np.select(
        [
            df['volatility_volume_ratio'] > 1.5,
            df['volatility_volume_ratio'] < 0.5
        ],
        ['high', 'low'],
        default='normal'
    )
    
    # Trend-Structure Regime
    df['trend_regime'] = np.select(
        [
            df['price_efficiency'].rolling(window=5).mean() > 0.6,
            df['price_efficiency'].rolling(window=5).mean() < 0.4
        ],
        ['trending', 'choppy'],
        default='normal'
    )
    
    # Volume-Distribution Regime
    df['volume_skew'] = df['volume'].rolling(window=20).skew()
    df['volume_kurtosis'] = df['volume'].rolling(window=20).kurtosis()
    
    df['volume_regime'] = np.select(
        [
            df['volume_kurtosis'] > 2,
            df['volume_kurtosis'] < -1
        ],
        ['concentrated', 'diffuse'],
        default='normal'
    )
    
    # Regime-Adaptive Asymmetry Measurement
    regime_weights = pd.Series(1.0, index=df.index)
    
    # High volatility regime: emphasize directional efficiency
    high_vol_mask = df['volatility_regime'] == 'high'
    regime_weights[high_vol_mask] *= 1.5
    
    # Trending markets: emphasize volume exhaustion
    trending_mask = df['trend_regime'] == 'trending'
    regime_weights[trending_mask] *= 1.3
    
    # Concentrated volume: emphasize volume-level interactions
    concentrated_mask = df['volume_regime'] == 'concentrated'
    regime_weights[concentrated_mask] *= 1.2
    
    # Choppy markets: emphasize volume-price elasticity
    choppy_mask = df['trend_regime'] == 'choppy'
    regime_weights[choppy_mask] *= 1.4
    
    # Temporal Pattern Integration
    df['day_of_week'] = df.index.dayofweek
    
    # Day-of-week directional efficiency
    weekday_efficiency = []
    for i in range(len(df)):
        if i >= 20:
            recent_data = df.iloc[i-20:i]
            weekday_eff = recent_data.groupby('day_of_week')['efficiency_ratio'].mean()
            current_weekday = df.iloc[i]['day_of_week']
            weekday_efficiency.append(weekday_eff.get(current_weekday, 1.0))
        else:
            weekday_efficiency.append(1.0)
    
    df['weekday_efficiency'] = weekday_efficiency
    
    # Structural Break Detection
    df['efficiency_ma'] = df['efficiency_ratio'].rolling(window=10).mean()
    df['efficiency_std'] = df['efficiency_ratio'].rolling(window=20).std()
    
    structural_break = abs(df['efficiency_ratio'] - df['efficiency_ma']) > 2 * df['efficiency_std']
    df['structural_break_signal'] = structural_break.astype(int)
    
    # Composite Alpha Construction
    for i in range(len(df)):
        if i < 20:  # Ensure enough data for calculations
            alpha.iloc[i] = 0
            continue
            
        current_data = df.iloc[i]
        
        # Base components
        directional_component = current_data['efficiency_ratio']
        concentration_component = current_data['high_impact_volume_ratio']
        acceleration_component = current_data['volume_acceleration_div']
        level_component = (current_data['volume_intensity_high'] + 
                          current_data['volume_intensity_low']) / 2
        
        # Temporal adjustments
        temporal_adjustment = current_data['weekday_efficiency']
        
        # Structural break adjustment
        break_adjustment = 1 + 0.5 * current_data['structural_break_signal']
        
        # Combine components with regime weights
        composite = (
            directional_component * 0.3 +
            concentration_component * 0.25 +
            acceleration_component * 0.2 +
            level_component * 0.25
        ) * temporal_adjustment * break_adjustment * regime_weights.iloc[i]
        
        alpha.iloc[i] = composite
    
    # Normalize and handle edge cases
    alpha = alpha.replace([np.inf, -np.inf], 0).fillna(0)
    
    # Apply persistence weighting (exponential decay)
    decay_factor = 0.9
    persistence_weighted = alpha.copy()
    for i in range(1, len(persistence_weighted)):
        persistence_weighted.iloc[i] = (
            decay_factor * persistence_weighted.iloc[i-1] + 
            (1 - decay_factor) * alpha.iloc[i]
        )
    
    return persistence_weighted
