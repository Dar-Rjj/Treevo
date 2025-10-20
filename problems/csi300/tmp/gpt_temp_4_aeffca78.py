import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Price-Volume Efficiency Momentum Divergence v2
    A multi-dimensional efficiency analysis factor combining price, volume, and range efficiency
    across multiple timeframes and regimes.
    """
    
    # Initialize result series
    result = pd.Series(index=df.index, dtype=float)
    
    # Calculate basic price and volume metrics
    df['returns'] = df['close'].pct_change()
    df['range'] = (df['high'] - df['low']) / df['close'].shift(1)
    df['volume_change'] = df['volume'].pct_change()
    
    # Efficiency calculations
    df['price_efficiency_5'] = df['returns'].abs().rolling(5).mean()
    df['price_efficiency_21'] = df['returns'].abs().rolling(21).mean()
    df['volume_efficiency_5'] = df['volume_change'].abs().rolling(5).mean()
    df['volume_efficiency_21'] = df['volume_change'].abs().rolling(21).mean()
    df['range_efficiency_5'] = df['range'].rolling(5).mean()
    df['range_efficiency_21'] = df['range'].rolling(21).mean()
    
    # Efficiency ratios for regime classification
    df['efficiency_ratio_5_21'] = df['price_efficiency_5'] / df['price_efficiency_21']
    df['volume_efficiency_ratio'] = df['volume_efficiency_5'] / df['volume_efficiency_21']
    df['range_efficiency_ratio'] = df['range_efficiency_5'] / df['range_efficiency_21']
    
    # Dynamic Efficiency Regime Classification
    df['price_regime'] = np.where(df['efficiency_ratio_5_21'] > 1.2, 2, 
                                 np.where(df['efficiency_ratio_5_21'] < 0.8, 0, 1))
    df['volume_regime'] = np.where(df['volume_efficiency_ratio'] > 1.1, 2,
                                  np.where(df['volume_efficiency_ratio'] < 0.9, 0, 1))
    df['range_regime'] = np.where(df['range_efficiency_ratio'] > 1.15, 2,
                                 np.where(df['range_efficiency_ratio'] < 0.85, 0, 1))
    
    # Regime transition analysis
    df['price_regime_change'] = df['price_regime'].diff().abs()
    df['volume_regime_change'] = df['volume_regime'].diff().abs()
    df['days_since_price_regime_change'] = df.groupby((df['price_regime_change'] != 0).cumsum()).cumcount()
    df['days_since_volume_regime_change'] = df.groupby((df['volume_regime_change'] != 0).cumsum()).cumcount()
    
    # Efficiency momentum calculations
    df['price_efficiency_momentum_5'] = df['price_efficiency_5'].pct_change(3)
    df['volume_efficiency_momentum_5'] = df['volume_efficiency_5'].pct_change(3)
    df['range_efficiency_momentum_5'] = df['range_efficiency_5'].pct_change(3)
    
    # Cross-asset efficiency momentum divergence
    df['price_volume_efficiency_divergence'] = (
        df['price_efficiency_momentum_5'] - df['volume_efficiency_momentum_5']
    )
    df['price_range_efficiency_divergence'] = (
        df['price_efficiency_momentum_5'] - df['range_efficiency_momentum_5']
    )
    
    # Range-adjusted efficiency momentum
    df['range_adjusted_price_momentum'] = (
        df['price_efficiency_momentum_5'] / (df['range_efficiency_5'] + 1e-8)
    )
    df['range_adjusted_volume_momentum'] = (
        df['volume_efficiency_momentum_5'] / (df['range_efficiency_5'] + 1e-8)
    )
    
    # Multi-timeframe efficiency convergence
    df['short_term_efficiency_trend'] = df['price_efficiency_5'].rolling(3).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0], raw=True
    )
    df['medium_term_efficiency_trend'] = df['price_efficiency_21'].rolling(8).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0], raw=True
    )
    df['timeframe_convergence'] = (
        np.sign(df['short_term_efficiency_trend']) == np.sign(df['medium_term_efficiency_trend'])
    ).astype(int)
    
    # Volume concentration analysis
    df['volume_rolling_mean'] = df['volume'].rolling(10).mean()
    df['volume_concentration'] = df['volume'] / df['volume_rolling_mean']
    df['volume_concentration_momentum'] = df['volume_concentration'].pct_change(3)
    
    # Efficiency acceleration patterns
    df['price_efficiency_acceleration'] = df['price_efficiency_momentum_5'].diff()
    df['volume_efficiency_acceleration'] = df['volume_efficiency_momentum_5'].diff()
    
    # Regime-adaptive efficiency signals
    df['regime_strength_price'] = (
        df['efficiency_ratio_5_21'].rolling(5).std() / 
        (df['efficiency_ratio_5_21'].rolling(21).std() + 1e-8)
    )
    df['regime_strength_volume'] = (
        df['volume_efficiency_ratio'].rolling(5).std() / 
        (df['volume_efficiency_ratio'].rolling(21).std() + 1e-8)
    )
    
    # Cross-dimension efficiency alignment
    df['price_volume_alignment'] = (
        (df['price_regime'] == df['volume_regime']).astype(int) * 
        (1 + df['price_volume_efficiency_divergence'].abs())
    )
    df['price_range_alignment'] = (
        (df['price_regime'] == df['range_regime']).astype(int) * 
        (1 + df['price_range_efficiency_divergence'].abs())
    )
    
    # Dynamic factor composition
    # Component 1: Regime-adaptive efficiency momentum
    regime_momentum = (
        df['price_efficiency_momentum_5'] * df['regime_strength_price'] +
        df['volume_efficiency_momentum_5'] * df['regime_strength_volume']
    )
    
    # Component 2: Cross-dimension divergence signals
    divergence_signals = (
        df['price_volume_efficiency_divergence'] * df['price_volume_alignment'] +
        df['price_range_efficiency_divergence'] * df['price_range_alignment']
    )
    
    # Component 3: Multi-timeframe convergence strength
    timeframe_strength = (
        df['timeframe_convergence'] * 
        (df['short_term_efficiency_trend'] + df['medium_term_efficiency_trend'])
    )
    
    # Component 4: Volume concentration momentum
    volume_signals = (
        df['volume_concentration_momentum'] * 
        np.where(df['volume_regime'] == 2, 1.5, 
                np.where(df['volume_regime'] == 0, 0.5, 1.0))
    )
    
    # Component 5: Efficiency acceleration patterns
    acceleration_signals = (
        df['price_efficiency_acceleration'] + 
        df['volume_efficiency_acceleration']
    )
    
    # Final factor composition with regime-dependent weighting
    regime_weights = np.where(df['price_regime'] == 2, 1.2,
                             np.where(df['price_regime'] == 0, 0.8, 1.0))
    
    result = (
        regime_momentum * 0.35 +
        divergence_signals * 0.25 +
        timeframe_strength * 0.20 +
        volume_signals * 0.15 +
        acceleration_signals * 0.05
    ) * regime_weights
    
    # Clean up intermediate columns
    cols_to_drop = [col for col in df.columns if col not in ['open', 'high', 'low', 'close', 'amount', 'volume']]
    df.drop(columns=cols_to_drop, inplace=True, errors='ignore')
    
    return result
