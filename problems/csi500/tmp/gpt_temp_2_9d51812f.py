import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(data):
    """
    Regime-Aware Momentum-Volume Divergence Alpha Factor
    """
    df = data.copy()
    
    # Momentum-Volume Divergence Core
    # Price Momentum Calculation
    df['price_momentum_5d'] = df['close'] / df['close'].shift(5) - 1
    df['price_momentum_10d'] = df['close'] / df['close'].shift(10) - 1
    df['price_momentum_20d'] = df['close'] / df['close'].shift(20) - 1
    
    # Volume Momentum Calculation
    df['volume_momentum_5d'] = df['volume'] / df['volume'].shift(5) - 1
    df['volume_momentum_10d'] = df['volume'] / df['volume'].shift(10) - 1
    df['volume_momentum_20d'] = df['volume'] / df['volume'].shift(20) - 1
    
    # Regime Detection System
    # Amount-Based Regime Classification
    df['amount_20d_ma'] = df['amount'].rolling(window=20).mean()
    df['amount_acceleration'] = (df['amount'] / df['amount'].shift(5)) - (df['amount'].shift(5) / df['amount'].shift(10))
    
    # Volatility Assessment
    df['price_range_20d'] = (df['high'] - df['low']) / df['close']
    df['volatility_regime'] = df['price_range_20d'].rolling(window=20).apply(
        lambda x: 2 if x.iloc[-1] > np.percentile(x, 75) else (0 if x.iloc[-1] < np.percentile(x, 25) else 1), 
        raw=False
    )
    
    # Exponential Smoothing Application
    alpha = 0.3
    momentum_columns = ['price_momentum_5d', 'price_momentum_10d', 'price_momentum_20d', 
                       'volume_momentum_5d', 'volume_momentum_10d', 'volume_momentum_20d']
    
    for col in momentum_columns:
        df[f'ema_{col}'] = df[col].ewm(alpha=alpha).mean()
    
    # Calculate momentum acceleration
    for col in momentum_columns:
        df[f'acceleration_{col}'] = df[f'ema_{col}'] - df[f'ema_{col}'].shift(1)
    
    # Cross-Sectional Processing
    # Dynamic Ranking System
    price_momentum_cols = ['ema_price_momentum_5d', 'ema_price_momentum_10d', 'ema_price_momentum_20d']
    volume_momentum_cols = ['ema_volume_momentum_5d', 'ema_volume_momentum_10d', 'ema_volume_momentum_20d']
    
    df['combined_price_momentum'] = df[price_momentum_cols].mean(axis=1)
    df['combined_volume_momentum'] = df[volume_momentum_cols].mean(axis=1)
    
    # Calculate momentum divergence
    df['momentum_divergence'] = df['combined_price_momentum'] - df['combined_volume_momentum']
    
    # Volatility-Normalized Ranking
    df['momentum_divergence_rank'] = df['momentum_divergence'].rolling(window=20).rank(pct=True)
    df['volatility_normalized_rank'] = df['momentum_divergence_rank'] / (df['price_range_20d'] + 1e-8)
    
    # Regime-Adaptive Combination
    # Divergence Signal Generation
    df['divergence_signal'] = np.where(
        df['combined_price_momentum'] > df['combined_volume_momentum'], 1,
        np.where(df['combined_price_momentum'] < df['combined_volume_momentum'], -1, 0)
    )
    
    # Regime-Weighted Integration
    def calculate_regime_weight(volatility_regime, amount_acceleration):
        if volatility_regime == 2:  # High volatility
            return 0.7  # Emphasize volume confirmation
        elif volatility_regime == 0:  # Low volatility
            return 0.3  # Emphasize price momentum
        else:  # Transition regimes
            return 0.5  # Balanced weighting
    
    df['regime_weight'] = df.apply(
        lambda x: calculate_regime_weight(x['volatility_regime'], x['amount_acceleration']), 
        axis=1
    )
    
    # Final Factor Construction
    # Multi-Timeframe Aggregation
    timeframe_weights = {'5d': 0.4, '10d': 0.35, '20d': 0.25}
    
    df['weighted_divergence'] = (
        timeframe_weights['5d'] * (df['ema_price_momentum_5d'] - df['ema_volume_momentum_5d']) +
        timeframe_weights['10d'] * (df['ema_price_momentum_10d'] - df['ema_volume_momentum_10d']) +
        timeframe_weights['20d'] * (df['ema_price_momentum_20d'] - df['ema_volume_momentum_20d'])
    )
    
    # Apply regime weighting
    df['regime_adjusted_divergence'] = (
        df['regime_weight'] * df['weighted_divergence'] + 
        (1 - df['regime_weight']) * df['volatility_normalized_rank']
    )
    
    # Adaptive Scaling
    df['amount_regime_strength'] = df['amount_acceleration'].abs()
    df['scaled_factor'] = (
        df['regime_adjusted_divergence'] * 
        (1 + df['amount_regime_strength']) / 
        (1 + df['price_range_20d'])
    )
    
    # Final exponential smoothing
    final_factor = df['scaled_factor'].ewm(alpha=0.2).mean()
    
    return final_factor
