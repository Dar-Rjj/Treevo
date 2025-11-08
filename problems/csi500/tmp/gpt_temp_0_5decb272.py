import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(data):
    """
    Regime-Adaptive Momentum-Volume Divergence Alpha Factor
    
    This factor combines price and volume momentum across multiple timeframes,
    applies exponential smoothing, detects market regimes, and creates an adaptive
    signal that adjusts to volatility and participation conditions.
    """
    df = data.copy()
    
    # Momentum Components
    # Price Momentum
    df['price_momentum_5d'] = df['close'] / df['close'].shift(5) - 1
    df['price_momentum_10d'] = df['close'] / df['close'].shift(10) - 1
    df['price_momentum_20d'] = df['close'] / df['close'].shift(20) - 1
    
    # Volume Momentum
    df['volume_momentum_5d'] = df['volume'] / df['volume'].shift(5) - 1
    df['volume_momentum_10d'] = df['volume'] / df['volume'].shift(10) - 1
    df['volume_momentum_20d'] = df['volume'] / df['volume'].shift(20) - 1
    
    # Regime Detection
    # Amount-Based Regime
    df['amount_momentum_5d'] = df['amount'] / df['amount'].shift(5) - 1
    df['amount_momentum_10d'] = df['amount'] / df['amount'].shift(10) - 1
    df['amount_acceleration'] = df['amount_momentum_5d'] - (df['amount'].shift(5) / df['amount'].shift(10) - 1)
    df['regime_strength'] = np.abs(df['amount_acceleration']) * np.abs(df['amount_momentum_5d'])
    
    # Volatility Regime
    df['daily_range'] = (df['high'] - df['low']) / df['close']
    df['avg_range_5d'] = df['daily_range'].rolling(window=5).mean()
    df['volatility_regime'] = df['daily_range'] / df['avg_range_5d']
    
    # Exponential Smoothing
    alpha = 0.3  # Smoothing factor
    
    # Price Momentum Smoothing
    df['ema_5d_price'] = df['price_momentum_5d'].ewm(alpha=alpha, adjust=False).mean()
    df['ema_10d_price'] = df['price_momentum_10d'].ewm(alpha=alpha, adjust=False).mean()
    df['ema_20d_price'] = df['price_momentum_20d'].ewm(alpha=alpha, adjust=False).mean()
    
    # Volume Momentum Smoothing
    df['ema_5d_volume'] = df['volume_momentum_5d'].ewm(alpha=alpha, adjust=False).mean()
    df['ema_10d_volume'] = df['volume_momentum_10d'].ewm(alpha=alpha, adjust=False).mean()
    df['ema_20d_volume'] = df['volume_momentum_20d'].ewm(alpha=alpha, adjust=False).mean()
    
    # Momentum-Volume Divergence
    df['divergence_5d'] = df['ema_5d_price'] - df['ema_5d_volume']
    df['divergence_10d'] = df['ema_10d_price'] - df['ema_10d_volume']
    df['divergence_20d'] = df['ema_20d_price'] - df['ema_20d_volume']
    
    # Adaptive Signal Combination
    def get_adaptive_signal(row):
        # Volatility-based weighting
        if row['volatility_regime'] > 1.2:  # High volatility
            weighted_divergence = (0.6 * row['divergence_5d'] + 
                                  0.4 * row['divergence_10d'])
        elif row['volatility_regime'] < 0.8:  # Low volatility
            weighted_divergence = (0.3 * row['divergence_5d'] + 
                                  0.7 * row['divergence_20d'])
        else:  # Normal volatility
            weighted_divergence = (row['divergence_5d'] + 
                                  row['divergence_10d'] + 
                                  row['divergence_20d']) / 3
        
        # Amount-regime adjustment
        signal = weighted_divergence * (1 + row['amount_momentum_5d'])
        return signal
    
    df['adaptive_signal'] = df.apply(get_adaptive_signal, axis=1)
    
    # Cross-sectional ranking with regime adjustment
    def get_regime_adjusted_rank(group):
        volatility_regime = group['volatility_regime'].iloc[-1] if not group.empty else 1.0
        
        if volatility_regime > 1.2:  # High volatility
            rank_base = group['divergence_5d'].rank(pct=True)
        elif volatility_regime < 0.8:  # Low volatility
            rank_base = group['divergence_20d'].rank(pct=True)
        else:  # Transition regime
            rank_base = group['divergence_10d'].rank(pct=True)
        
        # Amount-confidence adjustment
        regime_strength = group['regime_strength'].iloc[-1] if not group.empty else 0.0
        adjusted_rank = rank_base * (1 + regime_strength)
        
        return adjusted_rank
    
    # Apply cross-sectional ranking by date
    final_factor = pd.Series(index=df.index, dtype=float)
    
    for date in df.index:
        daily_data = df.loc[date]
        if isinstance(daily_data, pd.Series):
            # Single stock case
            final_factor.loc[date] = daily_data['adaptive_signal']
        else:
            # Multiple stocks case - cross-sectional ranking
            daily_group = df.loc[[date]]
            ranks = get_regime_adjusted_rank(daily_group)
            final_factor.loc[date] = ranks.iloc[0] if len(ranks) == 1 else ranks
    
    return final_factor
