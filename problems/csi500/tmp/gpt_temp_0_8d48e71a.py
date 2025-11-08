import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Regime-Transition Momentum-Volume Divergence Alpha Factor
    """
    # Initialize result series
    alpha = pd.Series(index=df.index, dtype=float)
    
    # Price Momentum Derivatives
    df['price_momentum_5'] = df['close'] / df['close'].shift(5) - 1
    df['price_momentum_10'] = df['close'] / df['close'].shift(10) - 1
    df['price_momentum_20'] = df['close'] / df['close'].shift(20) - 1
    
    # Volume Momentum Derivatives
    df['volume_momentum_5'] = df['volume'] / df['volume'].shift(5) - 1
    df['volume_momentum_10'] = df['volume'] / df['volume'].shift(10) - 1
    df['volume_momentum_20'] = df['volume'] / df['volume'].shift(20) - 1
    
    # Momentum-Volume Divergence Calculation
    df['divergence_5'] = df['price_momentum_5'] - df['volume_momentum_5']
    df['divergence_10'] = df['price_momentum_10'] - df['volume_momentum_10']
    df['divergence_20'] = df['price_momentum_20'] - df['volume_momentum_20']
    
    # Divergence Acceleration
    df['divergence_accel_5'] = df['divergence_5'] - df['divergence_5'].shift(1)
    df['divergence_accel_10'] = df['divergence_10'] - df['divergence_10'].shift(1)
    df['divergence_accel_20'] = df['divergence_20'] - df['divergence_20'].shift(1)
    
    # Volume Regime Transitions
    df['volume_accel'] = (df['volume'] / df['volume'].shift(5)) - (df['volume'].shift(5) / df['volume'].shift(10))
    df['volume_regime_break'] = (df['volume_accel'] * df['volume_accel'].shift(1) < 0).astype(int)
    df['volume_transition_strength'] = abs(df['volume_accel']) * df['volume_regime_break']
    
    # Volatility Regime Transitions
    df['true_range'] = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            abs(df['high'] - df['close'].shift(1)),
            abs(df['low'] - df['close'].shift(1))
        )
    )
    df['volatility_accel'] = (df['true_range'] / df['true_range'].shift(5)) - (df['true_range'].shift(5) / df['true_range'].shift(10))
    df['volatility_regime_break'] = (abs(df['volatility_accel']) > df['volatility_accel'].rolling(20).std()).astype(int)
    df['volatility_transition_strength'] = abs(df['volatility_accel']) * df['volatility_regime_break']
    
    # Amount-Based Participation Signals
    df['amount_momentum'] = df['amount'] / df['amount'].shift(5) - 1
    df['amount_accel'] = (df['amount'] / df['amount'].shift(5)) - (df['amount'].shift(5) / df['amount'].shift(10))
    df['participation_regime'] = np.where(df['amount_accel'] > 0, 1, -1)
    
    # Exponential Smoothing (alpha=0.3)
    alpha_ema = 0.3
    for col in ['price_momentum_5', 'price_momentum_10', 'price_momentum_20',
                'volume_momentum_5', 'volume_momentum_10', 'volume_momentum_20',
                'divergence_5', 'divergence_10', 'divergence_20']:
        df[f'ema_{col}'] = df[col].ewm(alpha=alpha_ema).mean()
    
    # Momentum Acceleration Signals (Second Derivatives)
    for col in ['price_momentum_5', 'volume_momentum_5']:
        ema_col = f'ema_{col}'
        df[f'{col}_accel_2nd'] = df[ema_col] - 2 * df[ema_col].shift(1) + df[ema_col].shift(2)
    
    # Cross-Sectional Ranking System
    for i in range(len(df)):
        if i >= 20:  # Ensure enough data for ranking
            current_data = df.iloc[:i+1]
            
            # Relative Divergence Strength Ranking
            divergence_rank_5 = current_data['divergence_5'].rank(pct=True).iloc[-1]
            divergence_rank_10 = current_data['divergence_10'].rank(pct=True).iloc[-1]
            divergence_rank_20 = current_data['divergence_20'].rank(pct=True).iloc[-1]
            
            # Divergence Acceleration Ranking
            accel_rank_5 = current_data['divergence_accel_5'].rank(pct=True).iloc[-1]
            accel_rank_10 = current_data['divergence_accel_10'].rank(pct=True).iloc[-1]
            accel_rank_20 = current_data['divergence_accel_20'].rank(pct=True).iloc[-1]
            
            # Regime Transition Strength Ranking
            volume_transition_rank = current_data['volume_transition_strength'].rank(pct=True).iloc[-1]
            volatility_transition_rank = current_data['volatility_transition_strength'].rank(pct=True).iloc[-1]
            
            # Core Divergence Signal
            positive_divergence = (df['divergence_5'].iloc[i] > 0 and 
                                 df['divergence_10'].iloc[i] > 0 and 
                                 df['divergence_20'].iloc[i] > 0)
            negative_divergence = (df['divergence_5'].iloc[i] < 0 and 
                                 df['divergence_10'].iloc[i] < 0 and 
                                 df['divergence_20'].iloc[i] < 0)
            
            # Signal Strength Calculation
            divergence_magnitude = (abs(df['divergence_5'].iloc[i]) + 
                                  abs(df['divergence_10'].iloc[i]) + 
                                  abs(df['divergence_20'].iloc[i])) / 3
            divergence_acceleration = (df['divergence_accel_5'].iloc[i] + 
                                     df['divergence_accel_10'].iloc[i] + 
                                     df['divergence_accel_20'].iloc[i]) / 3
            
            # Regime Transition Multiplier
            regime_multiplier = (volume_transition_rank * 0.4 + 
                               volatility_transition_rank * 0.4 + 
                               df['participation_regime'].iloc[i] * 0.2)
            
            # Cross-Sectional Enhancement
            leadership_rank = (divergence_rank_5 + divergence_rank_10 + divergence_rank_20) / 3
            consistency_score = (1 - abs(divergence_rank_5 - divergence_rank_10) - 
                               abs(divergence_rank_10 - divergence_rank_20)) / 3
            
            # Final Alpha Construction
            if positive_divergence:
                core_signal = divergence_magnitude * (1 + divergence_acceleration)
                alpha.iloc[i] = (core_signal * regime_multiplier * 
                               leadership_rank * consistency_score)
            elif negative_divergence:
                core_signal = -divergence_magnitude * (1 + abs(divergence_acceleration))
                alpha.iloc[i] = (core_signal * regime_multiplier * 
                               leadership_rank * consistency_score)
            else:
                # Mixed signals get reduced weighting
                mixed_signal = divergence_magnitude * divergence_acceleration
                alpha.iloc[i] = mixed_signal * regime_multiplier * 0.5
    
    # Fill initial NaN values with 0
    alpha = alpha.fillna(0)
    
    return alpha
