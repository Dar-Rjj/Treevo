import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(data):
    """
    Regime-Adaptive Momentum-Volume Convergence Factor
    """
    df = data.copy()
    
    # Calculate raw momentum components
    df['price_momentum_5d'] = df['close'] / df['close'].shift(5) - 1
    df['price_momentum_10d'] = df['close'] / df['close'].shift(10) - 1
    df['price_momentum_20d'] = df['close'] / df['close'].shift(20) - 1
    
    # Calculate volume momentum components
    df['volume_momentum_5d'] = df['volume'] / df['volume'].shift(5) - 1
    df['volume_momentum_10d'] = df['volume'] / df['volume'].shift(10) - 1
    df['volume_momentum_20d'] = df['volume'] / df['volume'].shift(20) - 1
    
    # Regime detection - Amount based
    df['amount_momentum'] = df['amount'] / df['amount'].shift(10) - 1
    df['amount_acceleration'] = (df['amount'] / df['amount'].shift(5)) - (df['amount'].shift(5) / df['amount'].shift(10))
    
    # Regime states
    df['high_participation'] = (df['amount_momentum'] > 0) & (df['amount_acceleration'] > 0)
    df['low_participation'] = (df['amount_momentum'] < 0) & (df['amount_acceleration'] < 0)
    df['transition_regime'] = ~(df['high_participation'] | df['low_participation'])
    
    # Volatility environment
    df['true_range'] = df['high'] - df['low']
    df['volatility_momentum'] = df['true_range'] / df['true_range'].shift(10) - 1
    df['high_volatility'] = df['volatility_momentum'] > 0.05
    df['low_volatility'] = df['volatility_momentum'] < -0.05
    df['stable_volatility'] = (df['volatility_momentum'] >= -0.05) & (df['volatility_momentum'] <= 0.05)
    
    # Initialize EMA columns
    for col in ['price_momentum_5d', 'price_momentum_10d', 'price_momentum_20d',
                'volume_momentum_5d', 'volume_momentum_10d', 'volume_momentum_20d']:
        df[f'EMA_{col}'] = np.nan
    
    # Calculate EMA for price and volume momentum (alpha = 0.3)
    alpha = 0.3
    for i in range(len(df)):
        if i >= 20:  # Ensure we have enough data
            for period in ['5d', '10d', '20d']:
                price_col = f'price_momentum_{period}'
                volume_col = f'volume_momentum_{period}'
                
                # Price momentum EMA
                if i == 20:
                    df.loc[df.index[i], f'EMA_{price_col}'] = df.loc[df.index[i], price_col]
                else:
                    prev_ema = df.loc[df.index[i-1], f'EMA_{price_col}']
                    if not np.isnan(prev_ema):
                        df.loc[df.index[i], f'EMA_{price_col}'] = (alpha * df.loc[df.index[i], price_col] + 
                                                                  (1-alpha) * prev_ema)
                
                # Volume momentum EMA
                if i == 20:
                    df.loc[df.index[i], f'EMA_{volume_col}'] = df.loc[df.index[i], volume_col]
                else:
                    prev_ema = df.loc[df.index[i-1], f'EMA_{volume_col}']
                    if not np.isnan(prev_ema):
                        df.loc[df.index[i], f'EMA_{volume_col}'] = (alpha * df.loc[df.index[i], volume_col] + 
                                                                   (1-alpha) * prev_ema)
    
    # Convergence measures
    df['convergence_5d'] = df['EMA_price_momentum_5d'] - df['EMA_volume_momentum_5d']
    df['convergence_10d'] = df['EMA_price_momentum_10d'] - df['EMA_volume_momentum_10d']
    df['convergence_20d'] = df['EMA_price_momentum_20d'] - df['EMA_volume_momentum_20d']
    
    # Multi-timeframe consistency
    df['consistent_convergence'] = ((df['convergence_5d'] > 0) & 
                                   (df['convergence_10d'] > 0) & 
                                   (df['convergence_20d'] > 0))
    df['consistent_divergence'] = ((df['convergence_5d'] < 0) & 
                                  (df['convergence_10d'] < 0) & 
                                  (df['convergence_20d'] < 0))
    
    # Cross-sectional ranking preparation
    df['rank_convergence_5d'] = df.groupby(df.index)['convergence_5d'].rank(pct=True)
    df['rank_convergence_10d'] = df.groupby(df.index)['convergence_10d'].rank(pct=True)
    df['rank_convergence_20d'] = df.groupby(df.index)['convergence_20d'].rank(pct=True)
    
    # Final factor construction with regime adaptation
    df['final_factor'] = 0.0
    
    for i in range(len(df)):
        if i >= 20:
            # Base weights
            weights = {'5d': 0.4, '10d': 0.35, '20d': 0.25}
            
            # Regime-based adjustments
            if df.loc[df.index[i], 'high_participation']:
                # Emphasize volume confirmation in high participation
                weights = {'5d': 0.3, '10d': 0.4, '20d': 0.3}
            elif df.loc[df.index[i], 'low_participation']:
                # Emphasize price momentum persistence in low participation
                weights = {'5d': 0.5, '10d': 0.3, '20d': 0.2}
            
            # Volatility adjustments
            if df.loc[df.index[i], 'high_volatility']:
                # Weight recent convergence more heavily
                weights = {'5d': 0.6, '10d': 0.3, '20d': 0.1}
            elif df.loc[df.index[i], 'low_volatility']:
                # Emphasize consistency across timeframes
                weights = {'5d': 0.25, '10d': 0.35, '20d': 0.4}
            
            # Calculate weighted convergence score
            convergence_score = (weights['5d'] * df.loc[df.index[i], 'rank_convergence_5d'] +
                               weights['10d'] * df.loc[df.index[i], 'rank_convergence_10d'] +
                               weights['20d'] * df.loc[df.index[i], 'rank_convergence_20d'])
            
            # Regime signal strength adjustment
            signal_strength = 1.0
            if df.loc[df.index[i], 'transition_regime']:
                signal_strength = 0.7
            
            # Multi-timeframe consistency bonus
            if df.loc[df.index[i], 'consistent_convergence']:
                convergence_score *= 1.2
            elif df.loc[df.index[i], 'consistent_divergence']:
                convergence_score *= 0.8
            
            df.loc[df.index[i], 'final_factor'] = convergence_score * signal_strength
    
    return df['final_factor']
