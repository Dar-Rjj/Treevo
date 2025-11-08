import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(data):
    """
    Multi-Timeframe Momentum Convergence with Volume-Price Divergence alpha factor
    """
    df = data.copy()
    epsilon = 1e-8
    
    # Multi-Timeframe Momentum Construction
    # Short-Term Momentum (1-day)
    df['intraday_return'] = df['close'] - df['open']
    df['intraday_range'] = df['high'] - df['low']
    df['short_momentum'] = df['intraday_return'] / (df['intraday_range'] + epsilon)
    
    # Medium-Term Momentum (3-day)
    df['medium_price_return'] = df['close'] / df['close'].shift(3) - 1
    df['medium_volatility'] = (df['high'] - df['low']).rolling(window=3, min_periods=3).mean()
    df['medium_momentum'] = df['medium_price_return'] / (df['medium_volatility'] + epsilon)
    
    # Long-Term Momentum (5-day)
    df['long_price_return'] = df['close'] / df['close'].shift(5) - 1
    df['long_volatility'] = (df['high'] - df['low']).rolling(window=5, min_periods=5).mean()
    df['long_momentum'] = df['long_price_return'] / (df['long_volatility'] + epsilon)
    
    # Momentum Convergence Signal
    # Direction Alignment Check
    df['short_dir'] = np.sign(df['short_momentum'])
    df['medium_dir'] = np.sign(df['medium_momentum'])
    df['long_dir'] = np.sign(df['long_momentum'])
    
    df['convergence_count'] = (df['short_dir'] == df['medium_dir']).astype(int) + \
                             (df['short_dir'] == df['long_dir']).astype(int) + \
                             (df['medium_dir'] == df['long_dir']).astype(int)
    
    # Magnitude Weighting
    momentum_magnitudes = []
    for i in range(len(df)):
        dirs = [df['short_dir'].iloc[i], df['medium_dir'].iloc[i], df['long_dir'].iloc[i]]
        mags = [abs(df['short_momentum'].iloc[i]), abs(df['medium_momentum'].iloc[i]), abs(df['long_momentum'].iloc[i])]
        
        # Find majority direction
        pos_count = sum(1 for d in dirs if d > 0)
        neg_count = sum(1 for d in dirs if d < 0)
        majority_dir = 1 if pos_count > neg_count else -1 if neg_count > pos_count else 0
        
        # Calculate weighted magnitude for convergent timeframes
        convergent_mags = []
        for j in range(3):
            for k in range(j+1, 3):
                if dirs[j] == dirs[k]:
                    convergent_mags.append((mags[j] + mags[k]) / 2)
        
        if convergent_mags:
            avg_magnitude = np.mean(convergent_mags)
            weighted_signal = majority_dir * avg_magnitude * (len(convergent_mags) / 3)
        else:
            weighted_signal = 0
        
        momentum_magnitudes.append(weighted_signal)
    
    df['momentum_convergence'] = momentum_magnitudes
    
    # Convergence Persistence
    persistence = []
    current_streak = 0
    for i in range(len(df)):
        if df['convergence_count'].iloc[i] >= 2:  # At least partial convergence
            current_streak += 1
        else:
            current_streak = 0
        persistence.append(current_streak)
    
    df['convergence_persistence'] = persistence
    df['persistence_weighted_convergence'] = df['momentum_convergence'] * (1 + df['convergence_persistence'] * 0.1)
    
    # Volume-Price Divergence Analysis
    # Multi-Timeframe Volume Momentum
    df['short_volume_momentum'] = df['volume'] / df['volume'].shift(1) - 1
    df['medium_volume_momentum'] = df['volume'] / df['volume'].shift(3) - 1
    df['long_volume_momentum'] = df['volume'] / df['volume'].shift(5) - 1
    
    # Volume-Price Divergence Detection
    divergence_scores = []
    for i in range(len(df)):
        price_dirs = [df['short_dir'].iloc[i], df['medium_dir'].iloc[i], df['long_dir'].iloc[i]]
        volume_dirs = [np.sign(df['short_volume_momentum'].iloc[i]), 
                      np.sign(df['medium_volume_momentum'].iloc[i]), 
                      np.sign(df['long_volume_momentum'].iloc[i])]
        
        timeframe_divergence = []
        for j in range(3):
            if price_dirs[j] != 0 and volume_dirs[j] != 0:
                if price_dirs[j] > 0 and volume_dirs[j] < 0:  # Positive divergence
                    timeframe_divergence.append(-1.0 * abs(df['short_volume_momentum'].iloc[i]))
                elif price_dirs[j] < 0 and volume_dirs[j] > 0:  # Negative divergence
                    timeframe_divergence.append(-1.0 * abs(df['short_volume_momentum'].iloc[i]))
                else:  # Confirmation
                    timeframe_divergence.append(1.0 * abs(df['short_volume_momentum'].iloc[i]))
            else:
                timeframe_divergence.append(0)
        
        # Multi-timeframe divergence consistency
        if len([x for x in timeframe_divergence if x != 0]) >= 2:
            avg_divergence = np.mean([x for x in timeframe_divergence if x != 0])
        else:
            avg_divergence = 0
        
        divergence_scores.append(avg_divergence)
    
    df['volume_divergence'] = divergence_scores
    
    # Factor Integration and Scaling
    # Combine Convergence and Divergence Signals
    df['combined_signal'] = df['persistence_weighted_convergence'] * (1 + df['volume_divergence'])
    
    # Volatility Context Adjustment
    df['market_volatility'] = (df['high'] - df['low']).rolling(window=5, min_periods=5).mean()
    df['volatility_adjusted_factor'] = df['combined_signal'] / (df['market_volatility'] + epsilon)
    
    # Final Alpha Output
    alpha = df['volatility_adjusted_factor']
    alpha.name = 'momentum_convergence_volume_divergence'
    
    return alpha
