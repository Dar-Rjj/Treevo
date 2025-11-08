import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(data):
    """
    Regime-Adaptive Price-Volume Momentum Factor
    """
    df = data.copy()
    
    # Calculate returns for volatility calculations
    df['returns'] = df['close'].pct_change()
    
    # Multi-Scale Volatility Regime Detection
    # Short-term volatility regime (5-day window)
    df['vol_5d'] = df['returns'].rolling(window=5).std()
    df['vol_20d_baseline'] = df['returns'].rolling(window=20).std()
    
    # Medium-term volatility regime (20-day window)
    df['vol_20d'] = df['returns'].rolling(window=20).std()
    df['vol_60d_baseline'] = df['returns'].rolling(window=60).std()
    
    # Regime classification
    def classify_regime(current_vol, baseline_vol):
        if current_vol > baseline_vol * 1.4:
            return 'high'
        elif current_vol < baseline_vol * 0.6:
            return 'low'
        else:
            return 'normal'
    
    df['short_term_regime'] = df.apply(
        lambda x: classify_regime(x['vol_5d'], x['vol_20d_baseline']), axis=1
    )
    df['medium_term_regime'] = df.apply(
        lambda x: classify_regime(x['vol_20d'], x['vol_60d_baseline']), axis=1
    )
    
    # Regime consistency scoring
    df['regime_consistency'] = np.where(
        df['short_term_regime'] == df['medium_term_regime'], 1.2, 1.0
    )
    df['primary_regime'] = df['medium_term_regime']  # Use medium-term for stability
    
    # Dynamic Momentum Construction
    def calculate_momentum(close_series, days):
        return (close_series - close_series.shift(days)) / close_series.shift(days)
    
    # Calculate all momentum horizons
    momentum_horizons = {
        '2d': calculate_momentum(df['close'], 2),
        '5d': calculate_momentum(df['close'], 5),
        '10d': calculate_momentum(df['close'], 10),
        '15d': calculate_momentum(df['close'], 15),
        '20d': calculate_momentum(df['close'], 20),
        '30d': calculate_momentum(df['close'], 30),
        '40d': calculate_momentum(df['close'], 40)
    }
    
    # Regime-specific momentum weighting
    def get_weighted_momentum(row):
        regime = row['primary_regime']
        
        if regime == 'high':
            horizons = ['2d', '5d', '10d']
            weights = [0.5, 0.3, 0.2]
        elif regime == 'low':
            horizons = ['10d', '20d', '40d']
            weights = [0.2, 0.3, 0.5]
        else:  # normal
            horizons = ['5d', '15d', '30d']
            weights = [0.4, 0.3, 0.3]
        
        weighted_sum = 0
        for horizon, weight in zip(horizons, weights):
            weighted_sum += momentum_horizons[horizon].loc[row.name] * weight
        
        return weighted_sum
    
    df['weighted_momentum'] = df.apply(get_weighted_momentum, axis=1)
    
    # Intelligent Volume-Price Alignment
    # Multi-timeframe volume analysis
    df['volume_3d_change'] = df['volume'] / df['volume'].shift(3)
    df['volume_3d_direction'] = np.sign(df['volume'] - df['volume'].shift(3))
    df['volume_10d_change'] = df['volume'] / df['volume'].shift(10)
    df['volume_10d_direction'] = np.sign(df['volume'] - df['volume'].shift(10))
    
    # Volume trend consistency
    def get_volume_consistency(row):
        if row['volume_3d_direction'] == row['volume_10d_direction']:
            return 'strong'
        elif row['volume_3d_direction'] * row['volume_10d_direction'] == 0:
            return 'moderate'
        else:
            return 'weak'
    
    df['volume_consistency'] = df.apply(get_volume_consistency, axis=1)
    
    # Price-volume convergence scoring
    df['price_3d_direction'] = np.sign(df['close'] - df['close'].shift(3))
    df['price_10d_direction'] = np.sign(df['close'] - df['close'].shift(10))
    
    def get_convergence_score(row):
        short_term_match = row['price_3d_direction'] == row['volume_3d_direction']
        medium_term_match = row['price_10d_direction'] == row['volume_10d_direction']
        
        if short_term_match and medium_term_match:
            return 1.5  # strong convergence
        elif short_term_match or medium_term_match:
            return 1.0  # moderate convergence
        else:
            return 0.5  # weak convergence
    
    df['base_convergence'] = df.apply(get_convergence_score, axis=1)
    
    # Volume intensity adjustment
    df['volume_20d_avg'] = df['volume'].rolling(window=20).mean()
    df['volume_intensity'] = df['volume'] / df['volume_20d_avg']
    
    def get_intensity_multiplier(intensity):
        if intensity > 2.0:
            return 1.8
        elif intensity > 1.5:
            return 1.3
        else:
            return 1.0
    
    df['intensity_multiplier'] = df['volume_intensity'].apply(get_intensity_multiplier)
    df['volume_alignment_score'] = df['base_convergence'] * df['intensity_multiplier']
    
    # Integrated Factor Generation
    # Regime-consistent momentum adjustment
    df['consistency_adjusted_momentum'] = df['weighted_momentum'] * df['regime_consistency']
    
    # Momentum persistence check
    def calculate_persistence_bonus(momentum_series, regime_series):
        persistence_bonus = pd.Series(0.0, index=momentum_series.index)
        current_direction = 0
        consecutive_days = 0
        
        for i in range(1, len(momentum_series)):
            current_dir = np.sign(momentum_series.iloc[i])
            prev_dir = np.sign(momentum_series.iloc[i-1])
            
            if current_dir == prev_dir and current_dir != 0:
                consecutive_days += 1
            else:
                consecutive_days = 1 if current_dir != 0 else 0
            
            regime = regime_series.iloc[i]
            threshold = 3 if regime == 'high' else 5
            
            if consecutive_days >= threshold:
                bonus = min(consecutive_days * 0.02, 0.1)
                persistence_bonus.iloc[i] = bonus
        
        return persistence_bonus
    
    df['persistence_bonus'] = calculate_persistence_bonus(
        df['consistency_adjusted_momentum'], df['primary_regime']
    )
    
    df['adjusted_momentum'] = df['consistency_adjusted_momentum'] + df['persistence_bonus']
    
    # Volume alignment application
    df['volume_confirmed_momentum'] = df['adjusted_momentum'] * df['volume_alignment_score']
    
    # Apply regime-specific scaling
    def apply_regime_scaling(row):
        regime = row['primary_regime']
        momentum = row['volume_confirmed_momentum']
        
        if regime == 'high':
            return momentum * 0.8
        elif regime == 'low':
            return momentum * 1.2
        else:
            return momentum
    
    df['final_alpha'] = df.apply(apply_regime_scaling, axis=1)
    
    return df['final_alpha']
