import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Regime-Adaptive Momentum Efficiency Factor
    Combines momentum analysis, volume-liquidity integration, and regime detection
    to generate adaptive alpha signals.
    """
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # 1. Regime Identification
    # Volatility Regime Detection
    # Calculate True Range
    data['prev_close'] = data['close'].shift(1)
    data['tr1'] = data['high'] - data['low']
    data['tr2'] = abs(data['high'] - data['prev_close'])
    data['tr3'] = abs(data['low'] - data['prev_close'])
    data['true_range'] = data[['tr1', 'tr2', 'tr3']].max(axis=1)
    
    # Rolling volatility measures
    data['volatility_20d'] = data['true_range'].rolling(window=20, min_periods=10).mean()
    data['volatility_median'] = data['volatility_20d'].rolling(window=60, min_periods=30).median()
    
    # Volatility regime classification
    data['high_vol_regime'] = (data['volatility_20d'] > data['volatility_median']).astype(int)
    
    # Market State Classification - Price Fractal Dimension approximation
    def calculate_fractal_dimension(high_series, low_series, window=20):
        fractal_scores = []
        for i in range(len(high_series)):
            if i < window:
                fractal_scores.append(np.nan)
                continue
            window_high = high_series.iloc[i-window:i]
            window_low = low_series.iloc[i-window:i]
            
            # Simplified fractal dimension calculation
            range_sum = (window_high - window_low).sum()
            price_range = window_high.max() - window_low.min()
            
            if price_range > 0:
                fractal_score = range_sum / (price_range * window)
            else:
                fractal_score = 1.0
            fractal_scores.append(fractal_score)
        
        return pd.Series(fractal_scores, index=high_series.index)
    
    data['fractal_score'] = calculate_fractal_dimension(data['high'], data['low'])
    data['trending_market'] = (data['fractal_score'] < 0.6).astype(int)
    
    # Combined regime classification
    data['regime'] = 0  # Default: consolidation
    data.loc[(data['high_vol_regime'] == 1) & (data['trending_market'] == 1), 'regime'] = 1  # Breakout
    data.loc[(data['high_vol_regime'] == 1) & (data['trending_market'] == 0), 'regime'] = 2  # Mean reversion
    data.loc[(data['high_vol_regime'] == 0) & (data['trending_market'] == 1), 'regime'] = 3  # Momentum
    
    # 2. Momentum Efficiency Analysis
    # Multi-Timeframe Momentum
    data['mom_5d'] = data['close'] / data['close'].shift(5) - 1
    data['mom_10d'] = data['close'] / data['close'].shift(10) - 1
    data['mom_20d'] = data['close'] / data['close'].shift(20) - 1
    
    # Momentum Acceleration
    data['mom_accel_short'] = data['mom_5d'] - data['mom_10d']
    data['mom_accel_medium'] = data['mom_10d'] - data['mom_20d']
    
    # Intraday Price Efficiency
    data['daily_range'] = data['high'] - data['low']
    data['abs_intraday_move'] = abs(data['close'] - data['open'])
    data['efficiency_ratio'] = data['abs_intraday_move'] / data['daily_range']
    data['efficiency_ratio'] = data['efficiency_ratio'].replace([np.inf, -np.inf], np.nan).fillna(0)
    data['efficiency_ratio'] = np.clip(data['efficiency_ratio'], 0, 1)
    
    # 3. Volume-Liquidity Integration
    # Volume Acceleration Analysis
    data['volume_mom_5d'] = data['volume'] / data['volume'].shift(5) - 1
    data['volume_mom_20d'] = data['volume'] / data['volume'].shift(20) - 1
    data['volume_accel'] = data['volume_mom_5d'] - data['volume_mom_20d']
    
    # Liquidity Quality Assessment
    data['volume_to_amount'] = data['amount'] / data['volume']
    data['volume_to_amount'] = data['volume_to_amount'].replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Volume clusters detection
    data['volume_median_20d'] = data['volume'].rolling(window=20, min_periods=10).median()
    data['high_volume_cluster'] = (data['volume'] > data['volume_median_20d'] * 1.5).astype(int)
    
    # Price-Volume Divergence
    data['price_volume_corr_10d'] = data['mom_5d'].rolling(window=10, min_periods=5).corr(data['volume_mom_5d'])
    data['price_volume_aligned'] = (data['price_volume_corr_10d'] > 0).astype(int)
    
    # 4. Regime-Adaptive Signal Weighting
    # Initialize component scores
    data['momentum_score'] = 0.0
    data['efficiency_score'] = 0.0
    data['volume_score'] = 0.0
    
    # High Volatility Regime (Breakout & Mean Reversion)
    high_vol_mask = data['regime'].isin([1, 2])
    
    # Mean Reversion emphasis in high volatility
    mean_rev_mask = (data['regime'] == 2) & high_vol_mask
    data.loc[mean_rev_mask, 'efficiency_score'] = (1 - data['efficiency_ratio']) * 2  # Higher weight to low efficiency
    data.loc[mean_rev_mask, 'momentum_score'] = -data['mom_5d'] * 0.5  # Contrarian momentum
    
    # Breakout emphasis in high volatility
    breakout_mask = (data['regime'] == 1) & high_vol_mask
    data.loc[breakout_mask, 'momentum_score'] = data['mom_accel_short'] * 1.5
    data.loc[breakout_mask, 'volume_score'] = data['volume_accel'] * 2.0
    
    # Low Volatility Regime (Momentum & Consolidation)
    low_vol_mask = data['regime'].isin([0, 3])
    
    # Momentum regime
    momentum_mask = (data['regime'] == 3) & low_vol_mask
    data.loc[momentum_mask, 'momentum_score'] = (
        data['mom_5d'] * 0.4 + 
        data['mom_10d'] * 0.3 + 
        data['mom_20d'] * 0.3
    )
    data.loc[momentum_mask, 'efficiency_score'] = data['efficiency_ratio'] * 1.2
    
    # Consolidation regime
    consolidation_mask = (data['regime'] == 0) & low_vol_mask
    data.loc[consolidation_mask, 'momentum_score'] = data['mom_accel_short'] * 0.8
    data.loc[consolidation_mask, 'efficiency_score'] = (1 - data['efficiency_ratio']) * 1.0
    
    # Volume weighting adjustments
    # Volume-aligned conditions
    volume_aligned_mask = data['price_volume_aligned'] == 1
    data.loc[volume_aligned_mask, 'volume_score'] += data['volume_accel'] * 1.5
    
    # Volume clusters boost
    cluster_mask = data['high_volume_cluster'] == 1
    data.loc[cluster_mask, 'volume_score'] += 0.5
    
    # Liquidity quality adjustment
    data['liquidity_quality'] = np.tanh(data['volume_to_amount'] / data['volume_to_amount'].rolling(window=20).std())
    data['volume_score'] *= (1 + data['liquidity_quality'] * 0.3)
    
    # 5. Composite Alpha Generation
    # Normalize component scores
    for col in ['momentum_score', 'efficiency_score', 'volume_score']:
        data[col] = (data[col] - data[col].rolling(window=60, min_periods=30).mean()) / data[col].rolling(window=60, min_periods=30).std()
        data[col] = data[col].fillna(0)
    
    # Regime-adaptive weights
    regime_weights = {
        0: [0.3, 0.4, 0.3],  # Consolidation: efficiency focus
        1: [0.5, 0.2, 0.3],  # Breakout: momentum focus
        2: [0.2, 0.5, 0.3],  # Mean reversion: efficiency focus  
        3: [0.6, 0.2, 0.2]   # Momentum: strong momentum focus
    }
    
    # Calculate final alpha
    alpha_values = []
    for idx, row in data.iterrows():
        if pd.isna(row['regime']):
            alpha_values.append(0.0)
            continue
            
        regime = int(row['regime'])
        weights = regime_weights.get(regime, [0.33, 0.33, 0.34])
        
        alpha = (
            weights[0] * row['momentum_score'] +
            weights[1] * row['efficiency_score'] + 
            weights[2] * row['volume_score']
        )
        alpha_values.append(alpha)
    
    data['alpha'] = alpha_values
    
    # Final smoothing and normalization
    data['alpha_smoothed'] = data['alpha'].rolling(window=5, min_periods=3).mean()
    data['final_alpha'] = (data['alpha_smoothed'] - data['alpha_smoothed'].rolling(window=60).mean()) / data['alpha_smoothed'].rolling(window=60).std()
    data['final_alpha'] = data['final_alpha'].fillna(0)
    
    return data['final_alpha']
