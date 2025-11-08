import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Generate alpha factor based on multi-timeframe divergence analysis with regime adaptation
    """
    # Create copies to avoid modifying original data
    data = df.copy()
    
    # Step 1: Trend Smoothing
    # Calculate typical price
    data['typical_price'] = (data['high'] + data['low'] + data['close']) / 3
    data['daily_range'] = (data['high'] - data['low']) / data['close']
    
    # 5-day EMAs
    data['price_smooth'] = data['typical_price'].ewm(span=5, adjust=False).mean()
    data['volume_smooth'] = data['volume'].ewm(span=5, adjust=False).mean()
    data['volatility_smooth'] = data['daily_range'].ewm(span=5, adjust=False).mean()
    
    # Step 2: Multi-Timeframe Momentum Calculation
    # Short-term momentum (3-day)
    data['price_mom_3d'] = data['price_smooth'] / data['price_smooth'].shift(3) - 1
    data['volume_mom_3d'] = data['volume_smooth'] / data['volume_smooth'].shift(3) - 1
    data['volatility_mom_3d'] = data['volatility_smooth'] / data['volatility_smooth'].shift(3) - 1
    
    # Medium-term momentum (10-day)
    data['price_mom_10d'] = data['price_smooth'] / data['price_smooth'].shift(10) - 1
    data['volume_mom_10d'] = data['volume_smooth'] / data['volume_smooth'].shift(10) - 1
    data['volatility_mom_10d'] = data['volatility_smooth'] / data['volatility_smooth'].shift(10) - 1
    
    # Step 3: Divergence Detection
    # Price-Volume divergence
    data['pv_divergence'] = (np.sign(data['price_mom_3d']) != np.sign(data['volume_mom_3d'])).astype(int)
    data['pv_divergence_10d'] = (np.sign(data['price_mom_10d']) != np.sign(data['volume_mom_10d'])).astype(int)
    
    # Price-Volatility divergence
    data['pvol_divergence'] = (np.sign(data['price_mom_3d']) != np.sign(data['volatility_mom_3d'])).astype(int)
    data['pvol_divergence_10d'] = (np.sign(data['price_mom_10d']) != np.sign(data['volatility_mom_10d'])).astype(int)
    
    # Cross-timeframe divergence
    data['timeframe_divergence'] = (np.sign(data['price_mom_3d']) != np.sign(data['price_mom_10d'])).astype(int)
    
    # Step 4: Persistence and Confirmation Framework
    # Divergence persistence scoring
    divergence_cols = ['pv_divergence', 'pvol_divergence', 'timeframe_divergence']
    
    for col in divergence_cols:
        data[f'{col}_persistence'] = 0
        current_streak = 0
        for i in range(len(data)):
            if data[col].iloc[i] == 1:
                current_streak += 1
            else:
                current_streak = 0
            data[f'{col}_persistence'].iloc[i] = current_streak
    
    # Multi-signal confirmation
    data['divergence_count'] = data[divergence_cols].sum(axis=1)
    data['multi_confirmation'] = (data['divergence_count'] >= 2).astype(int)
    
    # Trend strength assessment
    data['momentum_acceleration'] = data['price_mom_3d'] - data['price_mom_10d']
    
    # Calculate rolling variance for trend stability (5-day window)
    data['returns'] = data['close'].pct_change()
    data['trend_stability'] = data['returns'].rolling(window=5).var()
    
    # Volume-price correlation (5-day rolling)
    data['volume_price_corr'] = data['volume'].rolling(window=5).corr(data['close'])
    
    # Step 5: Regime-Adaptive Signal Generation
    # Volatility regime classification
    data['historical_vol'] = data['returns'].rolling(window=20).std()
    data['current_vol'] = data['returns'].rolling(window=5).std()
    data['vol_regime'] = np.where(data['current_vol'] > data['historical_vol'] * 1.2, 'high',
                         np.where(data['current_vol'] < data['historical_vol'] * 0.8, 'low', 'normal'))
    
    # Volatility trend
    data['vol_trend'] = data['current_vol'] / data['current_vol'].shift(5) - 1
    
    # Step 6: Composite Factor Construction
    # Primary divergence score
    data['base_divergence_score'] = (
        data['pv_divergence_persistence'] * 0.4 +
        data['pvol_divergence_persistence'] * 0.3 +
        data['timeframe_divergence_persistence'] * 0.3
    )
    
    # Apply divergence magnitude weighting
    pv_magnitude = np.abs(data['price_mom_3d'] * data['volume_mom_3d'])
    pvol_magnitude = np.abs(data['price_mom_3d'] * data['volatility_mom_3d'])
    timeframe_magnitude = np.abs(data['price_mom_3d'] * data['price_mom_10d'])
    
    data['weighted_divergence_score'] = (
        data['pv_divergence_persistence'] * pv_magnitude * 0.4 +
        data['pvol_divergence_persistence'] * pvol_magnitude * 0.3 +
        data['timeframe_divergence_persistence'] * timeframe_magnitude * 0.3
    )
    
    # Confidence multiplier
    data['confirmation_strength'] = (
        data['multi_confirmation'] * 0.5 +
        (data['divergence_count'] / 3) * 0.3 +
        np.clip(data['volume_price_corr'], -1, 0) * 0.2  # Negative correlation is better for divergence
    )
    
    # Regime-based adjustments
    regime_multiplier = np.where(data['vol_regime'] == 'high', 0.7,
                        np.where(data['vol_regime'] == 'low', 1.3, 1.0))
    
    # Final alpha signal construction
    data['raw_alpha'] = (
        data['weighted_divergence_score'] * 
        data['confirmation_strength'] * 
        regime_multiplier * 
        np.sign(data['momentum_acceleration'])
    )
    
    # Apply smoothing and capping
    data['alpha_smoothed'] = data['raw_alpha'].ewm(span=3, adjust=False).mean()
    
    # Cap extreme values (95th percentile)
    alpha_95pct = data['alpha_smoothed'].quantile(0.95)
    alpha_5pct = data['alpha_smoothed'].quantile(0.05)
    data['final_alpha'] = np.clip(data['alpha_smoothed'], alpha_5pct, alpha_95pct)
    
    # Normalize final output
    alpha_mean = data['final_alpha'].mean()
    alpha_std = data['final_alpha'].std()
    data['normalized_alpha'] = (data['final_alpha'] - alpha_mean) / alpha_std
    
    return data['normalized_alpha']
