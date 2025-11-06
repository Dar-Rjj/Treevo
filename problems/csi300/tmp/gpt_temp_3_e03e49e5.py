import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate Price Acceleration Components
    # Multi-Timeframe Price Changes
    data['price_change_3d'] = data['close'] - data['close'].shift(3)
    data['price_change_8d'] = data['close'] - data['close'].shift(8)
    data['price_change_15d'] = data['close'] - data['close'].shift(15)
    
    # Derive Acceleration Signals
    data['short_term_accel'] = np.where(
        data['price_change_8d'] != 0,
        (data['price_change_3d'] - data['price_change_8d']) / data['price_change_8d'].abs(),
        0
    )
    data['medium_term_accel'] = np.where(
        data['price_change_15d'] != 0,
        (data['price_change_8d'] - data['price_change_15d']) / data['price_change_15d'].abs(),
        0
    )
    
    # Calculate Volume-Weighted Price Efficiency
    # Volume-Adjusted Price Range
    data['daily_range'] = data['high'] - data['low']
    data['volume_weighted_range'] = data['daily_range'] * np.log(data['volume'] + 1)
    
    # Directional Efficiency
    data['abs_price_move'] = abs(data['close'] - data['open'])
    data['efficiency_ratio'] = np.where(
        data['volume_weighted_range'] > 0,
        data['abs_price_move'] / data['volume_weighted_range'],
        0
    )
    
    # Detect Market Regime Conditions
    # Calculate True Range for volatility
    data['tr1'] = data['high'] - data['low']
    data['tr2'] = abs(data['high'] - data['close'].shift(1))
    data['tr3'] = abs(data['low'] - data['close'].shift(1))
    data['true_range'] = data[['tr1', 'tr2', 'tr3']].max(axis=1)
    data['volatility_10d'] = data['true_range'].rolling(window=10, min_periods=5).mean()
    
    # Volume ratio
    data['volume_5d_avg'] = data['volume'].rolling(window=5, min_periods=3).mean()
    data['volume_20d_avg'] = data['volume'].rolling(window=20, min_periods=10).mean()
    data['volume_ratio'] = np.where(
        data['volume_20d_avg'] > 0,
        data['volume_5d_avg'] / data['volume_20d_avg'],
        1
    )
    
    # Price trend consistency using rolling correlation
    data['return_5d'] = data['close'].pct_change(5)
    data['return_10d'] = data['close'].pct_change(10)
    
    # Calculate rolling correlation
    correlation_window = 10
    correlations = []
    for i in range(len(data)):
        if i >= correlation_window - 1:
            start_idx = i - correlation_window + 1
            end_idx = i + 1
            corr_val = data['return_5d'].iloc[start_idx:end_idx].corr(
                data['return_10d'].iloc[start_idx:end_idx]
            )
            correlations.append(corr_val if not pd.isna(corr_val) else 0)
        else:
            correlations.append(0)
    data['trend_correlation'] = correlations
    
    # Classify Regime Types
    data['volatility_percentile'] = data['volatility_10d'].rolling(window=20, min_periods=10).apply(
        lambda x: (x.iloc[-1] > np.percentile(x.dropna(), 80)) if len(x.dropna()) > 0 else False
    )
    data['high_vol_regime'] = data['volatility_percentile'].astype(bool)
    data['high_volume_regime'] = (data['volume_ratio'] > 1.2)
    data['trending_regime'] = (data['trend_correlation'] > 0.7)
    
    # Combine Acceleration with Volume Efficiency
    data['accel_efficiency_short'] = data['short_term_accel'] * data['efficiency_ratio']
    data['accel_efficiency_medium'] = data['medium_term_accel'] * data['efficiency_ratio']
    
    # Weighted average favoring recent acceleration
    data['combined_signal'] = (
        0.6 * data['accel_efficiency_short'] + 
        0.4 * data['accel_efficiency_medium']
    )
    
    # Apply Regime-Based Signal Modulation
    # Initialize regime-adjusted signal
    data['regime_adjusted_signal'] = data['combined_signal']
    
    # High volatility regime adjustment
    vol_mask = data['high_vol_regime']
    data.loc[vol_mask, 'regime_adjusted_signal'] = (
        data.loc[vol_mask, 'regime_adjusted_signal'] * 
        (data.loc[vol_mask, 'volatility_10d'] / data.loc[vol_mask, 'volatility_10d'].rolling(window=20, min_periods=10).mean())
    )
    
    # High volume regime adjustment
    volume_mask = data['high_volume_regime']
    data.loc[volume_mask, 'regime_adjusted_signal'] = (
        data.loc[volume_mask, 'regime_adjusted_signal'] * data.loc[volume_mask, 'volume_ratio']
    )
    
    # Trending regime adjustment - enhance acceleration component
    trend_mask = data['trending_regime']
    data.loc[trend_mask, 'regime_adjusted_signal'] = (
        data.loc[trend_mask, 'regime_adjusted_signal'] * 1.2 + 
        0.3 * data.loc[trend_mask, 'short_term_accel']
    )
    
    # Generate Final Alpha Factor
    # Apply regime-specific scaling and ensure no lookahead bias
    alpha_factor = data['regime_adjusted_signal'].copy()
    
    # Clean any infinite or NaN values
    alpha_factor = alpha_factor.replace([np.inf, -np.inf], np.nan)
    alpha_factor = alpha_factor.fillna(0)
    
    return alpha_factor
