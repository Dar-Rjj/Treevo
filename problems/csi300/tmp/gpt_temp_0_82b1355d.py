import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Volume Distribution & Persistence Analysis
    # Calculate volume concentration (using rolling windows for intraday patterns)
    data['volume_ma5'] = data['volume'].rolling(window=5, min_periods=3).mean()
    data['volume_ma20'] = data['volume'].rolling(window=20, min_periods=10).mean()
    
    # Volume skewness (deviation from normal)
    data['volume_skew'] = (data['volume'] - data['volume_ma20']) / (data['volume_ma20'] + 1e-8)
    
    # Volume Persistence Score
    data['volume_deviation'] = np.abs(data['volume_skew'])
    
    # Calculate consecutive regime days for volume
    volume_regime = (data['volume'] > data['volume_ma20']).astype(int)
    data['volume_regime_change'] = volume_regime.diff().fillna(0)
    data['volume_consecutive_days'] = 0
    current_streak = 0
    for i in range(len(data)):
        if i == 0:
            data.iloc[i, data.columns.get_loc('volume_consecutive_days')] = 1
            current_streak = 1
        else:
            if data['volume_regime_change'].iloc[i] == 0:
                current_streak += 1
            else:
                current_streak = 1
            data.iloc[i, data.columns.get_loc('volume_consecutive_days')] = current_streak
    
    data['volume_persistence_score'] = data['volume_consecutive_days'] * data['volume_deviation']
    
    # Bidirectional Price-Volume Divergence Detection
    data['buy_side_pressure'] = (data['high'] - data['close']) * data['volume']
    data['sell_side_pressure'] = (data['close'] - data['low']) * data['volume']
    data['raw_gap'] = data['buy_side_pressure'] - data['sell_side_pressure']
    
    # Short-term divergence (3-day)
    data['price_change_3d'] = data['close'].pct_change(periods=3)
    data['volume_change_3d'] = data['volume'].pct_change(periods=3)
    data['price_magnitude_3d'] = np.abs(data['price_change_3d'])
    data['volume_magnitude_3d'] = np.abs(data['volume_change_3d'])
    
    # Directional divergence scoring
    data['directional_div_3d'] = np.sign(data['price_change_3d']) * np.sign(data['volume_change_3d'])
    data['magnitude_div_3d'] = data['price_magnitude_3d'] - data['volume_magnitude_3d']
    data['short_term_divergence'] = data['directional_div_3d'] * data['magnitude_div_3d']
    
    # Divergence persistence measure (3-day rolling correlation)
    data['price_returns_3d'] = data['close'].pct_change()
    data['volume_returns_3d'] = data['volume'].pct_change()
    
    # Calculate rolling correlation for divergence persistence
    corr_window = 3
    correlations = []
    for i in range(len(data)):
        if i < corr_window:
            correlations.append(0)
        else:
            price_changes = data['price_returns_3d'].iloc[i-corr_window+1:i+1]
            volume_changes = data['volume_returns_3d'].iloc[i-corr_window+1:i+1]
            if len(price_changes) > 1 and len(volume_changes) > 1:
                corr = np.corrcoef(price_changes, volume_changes)[0, 1]
                correlations.append(corr if not np.isnan(corr) else 0)
            else:
                correlations.append(0)
    data['divergence_persistence_3d'] = correlations
    
    # Medium-term divergence (8-day)
    data['price_change_8d'] = data['close'].pct_change(periods=8)
    data['volume_change_8d'] = data['volume'].pct_change(periods=8)
    data['price_magnitude_8d'] = np.abs(data['price_change_8d'])
    data['volume_magnitude_8d'] = np.abs(data['volume_change_8d'])
    
    # Cumulative divergence
    data['cumulative_divergence_8d'] = data['price_magnitude_8d'] - data['volume_magnitude_8d']
    
    # Trend-volume alignment
    data['trend_volume_alignment'] = np.sign(data['price_change_8d']) * np.sign(data['volume_change_8d'])
    data['medium_term_divergence'] = data['cumulative_divergence_8d'] * data['trend_volume_alignment']
    
    # Multi-Timeframe Volatility Regime Context
    data['daily_range'] = data['high'] - data['low']
    data['range_ma5'] = data['daily_range'].rolling(window=5, min_periods=3).mean()
    data['range_ma20'] = data['daily_range'].rolling(window=20, min_periods=10).mean()
    data['volatility_ratio'] = data['range_ma5'] / (data['range_ma20'] + 1e-8)
    
    # Price volatility regime
    data['price_vol_regime'] = (data['daily_range'] > data['range_ma20']).astype(int)
    
    # Volume volatility regime
    data['volume_volatility'] = data['volume'].rolling(window=20, min_periods=10).std()
    data['volume_vol_regime'] = (data['volume'] > data['volume_ma20'] + data['volume_volatility']).astype(int)
    
    # Multi-Scale Factor Synthesis
    # Volatility-Scaled Gap
    data['volatility_scaled_gap'] = data['raw_gap'] * data['volatility_ratio']
    
    # Volume-Filtered Gap
    data['volume_filtered_gap'] = data['volatility_scaled_gap'] * data['volume_persistence_score']
    
    # Multi-scale divergence synthesis
    data['divergence_strength_ratio'] = np.abs(data['short_term_divergence']) / (np.abs(data['medium_term_divergence']) + 1e-8)
    data['divergence_alignment'] = np.sign(data['short_term_divergence']) * np.sign(data['medium_term_divergence'])
    
    # Composite divergence strength score
    data['composite_divergence_strength'] = (
        np.abs(data['short_term_divergence']) * 0.6 + 
        np.abs(data['medium_term_divergence']) * 0.4
    ) * data['divergence_alignment']
    
    # Momentum-Enhanced Gap
    data['momentum_enhanced_gap'] = data['volume_filtered_gap'] * data['composite_divergence_strength']
    
    # Final Alpha with cube root transformation
    data['alpha'] = np.sign(data['momentum_enhanced_gap']) * np.power(np.abs(data['momentum_enhanced_gap']), 1/3)
    
    # Clean up and return
    result = data['alpha'].replace([np.inf, -np.inf], np.nan).fillna(0)
    return result
