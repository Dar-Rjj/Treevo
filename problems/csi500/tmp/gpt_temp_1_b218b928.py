import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Intraday Price Efficiency Analysis
    # Daily Price Range Utilization
    data['daily_range'] = data['high'] - data['low']
    data['close_open_gap'] = data['close'] - data['open']
    data['gap_efficiency'] = data['close_open_gap'] / np.where(data['daily_range'] == 0, 1, data['daily_range'])
    data['volatility_capture'] = np.abs(data['close_open_gap']) / np.where(data['daily_range'] == 0, 1, data['daily_range'])
    
    # Price Rejection Detection
    data['upper_shadow'] = data['high'] - np.maximum(data['open'], data['close'])
    data['lower_shadow'] = np.minimum(data['open'], data['close']) - data['low']
    data['upper_rejection'] = data['upper_shadow'] / np.where(data['daily_range'] == 0, 1, data['daily_range'])
    data['lower_rejection'] = data['lower_shadow'] / np.where(data['daily_range'] == 0, 1, data['daily_range'])
    
    # Multi-Period Efficiency Trends
    data['efficiency_score'] = (data['gap_efficiency'] + data['volatility_capture'] - 
                               data['upper_rejection'] - data['lower_rejection']) / 4
    
    # 3-day efficiency momentum
    data['eff_momentum_3d'] = data['efficiency_score'] - data['efficiency_score'].rolling(window=3, min_periods=1).mean()
    
    # Efficiency volatility (5-day std)
    data['eff_volatility_5d'] = data['efficiency_score'].rolling(window=5, min_periods=1).std()
    
    # Efficiency reversal patterns
    data['eff_zscore'] = (data['efficiency_score'] - data['efficiency_score'].rolling(window=20, min_periods=1).mean()) / \
                        np.where(data['efficiency_score'].rolling(window=20, min_periods=1).std() == 0, 1, 
                                data['efficiency_score'].rolling(window=20, min_periods=1).std())
    
    # Volume Distribution Analysis
    # Since we don't have intraday volume data, use daily patterns
    data['volume_ma_5'] = data['volume'].rolling(window=5, min_periods=1).mean()
    data['volume_ma_20'] = data['volume'].rolling(window=20, min_periods=1).mean()
    data['volume_trend'] = data['volume'] / data['volume_ma_5'] - 1
    data['volume_acceleration'] = data['volume_trend'] - data['volume_trend'].shift(1).fillna(0)
    
    # Volume volatility clustering
    data['volume_zscore'] = (data['volume'] - data['volume_ma_20']) / \
                           np.where(data['volume'].rolling(window=20, min_periods=1).std() == 0, 1,
                                   data['volume'].rolling(window=20, min_periods=1).std())
    
    # Price-Volume Divergence Measurement
    data['price_return'] = data['close'].pct_change()
    data['volume_change'] = data['volume'].pct_change()
    
    # Directional divergence detection
    data['bullish_div'] = ((data['price_return'] < 0) & (data['volume_change'] > 0)).astype(int)
    data['bearish_div'] = ((data['price_return'] > 0) & (data['volume_change'] < 0)).astype(int)
    data['neutral_extreme'] = ((np.abs(data['price_return']) < 0.005) & (np.abs(data['volume_zscore']) > 1)).astype(int)
    
    # Magnitude divergence assessment
    data['price_volume_ratio'] = np.abs(data['price_return']) / (np.abs(data['volume_change']) + 0.0001)
    
    # Multi-timeframe divergence patterns
    data['div_strength_3d'] = data['bullish_div'].rolling(window=3, min_periods=1).sum() - \
                             data['bearish_div'].rolling(window=3, min_periods=1).sum()
    data['div_strength_5d'] = data['bullish_div'].rolling(window=5, min_periods=1).sum() - \
                             data['bearish_div'].rolling(window=5, min_periods=1).sum()
    
    # Divergence persistence analysis
    data['consecutive_div'] = 0
    current_streak = 0
    for i in range(1, len(data)):
        if (data['bullish_div'].iloc[i] == 1 or data['bearish_div'].iloc[i] == 1):
            current_streak += 1
        else:
            current_streak = 0
        data.iloc[i, data.columns.get_loc('consecutive_div')] = current_streak
    
    # Signal Decay Modeling
    data['divergence_age'] = data['consecutive_div']
    data['decay_multiplier'] = np.exp(-data['divergence_age'] / 5.0)  # Half-life of 5 days
    
    # Factor Integration & Scoring
    # Divergence quality assessment
    data['efficiency_volume_mismatch'] = np.where(
        (data['efficiency_score'] > data['efficiency_score'].rolling(window=20, min_periods=1).mean()) & 
        (data['volume'] < data['volume_ma_20']), 1, 0
    ) - np.where(
        (data['efficiency_score'] < data['efficiency_score'].rolling(window=20, min_periods=1).mean()) & 
        (data['volume'] > data['volume_ma_20']), 1, 0
    )
    
    # Pattern consistency evaluation
    data['pattern_consistency'] = (np.sign(data['div_strength_3d']) == np.sign(data['div_strength_5d'])).astype(int)
    
    # Statistical significance
    data['div_zscore'] = (data['price_volume_ratio'] - data['price_volume_ratio'].rolling(window=20, min_periods=1).mean()) / \
                        np.where(data['price_volume_ratio'].rolling(window=20, min_periods=1).std() == 0, 1,
                                data['price_volume_ratio'].rolling(window=20, min_periods=1).std())
    
    # Decay-adjusted signal generation
    data['raw_divergence_strength'] = (
        data['bullish_div'] - data['bearish_div'] + 
        data['efficiency_volume_mismatch'] +
        np.sign(data['div_strength_3d']) * 0.5 +
        np.sign(data['div_strength_5d']) * 0.3
    )
    
    data['volume_confirmation_weight'] = np.where(
        np.abs(data['volume_zscore']) > 1, 1.2,
        np.where(np.abs(data['volume_zscore']) > 0.5, 1.0, 0.8)
    )
    
    # Efficiency context integration
    data['efficiency_regime'] = np.where(
        data['eff_zscore'] > 1, 1.2,  # High efficiency
        np.where(data['eff_zscore'] < -1, 0.8, 1.0)  # Low efficiency
    )
    
    # Final factor calculation
    data['alpha_factor'] = (
        data['raw_divergence_strength'] * 
        data['decay_multiplier'] * 
        data['volume_confirmation_weight'] * 
        data['efficiency_regime'] * 
        data['pattern_consistency'] *
        (1 + 0.1 * np.clip(np.abs(data['div_zscore']), 0, 2))
    )
    
    return data['alpha_factor']
