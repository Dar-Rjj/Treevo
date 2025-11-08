import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Initialize result series
    result = pd.Series(index=data.index, dtype=float)
    
    # Calculate basic price changes and ratios
    data['prev_close'] = data['close'].shift(1)
    data['prev_high'] = data['high'].shift(1)
    data['prev_low'] = data['low'].shift(1)
    data['prev_volume'] = data['volume'].shift(1)
    
    # Cross-Asset Pressure Transmission (simplified - using market data only)
    data['market_return'] = (data['close'] / data['prev_close'] - 1).rolling(window=20).mean()
    data['sector_pressure'] = (data['close'] / data['prev_close'] - 1) - data['market_return']
    
    # Microstructure Momentum Patterns
    data['gap_momentum'] = (data['open'] - data['prev_close']) / data['prev_close']
    data['opening_absorption'] = abs(data['open'] - data['prev_close']) / (data['high'] - data['low']).replace(0, np.nan)
    data['opening_momentum_score'] = data['gap_momentum'] * (1 - data['opening_absorption'])
    
    # Intraday Momentum Acceleration
    data['morning_momentum'] = (data['prev_high'] - data['open'].shift(1)) / data['open'].shift(1)
    data['afternoon_momentum'] = (data['prev_close'] - data['prev_low']) / data['prev_low']
    data['intraday_pattern'] = data['morning_momentum'] - data['afternoon_momentum']
    
    # Volume-Volatility Convergence
    data['volume_ma_20'] = data['volume'].rolling(window=20).mean()
    data['volume_std_20'] = data['volume'].rolling(window=20).std()
    data['volume_zscore'] = (data['volume'] - data['volume_ma_20']) / data['volume_std_20'].replace(0, np.nan)
    data['volume_momentum'] = data['volume'] / data['prev_volume'] - 1
    
    data['range_compression'] = ((data['high'] - data['low']) / 
                                (data['prev_high'] - data['prev_low'])).replace([np.inf, -np.inf], np.nan)
    data['volatility_momentum'] = data['range_compression'] - 1
    data['volume_volatility_ratio'] = data['volume_momentum'] / data['volatility_momentum'].replace(0, np.nan)
    
    # Price Rejection Analysis
    data['upper_shadow'] = (data['high'] - np.maximum(data['open'], data['close'])) / (data['high'] - data['low']).replace(0, np.nan)
    data['upper_rejection_score'] = data['upper_shadow'] * data['volume']
    data['rejection_momentum'] = data['upper_shadow'] - data['upper_shadow'].shift(1)
    
    data['lower_shadow'] = (np.minimum(data['open'], data['close']) - data['low']) / (data['high'] - data['low']).replace(0, np.nan)
    data['support_score'] = data['lower_shadow'] * data['volume']
    data['support_momentum'] = data['lower_shadow'] - data['lower_shadow'].shift(1)
    
    # Cross-Timeframe Momentum Alignment
    data['ultra_short_momentum'] = data['close'] / data['open'] - 1
    data['short_term_momentum'] = data['close'] / data['close'].shift(2) - 1
    data['medium_term_momentum'] = data['close'] / data['close'].shift(5) - 1
    
    data['momentum_alignment'] = (np.sign(data['ultra_short_momentum']) * 
                                 np.sign(data['short_term_momentum']) * 
                                 np.sign(data['medium_term_momentum']))
    
    data['acceleration_pattern'] = ((data['short_term_momentum'] - data['ultra_short_momentum']) * 
                                   (data['medium_term_momentum'] - data['short_term_momentum']))
    data['momentum_strength'] = (abs(data['ultra_short_momentum']) + 
                                abs(data['short_term_momentum']) + 
                                abs(data['medium_term_momentum']))
    data['convergence_score'] = data['acceleration_pattern'] * data['momentum_strength']
    
    # Market Microstructure Regimes
    data['volume_ma_10'] = data['volume'].rolling(window=10).mean()
    data['high_volume_concentration'] = (data['volume'] > 1.5 * data['volume_ma_10']).astype(int)
    data['low_volume_concentration'] = (data['volume'] < 0.7 * data['volume_ma_10']).astype(int)
    data['volume_regime'] = data['high_volume_concentration'] - data['low_volume_concentration']
    
    # Cross-Asset Momentum Transmission
    data['sector_momentum'] = data['sector_pressure'] * data['volume']
    data['cross_sector_correlation'] = data['sector_pressure'].rolling(window=10).corr(data['close'] / data['prev_close'] - 1)
    data['sector_flow'] = data['sector_momentum'] * data['cross_sector_correlation']
    
    # Microstructure Factor Construction
    data['base_momentum'] = data['opening_momentum_score'] * data['intraday_pattern']
    data['volume_enhanced'] = data['base_momentum'] * data['volume_volatility_ratio']
    data['rejection_adjusted'] = data['volume_enhanced'] * (1 + data['rejection_momentum'] - data['support_momentum'])
    
    data['external_momentum'] = data['sector_flow']
    data['market_aligned'] = data['external_momentum'] * data['momentum_alignment']
    data['regime_weighted'] = data['market_aligned'] * data['volume_regime']
    
    # Advanced Pattern Recognition
    for i in range(len(data)):
        if i >= 3:
            window_data = data.iloc[i-3:i+1]
            data.loc[data.index[i], 'consecutive_rejection'] = (window_data['upper_rejection_score'] > 0.1).sum()
            data.loc[data.index[i], 'support_building'] = (window_data['support_score'] > 0.1).sum()
    
    data['pattern_strength'] = data['consecutive_rejection'] - data['support_building']
    
    # Volume pattern analysis
    data['volume_spike'] = (data['volume'] > 2 * data['volume_ma_20']).astype(int)
    for i in range(len(data)):
        if i >= 5:
            window_data = data.iloc[i-5:i+1]
            data.loc[data.index[i], 'volume_cluster'] = window_data['volume_spike'].sum()
    
    # Calculate volume trend (simplified)
    data['volume_trend'] = data['volume'].rolling(window=5).apply(
        lambda x: (x.iloc[-1] - x.iloc[0]) / x.iloc[0] if x.iloc[0] != 0 else 0, raw=False
    )
    data['volume_pattern'] = data['volume_cluster'] * data['volume_trend']
    
    # Final Factor Integration
    data['primary_factor'] = data['rejection_adjusted'] * data['regime_weighted']
    data['pattern_enhanced'] = data['primary_factor'] * (1 + data['pattern_strength'])
    data['volume_confirmed'] = data['pattern_enhanced'] * data['volume_pattern']
    data['cross_timeframe_validated'] = data['volume_confirmed'] * data['convergence_score']
    
    # Alpha Generation
    data['microstructure_alpha'] = data['cross_timeframe_validated']
    data['risk_adjustment'] = data['microstructure_alpha'] * (1 - abs(data['volume_zscore']))
    
    # Session momentum (simplified - using time index)
    if hasattr(data.index, 'hour'):
        data['session_momentum'] = np.where(data.index.hour < 12, 
                                           data['opening_momentum_score'], 
                                           data['intraday_pattern'])
    else:
        # Fallback if no hour information
        data['session_momentum'] = data['opening_momentum_score']
    
    data['final_alpha'] = data['risk_adjustment'] * data['session_momentum']
    
    # Fill NaN values with 0
    result = data['final_alpha'].fillna(0)
    
    return result
