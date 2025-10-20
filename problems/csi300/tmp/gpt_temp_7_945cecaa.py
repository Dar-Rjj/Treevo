import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Multi-timeframe momentum calculation
    data['short_momentum'] = data['close'].pct_change(periods=3)
    data['medium_momentum'] = data['close'].pct_change(periods=10)
    data['momentum_acceleration'] = (data['short_momentum'] - data['medium_momentum']) / 3
    
    # Volume acceleration analysis
    data['volume_momentum'] = data['volume'] / data['volume'].shift(5) - 1
    data['volume_2day_change'] = data['volume'].pct_change(periods=2)
    data['volume_5day_change'] = data['volume'].pct_change(periods=5)
    data['volume_acceleration'] = (data['volume_2day_change'] - data['volume_5day_change']) / 2
    
    # Volatility-Regime Adaptive Detection
    data['daily_range_pct'] = (data['high'] - data['low']) / data['close']
    data['volatility_5day'] = data['close'].pct_change().rolling(window=5).std()
    data['volatility_10day'] = data['close'].pct_change().rolling(window=10).std()
    data['range_efficiency'] = data['daily_range_pct'] / data['volatility_20day']
    data['range_5day_avg'] = data['daily_range_pct'].rolling(window=5).mean()
    data['range_comparison'] = data['daily_range_pct'] / data['range_5day_avg']
    
    # Volatility regime classification
    data['volatility_ratio'] = data['volatility_5day'] / data['volatility_10day']
    data['high_vol_regime'] = data['volatility_5day'] > data['volatility_10day']
    
    # Divergence Pattern Integration
    data['bullish_divergence'] = -data['momentum_acceleration'] * data['volume_momentum']
    data['bearish_divergence'] = data['momentum_acceleration'] * data['volume_momentum']
    
    # Intraday pressure validation
    data['intraday_pressure'] = (data['close'] - data['open']) / (data['high'] - data['low']).replace(0, np.nan)
    data['pressure_3day_sum'] = data['intraday_pressure'].rolling(window=3).sum()
    data['pressure_5day_sum'] = data['intraday_pressure'].rolling(window=5).sum()
    data['volume_weighted_pressure'] = data['intraday_pressure'] * data['volume_acceleration']
    
    # Synchronized Composite Signal Generation
    data['momentum_volatility_sync'] = data['momentum_acceleration'] * data['range_efficiency']
    data['volume_sensitivity_weight'] = 1 + data['volume_acceleration'].abs()
    data['sync_with_volume'] = data['momentum_volatility_sync'] * data['volume_sensitivity_weight']
    
    # Divergence enhancement
    data['divergence_strength'] = np.where(data['momentum_acceleration'] > 0, 
                                         data['bearish_divergence'], 
                                         data['bullish_divergence'])
    data['enhanced_sync'] = data['sync_with_volume'] * data['divergence_strength'] * data['intraday_pressure']
    
    # VWAP calculation for high volatility regime
    data['vwap_3day'] = (data['close'] * data['volume']).rolling(window=3).sum() / data['volume'].rolling(window=3).sum()
    data['vwap_efficiency'] = (data['close'] - data['vwap_3day']) / data['close']
    
    # Regime-Adaptive Alpha Output
    # High volatility regime processing
    high_vol_factor = (data['enhanced_sync'] * 
                      data['divergence_strength'] * 
                      data['volume_weighted_pressure'] * 
                      data['vwap_efficiency'])
    
    # Low volatility regime processing
    low_vol_factor = (data['enhanced_sync'] * 
                     data['divergence_strength'] * 
                     data['range_efficiency'] * 
                     data['pressure_3day_sum'])
    
    # Final alpha generation with regime selection
    alpha = np.where(data['high_vol_regime'], high_vol_factor, low_vol_factor)
    alpha = alpha * data['close'].pct_change().abs()  # Magnitude scaling
    
    # Return as pandas Series
    return pd.Series(alpha, index=data.index)
