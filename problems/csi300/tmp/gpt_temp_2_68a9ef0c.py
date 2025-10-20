import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    data = df.copy()
    
    # Multi-Frequency Momentum Divergence
    data['short_momentum'] = data['close'] / data['close'].shift(1) - 1
    data['medium_momentum'] = data['close'] / data['close'].shift(5) - 1
    data['long_momentum'] = data['close'] / data['close'].shift(10) - 1
    data['momentum_acceleration'] = (data['short_momentum'] - data['medium_momentum']) / 4
    data['momentum_divergence'] = (data['short_momentum'] + data['medium_momentum'] + data['long_momentum']) / 3
    
    # Volume Synchronization Assessment
    data['volume_direction_alignment'] = np.sign(data['close'] - data['close'].shift(1)) * np.sign(data['volume'] - data['volume'].shift(1))
    data['volume_momentum'] = data['volume'] / data['volume'].shift(5) - 1
    
    # Volume persistence calculation
    volume_change_sign = np.sign(data['volume'] - data['volume'].shift(1))
    volume_persistence = pd.Series(index=data.index, dtype=float)
    for i in range(len(data)):
        if i < 3:
            volume_persistence.iloc[i] = 0
        else:
            window = volume_change_sign.iloc[i-2:i+1]
            if len(window[window == window.iloc[-1]]) == 3:
                volume_persistence.iloc[i] = 3
            elif len(window[window == window.iloc[-1]]) >= 2:
                volume_persistence.iloc[i] = 2
            else:
                volume_persistence.iloc[i] = 1
    data['volume_persistence'] = volume_persistence
    
    data['intraday_pressure'] = (data['close'] - data['open']) / (data['high'] - data['low']).replace(0, np.nan)
    data['volume_momentum_coherence'] = data['volume_direction_alignment'] * data['momentum_divergence']
    
    # Price Efficiency Context
    data['range_efficiency'] = (data['high'] - data['low']) / data['close']
    data['intraday_efficiency'] = (data['close'] - data['open']) / (data['high'] - data['low']).replace(0, np.nan)
    data['overnight_efficiency'] = (data['open'] - data['close'].shift(1)) / (data['high'] - data['low']).replace(0, np.nan)
    data['total_price_efficiency'] = data['intraday_efficiency'] + data['overnight_efficiency']
    data['efficiency_consistency'] = data['total_price_efficiency'].rolling(window=5, min_periods=3).std()
    
    # Volatility Regime Classification
    data['range_volatility'] = (data['high'] - data['low']) / data['close']
    data['price_volatility'] = data['close'].rolling(window=5, min_periods=3).std()
    data['volatility_ratio'] = data['close'].rolling(window=5, min_periods=3).std() / data['close'].rolling(window=10, min_periods=5).std()
    data['avg_range_volatility'] = data['range_volatility'].rolling(window=20, min_periods=10).mean()
    data['high_volatility'] = (data['volatility_ratio'] > 1) | (data['range_volatility'] > data['avg_range_volatility'])
    
    # Regime-Adaptive Synchronization
    # High Volatility Synchronization
    data['range_breakout'] = (data['close'] - data['low']) / (data['high'] - data['low']).replace(0, np.nan)
    data['volume_significance'] = data['volume'] / data['volume'].rolling(window=10, min_periods=5).mean()
    data['high_vol_core_sync'] = -data['momentum_acceleration'] * data['volume_momentum'] * data['intraday_pressure']
    data['high_vol_enhanced'] = data['high_vol_core_sync'] * data['range_breakout'] * data['volume_significance']
    
    # Low Volatility Synchronization
    data['trend_persistence'] = np.sign(data['close'] - data['close'].shift(1)).rolling(window=5, min_periods=3).sum()
    data['liquidity_quality'] = data['amount'] / data['volume'].replace(0, np.nan)
    data['low_vol_core_sync'] = data['volume_momentum_coherence'] * data['total_price_efficiency']
    data['low_vol_enhanced'] = data['low_vol_core_sync'] * data['trend_persistence'] * data['liquidity_quality']
    
    # Regime Selection
    data['regime_enhanced_factor'] = np.where(
        data['high_volatility'],
        data['high_vol_enhanced'],
        data['low_vol_enhanced']
    )
    
    # Final Alpha Generation
    data['persistence_adjusted'] = data['regime_enhanced_factor'] * data['volume_persistence']
    data['efficiency_weighted'] = data['persistence_adjusted'] * data['efficiency_consistency']
    data['final_alpha'] = data['efficiency_weighted'] * np.abs(data['momentum_divergence'])
    
    return data['final_alpha']
