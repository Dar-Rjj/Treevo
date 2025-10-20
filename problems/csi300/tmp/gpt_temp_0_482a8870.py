import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Price Efficiency Framework
    # 1-day efficiency
    data['efficiency_1d'] = (data['close'] - data['open']) / (data['high'] - data['low']).replace(0, np.nan)
    
    # 3-day cumulative efficiency
    data['efficiency_3d_cum'] = data['efficiency_1d'].rolling(window=3, min_periods=1).sum()
    
    # 5-day efficiency persistence
    data['efficiency_5d_persistence'] = (data['efficiency_1d'].rolling(window=5, min_periods=1) > 0).sum() / 5
    
    # Efficiency acceleration analysis
    data['efficiency_momentum'] = data['efficiency_3d_cum'] - data['efficiency_1d']
    data['efficiency_consistency'] = data['efficiency_1d'].rolling(window=5, min_periods=1).std()
    data['efficiency_reversal'] = (np.sign(data['efficiency_1d']) != np.sign(data['efficiency_3d_cum'])).astype(float)
    
    # Range-normalized price movement
    data['range_adjusted_close'] = data['efficiency_1d']
    data['range_persistence'] = (data['high'] - data['low']) / (data['high'] - data['low']).rolling(window=5, min_periods=1).mean().shift(1)
    data['gap_efficiency'] = (data['open'] - data['close'].shift(1)) / (data['high'] - data['low']).replace(0, np.nan)
    
    # Volume Distribution Analysis
    # Volume concentration metrics
    data['volume_clustering'] = data['volume'] / data['volume'].rolling(window=5, min_periods=1).mean().shift(1)
    data['volume_persistence'] = data['volume'] / data['volume'].shift(3)
    data['volume_stability'] = data['volume'] / data['volume'].rolling(window=10, min_periods=1).mean().shift(1)
    
    # Volume-price efficiency relationship
    data['volume_weighted_efficiency'] = data['efficiency_1d'] * data['volume_clustering']
    
    # Calculate rolling correlation between volume and efficiency
    def rolling_corr(x, y, window):
        return x.rolling(window=window).corr(y)
    
    data['volume_efficiency_corr'] = rolling_corr(data['volume'], data['efficiency_1d'], 5)
    data['volume_acceleration_efficiency'] = data['volume_persistence'] * data['efficiency_momentum']
    
    # Dynamic Regime Detection
    # Market state classification
    data['ma_5'] = data['close'].rolling(window=5, min_periods=1).mean()
    data['ma_20'] = data['close'].rolling(window=20, min_periods=1).mean()
    data['ma_10'] = data['close'].rolling(window=10, min_periods=1).mean()
    
    data['trend_regime'] = (data['ma_5'] > data['ma_20']).astype(float)
    data['range_bound_regime'] = ((data['high'] - data['low']) < (data['high'] - data['low']).rolling(window=10, min_periods=1).mean()).astype(float)
    data['transition_regime'] = (abs(data['close'] - data['ma_10']) / data['ma_10'] > 0.02).astype(float)
    
    # Volatility regime assessment
    data['volatility_expansion'] = (data['high'] - data['low']) / (data['high'] - data['low']).rolling(window=10, min_periods=1).mean()
    data['volatility_clustering'] = ((data['high'] - data['low']) > (data['high'] - data['low']).rolling(window=10, min_periods=1).mean()).rolling(window=5, min_periods=1).sum() / 5
    data['volatility_persistence'] = (data['high'] - data['low']).rolling(window=5, min_periods=1).std() / (data['high'] - data['low']).rolling(window=10, min_periods=1).mean()
    
    # Efficiency regime identification
    data['high_efficiency'] = (abs(data['efficiency_1d']) > 0.7).astype(float)
    data['low_efficiency'] = (abs(data['efficiency_1d']) < 0.3).astype(float)
    
    # Cross-Regime Signal Generation
    # Trend regime signals
    data['trend_efficiency'] = data['efficiency_1d'] * data['trend_regime']
    data['volume_confirmed_trend'] = data['volume_clustering'] * data['trend_efficiency']
    data['trend_acceleration'] = data['efficiency_momentum'] * data['trend_regime']
    
    # Range-bound regime signals
    data['range_efficiency'] = data['efficiency_1d'] * data['range_persistence']
    data['volume_compression'] = data['volume_stability'] * data['range_efficiency']
    data['breakout_potential'] = data['volume_clustering'] * data['range_efficiency']
    
    # Transition regime signals
    data['regime_change_momentum'] = data['efficiency_momentum'] * data['volatility_expansion']
    data['volume_regime_confirmation'] = data['volume_acceleration_efficiency'] * data['regime_change_momentum']
    data['transition_efficiency'] = data['efficiency_1d'] * data['volatility_expansion']
    
    # Multi-Timeframe Signal Integration
    # Short-term signal weighting
    data['signal_strength_1d'] = data['efficiency_1d'] * data['volume_clustering']
    data['signal_persistence_3d'] = data['efficiency_3d_cum'] * data['volume_persistence']
    data['signal_consistency_5d'] = data['efficiency_consistency'] * data['volume_stability']
    
    # Signal convergence detection
    data['multi_timeframe_alignment'] = data['signal_strength_1d'] * data['signal_persistence_3d']
    data['signal_acceleration'] = data['signal_strength_1d'] - data['signal_persistence_3d']
    data['signal_divergence'] = abs(data['signal_strength_1d']) / (abs(data['signal_persistence_3d']).replace(0, np.nan))
    
    # Regime-adaptive signal combination
    data['trend_optimized'] = data['trend_acceleration'] * data['volume_confirmed_trend']
    data['range_optimized'] = data['range_efficiency'] * data['volume_compression']
    data['transition_optimized'] = data['transition_efficiency'] * data['volume_regime_confirmation']
    
    # Dynamic Alpha Synthesis
    # Regime-specific factor calculation
    data['trend_regime_factor'] = data['trend_optimized'] * data['multi_timeframe_alignment']
    data['range_regime_factor'] = data['range_optimized'] * data['signal_consistency_5d']
    data['transition_regime_factor'] = data['transition_optimized'] * data['signal_acceleration']
    
    # Apply regime-specific weighting
    data['final_alpha'] = (
        data['trend_regime'] * data['trend_regime_factor'] +
        data['range_bound_regime'] * data['range_regime_factor'] +
        data['transition_regime'] * data['transition_regime_factor']
    )
    
    # Incorporate multi-timeframe signal strength
    data['final_alpha'] *= (data['signal_strength_1d'] + data['signal_persistence_3d'] + data['signal_consistency_5d']) / 3
    
    # Adjust for volume distribution characteristics
    data['final_alpha'] *= data['volume_clustering']
    
    # Validate with efficiency persistence metrics
    data['final_alpha'] *= data['efficiency_5d_persistence']
    
    return data['final_alpha']
