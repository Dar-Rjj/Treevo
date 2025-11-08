import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Momentum Acceleration & Regime Adaptation
    # Multi-Timeframe Momentum Calculation
    data['ultra_short_momentum'] = data['close'] / data['close'].shift(1) - 1
    data['short_term_momentum'] = data['close'] / data['close'].shift(3) - 1
    data['medium_term_momentum'] = data['close'] / data['close'].shift(7) - 1
    data['long_term_momentum'] = data['close'] / data['close'].shift(15) - 1
    
    # Momentum Acceleration Signals
    data['ultra_to_short_acc'] = data['short_term_momentum'] - data['ultra_short_momentum']
    data['short_to_medium_acc'] = data['medium_term_momentum'] - data['short_term_momentum']
    data['medium_to_long_acc'] = data['long_term_momentum'] - data['medium_term_momentum']
    data['acceleration_convergence'] = np.sign(data['ultra_to_short_acc'] * data['short_to_medium_acc'] * data['medium_to_long_acc']) * \
                                     np.abs(data['ultra_to_short_acc'] * data['short_to_medium_acc'] * data['medium_to_long_acc']) ** (1/3)
    
    # Volatility Regime Detection
    data['current_daily_range'] = (data['high'] - data['low']) / data['close']
    data['volatility_baseline'] = data['current_daily_range'].rolling(window=10).mean()
    data['volatility_regime'] = data['current_daily_range'] / data['volatility_baseline']
    data['regime_class'] = np.where(data['volatility_regime'] > 1.2, 'high', 
                                  np.where(data['volatility_regime'] < 0.8, 'low', 'normal'))
    
    # Regime-Adaptive Momentum
    data['raw_momentum_score'] = np.sign(data['ultra_short_momentum'] * data['short_term_momentum'] * 
                                       data['medium_term_momentum'] * data['long_term_momentum']) * \
                               np.abs(data['ultra_short_momentum'] * data['short_term_momentum'] * 
                                    data['medium_term_momentum'] * data['long_term_momentum']) ** (1/4)
    data['acceleration_adjusted_momentum'] = data['raw_momentum_score'] * (1 + data['acceleration_convergence'])
    data['regime_scaled_momentum'] = data['acceleration_adjusted_momentum'] / data['volatility_regime']
    
    # Volume-Price Efficiency Convergence
    # Price Efficiency Metrics
    data['opening_efficiency'] = (data['open'] - data['close'].shift(1)) / (data['high'] - data['low'])
    data['intraday_efficiency'] = (data['close'] - data['open']) / (data['high'] - data['low'])
    data['range_utilization'] = (data['close'] - data['low']) / (data['high'] - data['low'])
    data['price_persistence'] = (data['close'] - data['close'].shift(1)) / (data['high'] - data['low'])
    
    # Volume Anomaly Detection
    data['volume_spike'] = data['volume'] / data['volume'].rolling(window=5).mean()
    data['volume_trend'] = data['volume'] / data['volume'].shift(3)
    data['volume_persistence'] = data['volume'] / data['volume'].rolling(window=10).mean()
    data['volume_volatility'] = data['volume'].rolling(window=5).std() / data['volume'].rolling(window=5).mean()
    
    # Efficiency-Volume Integration
    data['composite_price_efficiency'] = np.sign(data['opening_efficiency'] * data['intraday_efficiency'] * 
                                               data['range_utilization'] * data['price_persistence']) * \
                                      np.abs(data['opening_efficiency'] * data['intraday_efficiency'] * 
                                           data['range_utilization'] * data['price_persistence']) ** (1/4)
    data['composite_volume_anomaly'] = np.sign(data['volume_spike'] * data['volume_trend'] * 
                                             data['volume_persistence']) * \
                                    np.abs(data['volume_spike'] * data['volume_trend'] * 
                                         data['volume_persistence']) ** (1/3)
    data['efficiency_divergence'] = data['composite_price_efficiency'] / data['composite_volume_anomaly']
    
    # Regime-Contextual Efficiency
    data['high_vol_efficiency'] = data['efficiency_divergence'] * data['volatility_regime']
    data['low_vol_efficiency'] = data['efficiency_divergence'] / data['volatility_regime']
    data['adaptive_efficiency'] = np.where(data['volatility_regime'] > 1.2, data['high_vol_efficiency'], data['low_vol_efficiency'])
    
    # Multi-Timeframe Signal Convergence
    # Ultra-short Horizon (1-3 days)
    data['ultra_price_momentum'] = data['close'] / data['close'].shift(2) - 1
    data['ultra_volume_acc'] = data['volume'] / data['volume'].shift(2)
    data['ultra_range_efficiency'] = (data['close'] - data['close'].shift(1)) / (data['high'] - data['low'])
    data['ultra_short_convergence'] = np.sign(data['ultra_price_momentum'] * data['ultra_volume_acc'] * 
                                            (1 + data['ultra_range_efficiency'])) * \
                                    np.abs(data['ultra_price_momentum'] * data['ultra_volume_acc'] * 
                                         (1 + data['ultra_range_efficiency'])) ** (1/3)
    
    # Short-term Horizon (3-7 days)
    data['short_price_trend'] = data['close'] / data['close'].shift(5) - 1
    data['short_volume_persistence'] = data['volume'] / data['volume'].rolling(window=5).mean()
    data['short_vol_adaptation'] = (data['high'] - data['low']) / (data['high'] - data['low']).rolling(window=5).mean()
    data['short_term_convergence'] = np.sign(data['short_price_trend'] * data['short_volume_persistence'] * 
                                           (1 + data['short_vol_adaptation'])) * \
                                   np.abs(data['short_price_trend'] * data['short_volume_persistence'] * 
                                        (1 + data['short_vol_adaptation'])) ** (1/3)
    
    # Medium-term Horizon (7-15 days)
    data['medium_price_momentum'] = data['close'] / data['close'].shift(10) - 1
    data['medium_volume_stability'] = 1 / (data['volume'].rolling(window=10).std() / data['volume'].rolling(window=10).mean())
    
    # Calculate efficiency persistence
    efficiency_persistence = []
    for i in range(len(data)):
        if i >= 9:
            window_data = data.iloc[i-9:i+1]
            efficiencies = []
            for j in range(1, len(window_data)):
                if window_data.iloc[j]['high'] != window_data.iloc[j]['low']:
                    eff = (window_data.iloc[j]['close'] - window_data.iloc[j-1]['close']) / \
                          (window_data.iloc[j]['high'] - window_data.iloc[j]['low'])
                    efficiencies.append(eff)
            if efficiencies:
                efficiency_persistence.append(np.mean(efficiencies))
            else:
                efficiency_persistence.append(np.nan)
        else:
            efficiency_persistence.append(np.nan)
    
    data['medium_efficiency_persistence'] = efficiency_persistence
    data['medium_term_convergence'] = np.sign(data['medium_price_momentum'] * data['medium_volume_stability'] * 
                                            (1 + data['medium_efficiency_persistence'])) * \
                                    np.abs(data['medium_price_momentum'] * data['medium_volume_stability'] * 
                                         (1 + data['medium_efficiency_persistence'])) ** (1/3)
    
    # Multi-Timeframe Integration
    data['horizon_alignment'] = data['ultra_short_convergence'] * data['short_term_convergence'] * data['medium_term_convergence']
    data['convergence_acceleration'] = (data['short_term_convergence'] / data['ultra_short_convergence']) * \
                                     (data['medium_term_convergence'] / data['short_term_convergence'])
    data['final_convergence_score'] = data['horizon_alignment'] * (1 + data['convergence_acceleration'])
    
    # Cross-Factor Synthesis
    # Momentum-Efficiency Alignment
    data['momentum_regime'] = np.where(data['acceleration_convergence'] > 0, 'accelerating', 'decelerating')
    data['efficiency_regime'] = np.where(data['efficiency_divergence'] > 1, 'efficient', 'inefficient')
    data['alignment_score'] = data['regime_scaled_momentum'] * data['adaptive_efficiency']
    
    # Multi-Timeframe Confirmation
    data['short_term_confirmation'] = np.sign(data['ultra_short_convergence']) * np.sign(data['short_term_convergence'])
    data['medium_term_confirmation'] = np.sign(data['short_term_convergence']) * np.sign(data['medium_term_convergence'])
    data['confirmation_strength'] = (1 + data['short_term_confirmation']) * (1 + data['medium_term_confirmation'])
    
    # Final Alpha Synthesis
    data['core_alpha'] = data['alignment_score'] * data['final_convergence_score']
    data['confirmation_adjustment'] = data['core_alpha'] * data['confirmation_strength']
    data['final_alpha'] = data['confirmation_adjustment'] / data['volatility_regime']
    
    return data['final_alpha']
