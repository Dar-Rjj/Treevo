import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate rolling windows for various timeframes
    for window in [3, 5, 7]:
        data[f'high_roll_{window}'] = data['high'].rolling(window=window, min_periods=1).max()
        data[f'low_roll_{window}'] = data['low'].rolling(window=window, min_periods=1).min()
    
    # Cross-Fractal Efficiency Framework
    # Using market and sector proxies from the data (assuming 'amount' as proxy for market, 'volume' as proxy for sector)
    data['stock_vs_sector_eff'] = (abs(data['close'] - data['close'].shift(5)) / (data['high'] - data['low'])) * \
                                 ((data['close'] / data['close'].shift(5) - 1) / (data['volume'] / data['volume'].shift(5) - 1 + 1e-8))
    
    data['stock_vs_market_eff'] = (abs(data['close'] - data['close'].shift(5)) / (data['high'] - data['low'])) * \
                                 ((data['close'] / data['close'].shift(5) - 1) / (data['amount'] / data['amount'].shift(5) - 1 + 1e-8))
    
    data['cross_fractal_divergence'] = data['stock_vs_sector_eff'] - data['stock_vs_market_eff']
    
    # Multi-Timeframe Efficiency Dynamics
    data['ultra_short_cross_eff'] = (abs(data['close'] - data['close'].shift(3)) / 
                                   (data['high_roll_3'] - data['low_roll_3'])) * np.sign(data['cross_fractal_divergence'])
    
    data['medium_term_cross_eff'] = (abs(data['close'] - data['close'].shift(7)) / 
                                   (data['high_roll_7'] - data['low_roll_7'])) * np.sign(data['cross_fractal_divergence'])
    
    data['cross_efficiency_momentum'] = (data['medium_term_cross_eff'] - data['ultra_short_cross_eff']) * \
                                      np.sign(data['close'] - data['close'].shift(1))
    
    # Volume-Enhanced Cross Efficiency
    data['cross_eff_volume_ratio'] = data['cross_fractal_divergence'] * (data['volume'] / data['volume'].shift(5) - 1)
    data['cross_eff_volume_momentum'] = data['cross_eff_volume_ratio'] * np.sign(data['volume'] - data['volume'].shift(1))
    
    # Cross Efficiency Persistence
    def calculate_persistence(series):
        current_sign = np.sign(series.iloc[-1])
        return sum(np.sign(series.iloc[-5:]) == current_sign)
    
    data['cross_efficiency_persistence'] = data['cross_fractal_divergence'].rolling(window=5).apply(calculate_persistence, raw=False)
    
    # Asymmetric Cross-Fractal Dynamics
    data['cross_fractal_vol_asymmetry'] = ((data['high_roll_5'] - data['close'].shift(4)) / 
                                         (data['close'].shift(4) - data['low_roll_5'] + 1e-8)) * \
                                         np.sign(data['cross_fractal_divergence'])
    
    # Cross-Fractal Momentum Asymmetry
    returns = data['close'].pct_change()
    pos_returns = returns.rolling(7).apply(lambda x: x[x > 0].sum(), raw=False)
    neg_returns = returns.rolling(7).apply(lambda x: x[x < 0].sum(), raw=False)
    data['cross_fractal_momentum_asymmetry'] = (pos_returns / (pos_returns + abs(neg_returns) + 1e-8)) * \
                                             np.sign(data['cross_fractal_divergence'])
    
    data['asymmetric_cross_fractal_pressure'] = data['cross_fractal_vol_asymmetry'] * \
                                              data['cross_fractal_momentum_asymmetry'] * \
                                              (data['close'] - data['close'].shift(1)) / (data['high'] - data['low'] + 1e-8)
    
    # Microstructure Asymmetry
    data['opening_cross_fractal_asymmetry'] = ((data['open'] - data['low']) - (data['high'] - data['open'])) * \
                                            np.sign(data['cross_fractal_divergence']) * \
                                            abs(data['open'] - data['close'].shift(1)) / (data['high'] - data['low'] + 1e-8)
    
    data['closing_cross_fractal_asymmetry'] = ((data['close'] - data['low']) - (data['high'] - data['close'])) * \
                                            np.sign(data['cross_fractal_divergence']) * \
                                            abs(data['close'] - data['open']) / (data['high'] - data['low'] + 1e-8)
    
    data['cross_fractal_session_asymmetry'] = data['opening_cross_fractal_asymmetry'] - data['closing_cross_fractal_asymmetry']
    
    # Volume Asymmetry Dynamics
    up_volume = data['volume'].where(data['close'] > data['close'].shift(1), 0)
    down_volume = data['volume'].where(data['close'] < data['close'].shift(1), 0)
    
    data['cross_volume_asymmetry_momentum'] = (up_volume.rolling(7).sum() / (down_volume.rolling(7).sum() + 1e-8)) * \
                                            np.sign(data['cross_fractal_divergence'])
    
    data['cross_volume_fractal_shock'] = np.where(data['volume'] > 1.8 * data['volume'].shift(1), -1,
                                                np.where(data['volume'] < 0.6 * data['volume'].shift(1), 1, 0))
    
    data['cross_volume_fractal_alignment'] = data['cross_volume_asymmetry_momentum'] * \
                                           data['cross_volume_fractal_shock'] * \
                                           (data['close'] - data['open']) / (data['high'] - data['low'] + 1e-8)
    
    # Cross-Fractal Regime Classification
    data['high_efficiency_regime'] = (abs(data['cross_fractal_divergence']) > 0.8 * (data['high'] - data['low'])) & \
                                   (data['cross_efficiency_momentum'] > 0)
    
    data['moderate_efficiency_regime'] = (abs(data['cross_fractal_divergence']) >= 0.3 * (data['high'] - data['low'])) & \
                                       (abs(data['cross_fractal_divergence']) <= 0.8 * (data['high'] - data['low']))
    
    data['low_efficiency_regime'] = (abs(data['cross_fractal_divergence']) < 0.3 * (data['high'] - data['low'])) | \
                                  (data['cross_efficiency_momentum'] < 0)
    
    data['high_volume_regime'] = (data['volume'] / data['volume'].shift(5) > 1.5) & \
                               (data['cross_volume_fractal_shock'] == -1)
    
    data['low_volume_regime'] = (data['volume'] / data['volume'].shift(5) < 0.7) | \
                              (data['cross_volume_fractal_shock'] == 1)
    
    # Regime Multipliers
    data['regime_multiplier'] = 1.0  # Default normal regime
    
    data.loc[data['high_efficiency_regime'] & data['high_volume_regime'], 'regime_multiplier'] = 2.2
    data.loc[data['high_efficiency_regime'] & data['low_volume_regime'], 'regime_multiplier'] = 1.8
    data.loc[data['low_efficiency_regime'] & data['high_volume_regime'], 'regime_multiplier'] = 1.4
    data.loc[data['low_efficiency_regime'] & data['low_volume_regime'], 'regime_multiplier'] = 0.6
    
    # Integrated Cross-Fractal Signal Synthesis
    # Core Cross-Fractal Signal
    base_cross_fractal = data['cross_efficiency_momentum'] * data['cross_efficiency_persistence']
    asymmetric_enhanced = base_cross_fractal * data['asymmetric_cross_fractal_pressure']
    volume_integrated = asymmetric_enhanced * data['cross_volume_fractal_alignment']
    
    # Microstructure Cross-Fractal Integration
    session_asymmetry_aligned = volume_integrated * data['cross_fractal_session_asymmetry']
    opening_closing_flow = session_asymmetry_aligned * (data['opening_cross_fractal_asymmetry'] + data['closing_cross_fractal_asymmetry'])
    microstructure_momentum = opening_closing_flow * abs(data['close'] - data['open']) / (data['high'] - data['low'] + 1e-8)
    
    # Regime-Adaptive Cross-Fractal
    efficiency_scaled = microstructure_momentum * data['regime_multiplier']
    volume_aligned = efficiency_scaled * np.where(data['high_volume_regime'], 1.2, 
                                                np.where(data['low_volume_regime'], 0.8, 1.0))
    persistence_enhanced = volume_aligned * data['cross_efficiency_persistence']
    
    # Final Composite Cross-Fractal Alpha
    final_alpha = (persistence_enhanced * 
                  data['cross_efficiency_momentum'] * 
                  data['asymmetric_cross_fractal_pressure'] * 
                  data['cross_fractal_session_asymmetry'] * 
                  data['cross_volume_fractal_alignment'] * 
                  data['regime_multiplier'])
    
    return final_alpha
