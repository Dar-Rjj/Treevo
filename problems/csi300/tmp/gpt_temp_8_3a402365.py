import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate rolling highs and lows for various windows
    data['high_2d'] = data['high'].rolling(window=3).max()
    data['low_2d'] = data['low'].rolling(window=3).min()
    data['high_6d'] = data['high'].rolling(window=7).max()
    data['low_6d'] = data['low'].rolling(window=7).min()
    data['high_4d'] = data['high'].rolling(window=5).max()
    data['low_4d'] = data['low'].rolling(window=5).min()
    
    # Calculate returns for stock and sector/market (using close as proxy)
    data['ret_5d'] = data['close'] / data['close'].shift(5) - 1
    data['ret_3d'] = data['close'] / data['close'].shift(3) - 1
    data['ret_7d'] = data['close'] / data['close'].shift(7) - 1
    data['ret_1d'] = data['close'] / data['close'].shift(1) - 1
    
    # Cross-Fractal Efficiency Framework
    data['stock_sector_efficiency'] = (abs(data['close'] - data['close'].shift(5)) / 
                                     (data['high'] - data['low'])) * data['ret_5d']
    data['stock_market_efficiency'] = (abs(data['close'] - data['close'].shift(5)) / 
                                     (data['high'] - data['low'])) * data['ret_5d']
    data['cross_fractal_divergence'] = (data['stock_sector_efficiency'] - 
                                      data['stock_market_efficiency'])
    
    # Multi-Timeframe Efficiency Dynamics
    data['ultra_short_cross_efficiency'] = (abs(data['close'] - data['close'].shift(3)) / 
                                          (data['high_2d'] - data['low_2d'])) * np.sign(data['cross_fractal_divergence'])
    data['medium_term_cross_efficiency'] = (abs(data['close'] - data['close'].shift(7)) / 
                                          (data['high_6d'] - data['low_6d'])) * np.sign(data['cross_fractal_divergence'])
    data['cross_efficiency_momentum'] = ((data['medium_term_cross_efficiency'] - 
                                        data['ultra_short_cross_efficiency']) * 
                                       np.sign(data['close'] - data['close'].shift(1)))
    
    # Volume-Enhanced Cross Efficiency
    data['volume_ratio_5d'] = data['volume'] / data['volume'].shift(5) - 1
    data['cross_efficiency_volume_ratio'] = data['cross_fractal_divergence'] * data['volume_ratio_5d']
    data['cross_efficiency_volume_momentum'] = (data['cross_efficiency_volume_ratio'] * 
                                              np.sign(data['volume'] - data['volume'].shift(1)))
    
    # Cross Efficiency Persistence
    def calculate_persistence(series):
        if len(series) < 5:
            return np.nan
        current_sign = np.sign(series.iloc[-1])
        persistence = sum(np.sign(series.iloc[-5:]) == current_sign)
        return persistence
    
    data['cross_efficiency_persistence'] = (data['cross_fractal_divergence']
                                          .rolling(window=5)
                                          .apply(calculate_persistence, raw=False))
    
    # Asymmetric Cross-Fractal Dynamics
    data['cross_fractal_volatility_asymmetry'] = ((data['high_4d'] - data['close'].shift(4)) / 
                                                (data['close'].shift(4) - data['low_4d'])) * np.sign(data['cross_fractal_divergence'])
    
    # Cross-Fractal Momentum Asymmetry
    def momentum_asymmetry(returns):
        if len(returns) < 7:
            return np.nan
        pos_returns = returns[returns > 0].sum()
        neg_returns = abs(returns[returns < 0].sum())
        if pos_returns + neg_returns == 0:
            return 0
        return pos_returns / (pos_returns + neg_returns)
    
    data['returns_7d'] = data['close'].pct_change(periods=7)
    data['cross_fractal_momentum_asymmetry'] = (data['returns_7d']
                                              .rolling(window=7)
                                              .apply(momentum_asymmetry, raw=False) * 
                                              np.sign(data['cross_fractal_divergence']))
    
    data['asymmetric_cross_fractal_pressure'] = (data['cross_fractal_volatility_asymmetry'] * 
                                               data['cross_fractal_momentum_asymmetry'] * 
                                               (data['close'] - data['close'].shift(1)) / 
                                               (data['high'] - data['low']))
    
    # Microstructure Asymmetry
    data['opening_cross_fractal_asymmetry'] = (((data['open'] - data['low']) - 
                                              (data['high'] - data['open'])) * 
                                             np.sign(data['cross_fractal_divergence']) * 
                                             abs(data['open'] - data['close'].shift(1)) / 
                                             (data['high'] - data['low']))
    
    data['closing_cross_fractal_asymmetry'] = (((data['close'] - data['low']) - 
                                              (data['high'] - data['close'])) * 
                                             np.sign(data['cross_fractal_divergence']) * 
                                             abs(data['close'] - data['open']) / 
                                             (data['high'] - data['low']))
    
    data['cross_fractal_session_asymmetry'] = (data['opening_cross_fractal_asymmetry'] - 
                                             data['closing_cross_fractal_asymmetry'])
    
    # Volume Asymmetry Dynamics
    def volume_asymmetry_momentum(volume_series):
        if len(volume_series) < 7:
            return np.nan
        price_changes = volume_series.index.map(lambda x: data.loc[x, 'close'] - data.loc[x, 'open'])
        up_volume = volume_series[price_changes > 0].sum()
        down_volume = volume_series[price_changes < 0].sum()
        if down_volume == 0:
            return up_volume / 1e-6
        return up_volume / down_volume
    
    data['cross_volume_asymmetry_momentum'] = (data['volume']
                                             .rolling(window=7)
                                             .apply(volume_asymmetry_momentum, raw=False) * 
                                             np.sign(data['cross_fractal_divergence']))
    
    data['cross_volume_fractal_shock'] = np.where(data['volume'] > 1.8 * data['volume'].shift(1), -1,
                                                 np.where(data['volume'] < 0.6 * data['volume'].shift(1), 1, 0))
    
    data['cross_volume_fractal_alignment'] = (data['cross_volume_asymmetry_momentum'] * 
                                            data['cross_volume_fractal_shock'] * 
                                            (data['close'] - data['open']) / 
                                            (data['high'] - data['low']))
    
    # Cross-Fractal Regime Classification
    data['high_efficiency'] = ((abs(data['cross_fractal_divergence']) > 0.8 * (data['high'] - data['low'])) & 
                             (data['cross_efficiency_momentum'] > 0))
    data['moderate_efficiency'] = ((abs(data['cross_fractal_divergence']) >= 0.3 * (data['high'] - data['low'])) & 
                                 (abs(data['cross_fractal_divergence']) <= 0.8 * (data['high'] - data['low'])))
    data['low_efficiency'] = ((abs(data['cross_fractal_divergence']) < 0.3 * (data['high'] - data['low'])) | 
                            (data['cross_efficiency_momentum'] < 0))
    
    data['high_volume_cross_fractal'] = ((data['volume'] / data['volume'].shift(5) > 1.5) & 
                                       (data['cross_volume_fractal_shock'] == -1))
    data['low_volume_cross_fractal'] = ((data['volume'] / data['volume'].shift(5) < 0.7) | 
                                      (data['cross_volume_fractal_shock'] == 1))
    data['normal_volume_cross_fractal'] = (~data['high_volume_cross_fractal'] & 
                                         ~data['low_volume_cross_fractal'])
    
    # Cross-Fractal Regime Multipliers
    data['regime_multiplier'] = 1.0
    data.loc[data['high_efficiency'] & data['high_volume_cross_fractal'], 'regime_multiplier'] = 2.2
    data.loc[data['high_efficiency'] & data['low_volume_cross_fractal'], 'regime_multiplier'] = 1.8
    data.loc[data['low_efficiency'] & data['high_volume_cross_fractal'], 'regime_multiplier'] = 1.4
    data.loc[data['low_efficiency'] & data['low_volume_cross_fractal'], 'regime_multiplier'] = 0.6
    
    # Integrated Cross-Fractal Signal Synthesis
    # Core Cross-Fractal Signal
    data['base_cross_fractal'] = data['cross_efficiency_momentum'] * data['cross_efficiency_persistence']
    data['asymmetric_enhanced'] = data['base_cross_fractal'] * data['asymmetric_cross_fractal_pressure']
    data['volume_integrated_cross_fractal'] = data['asymmetric_enhanced'] * data['cross_volume_fractal_alignment']
    
    # Microstructure Cross-Fractal Integration
    data['session_asymmetry_alignment'] = (data['volume_integrated_cross_fractal'] * 
                                         data['cross_fractal_session_asymmetry'])
    data['opening_closing_fractal_flow'] = (data['session_asymmetry_alignment'] * 
                                          (data['opening_cross_fractal_asymmetry'] + 
                                           data['closing_cross_fractal_asymmetry']))
    data['cross_fractal_microstructure_momentum'] = (data['opening_closing_fractal_flow'] * 
                                                   abs(data['close'] - data['open']) / 
                                                   (data['high'] - data['low']))
    
    # Regime-Adaptive Cross-Fractal
    data['efficiency_regime_scaling'] = (data['cross_fractal_microstructure_momentum'] * 
                                       data['regime_multiplier'])
    data['volume_regime_alignment'] = (data['efficiency_regime_scaling'] * 
                                     np.where(data['high_volume_cross_fractal'], 1.2,
                                             np.where(data['low_volume_cross_fractal'], 0.8, 1.0)))
    data['cross_fractal_persistence_enhancement'] = (data['volume_regime_alignment'] * 
                                                   data['cross_efficiency_persistence'])
    
    # Final Composite Cross-Fractal Alpha
    alpha = (data['cross_efficiency_momentum'] * 
             data['asymmetric_cross_fractal_pressure'] * 
             data['cross_fractal_session_asymmetry'] * 
             data['cross_volume_fractal_alignment'] * 
             data['regime_multiplier'] * 
             data['cross_efficiency_persistence'])
    
    return alpha
