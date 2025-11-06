import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    data = df.copy()
    
    # Volatility Regime Detection
    data['short_term_vol'] = (data['high'] - data['low']).rolling(window=3, min_periods=3).sum()
    data['medium_term_vol'] = (data['high'] - data['low']).rolling(window=10, min_periods=10).sum()
    data['volatility_regime'] = data['short_term_vol'] > (data['medium_term_vol'] / 3)
    
    # Microstructure Regime Detection
    data['price_efficiency'] = np.abs(data['close'] - data['open']) / (data['high'] - data['low'] + 0.0001)
    data['volume_efficiency'] = data['volume'] / (np.abs(data['close'] - data['open']) + 0.0001)
    data['microstructure_regime'] = data['price_efficiency'] > data['volume_efficiency']
    
    # High-Volatility Component
    data['extreme_move'] = (data['close'] - data['close'].shift(1)) / (
        data['high'].rolling(window=5, min_periods=5).max() - 
        data['low'].rolling(window=5, min_periods=5).min() + 0.0001
    )
    
    vol_breakout_window = data['high'] - data['low']
    data['volatility_breakout'] = vol_breakout_window / vol_breakout_window.shift(5).rolling(window=5, min_periods=5).mean()
    
    # Low-Volatility Component
    data['steady_trend'] = (data['close'] - data['close'].rolling(window=5, min_periods=5).mean()) / (
        data['close'].rolling(window=5, min_periods=5).max() - 
        data['close'].rolling(window=5, min_periods=5).min() + 0.0001
    )
    
    def calc_trend_consistency(series):
        current_sign = np.sign(series.iloc[-1] - series.iloc[-2])
        window_signs = np.sign(series.diff().iloc[-7:])
        return np.sum(window_signs == current_sign) / 7
    
    data['trend_consistency'] = data['close'].rolling(window=8, min_periods=8).apply(
        calc_trend_consistency, raw=False
    )
    
    # Regime-Adaptive Momentum
    data['high_vol_component'] = data['extreme_move'] * data['volatility_breakout']
    data['low_vol_component'] = data['steady_trend'] * data['trend_consistency']
    data['regime_adaptive_momentum'] = np.where(
        data['volatility_regime'], 
        data['high_vol_component'], 
        data['low_vol_component']
    )
    
    # Microstructure Pressure
    data['opening_rejection'] = (data['high'] - np.maximum(data['open'], data['close'])) / (data['high'] - data['low'] + 0.0001)
    data['closing_pressure'] = (np.minimum(data['open'], data['close']) - data['low']) / (data['high'] - data['low'] + 0.0001)
    data['pressure_balance'] = data['closing_pressure'] - data['opening_rejection']
    
    # Volume Dynamics
    data['volume_entropy'] = data['volume'] / (data['volume'] - data['volume'].shift(1) + 0.0001)
    
    def calc_volume_trend(series):
        current_sign = np.sign(series.iloc[-1] - series.iloc[-2])
        window_signs = np.sign(series.diff().iloc[-5:])
        return np.sum(window_signs == current_sign) / 5
    
    data['volume_trend'] = data['volume'].rolling(window=6, min_periods=6).apply(
        calc_volume_trend, raw=False
    )
    data['volume_signal'] = data['volume_entropy'] * data['volume_trend']
    
    # Amount Dynamics
    data['per_unit_impact'] = (data['close'] - data['open']) * (data['amount'] / (data['volume'] + 0.0001)) / (data['high'] - data['low'] + 0.0001)
    data['amount_momentum'] = data['per_unit_impact'] * (data['amount'] / (data['amount'].shift(1) + 0.0001))
    
    # Alpha Synthesis
    data['core_signal'] = data['regime_adaptive_momentum'] * data['pressure_balance'] * data['volume_signal'] * data['amount_momentum']
    data['market_adjustment'] = data['core_signal'] * np.sign(data['close'] - data['close'].shift(5))
    
    recent_range = np.abs(data['close'] - data['open']).rolling(window=5, min_periods=5).sum()
    longer_range = np.abs(data['close'] - data['open']).rolling(window=10, min_periods=10).sum()
    data['range_adjustment'] = 1 + (recent_range / (longer_range + 0.0001) - 1)
    
    data['alpha'] = data['market_adjustment'] * data['range_adjustment']
    
    return data['alpha']
