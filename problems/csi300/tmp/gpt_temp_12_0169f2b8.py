import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Regime-Adaptive Gap Momentum with Volume-Pressure Anchoring
    """
    data = df.copy()
    
    # Multi-Timeframe Gap Momentum Analysis
    # Overnight gap momentum calculation
    data['raw_gap'] = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
    data['gap_persistence'] = np.sign(data['open'] - data['close'].shift(1)) * np.sign(data['open'].shift(1) - data['close'].shift(2))
    data['gap_acceleration'] = (data['open'] - data['close'].shift(1)) - (data['open'].shift(1) - data['close'].shift(2))
    
    # Intraday gap momentum calculation
    data['range_utilization'] = (data['close'] - data['low']) / (data['high'] - data['low'] + 1e-8)
    data['intraday_momentum'] = (data['high'] - data['low']) / data['open']
    data['gap_to_close_persistence'] = np.sign(data['close'] - data['open']) * np.sign(data['open'] - data['close'].shift(1))
    
    # Multi-timeframe gap alignment
    data['short_term_gap_consistency'] = data['raw_gap'].rolling(window=2).apply(lambda x: np.sum(np.sign(x) == np.sign(x.iloc[-1])) / len(x) if len(x) == 2 else np.nan)
    data['medium_term_gap_persistence'] = data['raw_gap'].rolling(window=5).apply(lambda x: np.sum(np.sign(x) == np.sign(x.iloc[-1])) / len(x) if len(x) == 5 else np.nan)
    data['gap_momentum_decay'] = data['raw_gap'] - data['intraday_momentum']
    
    # Volume-Pressure Confirmation System
    # Volume trend analysis
    data['volume_momentum'] = data['volume'] / data['volume'].shift(1)
    
    def count_volume_increase(window):
        if len(window) < 5:
            return np.nan
        return np.sum([window.iloc[i] > window.iloc[i-1] for i in range(1, len(window))])
    
    data['volume_trend_strength'] = data['volume'].rolling(window=5).apply(count_volume_increase, raw=False)
    data['volume_deviation'] = data['volume'] - data['volume'].rolling(window=5).mean()
    
    # Intraday pressure signals
    data['range_pressure'] = data['range_utilization'] * data['volume']
    data['gap_pressure'] = data['raw_gap'] * data['volume']
    data['pressure_intensity'] = data['range_pressure'] * data['gap_pressure']
    
    # Volume-pressure anchoring
    data['anchoring_strength'] = data['volume_deviation'] * data['pressure_intensity']
    data['volume_confirmation'] = np.sign(data['volume_momentum']) * np.sign(data['raw_gap'])
    
    def count_pressure_increase(window):
        if len(window) < 3:
            return np.nan
        return np.sum([window.iloc[i] > window.iloc[i-1] for i in range(1, len(window))])
    
    data['pressure_persistence'] = data['pressure_intensity'].rolling(window=3).apply(count_pressure_increase, raw=False)
    
    # Volatility-Regime Adaptive Framework
    # Volatility regime detection
    data['short_term_vol'] = data['close'].rolling(window=5).std()
    data['long_term_vol'] = data['close'].rolling(window=20).std()
    data['volatility_ratio'] = data['short_term_vol'] / (data['long_term_vol'] + 1e-8)
    
    # Regime classification
    data['high_vol_regime'] = data['volatility_ratio'] > 1
    data['regime_transition'] = np.sign(data['volatility_ratio'] - data['volatility_ratio'].shift(1))
    
    # Average True Range calculation for low volatility regime
    data['tr1'] = data['high'] - data['low']
    data['tr2'] = abs(data['high'] - data['close'].shift(1))
    data['tr3'] = abs(data['low'] - data['close'].shift(1))
    data['true_range'] = data[['tr1', 'tr2', 'tr3']].max(axis=1)
    data['atr_5'] = data['true_range'].rolling(window=5).mean()
    
    # Regime-Adaptive Factor Synthesis
    # High Volatility Regime Processing
    data['gap_momentum_adjustment'] = data['raw_gap'] * data['volatility_ratio']
    data['volume_pressure_scaling'] = data['anchoring_strength'] / (data['short_term_vol'] + 1e-8)
    data['gap_persistence_weighting'] = data['gap_persistence'] * data['volume_confirmation']
    data['high_vol_factor'] = data['gap_momentum_adjustment'] * data['volume_pressure_scaling'] * data['gap_persistence_weighting']
    
    # Low Volatility Regime Processing
    data['volatility_scaling'] = data['atr_5']
    data['gap_momentum_smoothing'] = data['raw_gap'] / (data['volatility_scaling'] + 1e-8)
    data['anchored_gap'] = data['gap_momentum_smoothing'] * data['anchoring_strength']
    data['intraday_confirmation'] = data['range_utilization'] * data['volume_trend_strength']
    data['low_vol_factor'] = data['anchored_gap'] * data['intraday_confirmation'] * data['pressure_persistence']
    
    # Final Alpha Factor - Adaptive Gap Momentum Anchor Score
    data['regime_adaptive_gap_momentum_anchor'] = np.where(
        data['high_vol_regime'],
        data['high_vol_factor'],
        data['low_vol_factor']
    )
    
    # Clean up intermediate columns
    result = data['regime_adaptive_gap_momentum_anchor'].copy()
    
    return result
