import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Small epsilon to avoid division by zero
    eps = 1e-8
    
    # Multi-Frequency Volatility Regime Detection
    data['micro_vol'] = (data['high'] - data['low']) / (
        data['high'].rolling(window=3).max().shift(1) - 
        data['low'].rolling(window=3).min().shift(1) + eps
    )
    
    data['meso_vol'] = (
        data['high'].rolling(window=5).max() - 
        data['low'].rolling(window=5).min()
    ) / (
        data['high'].rolling(window=10).max() - 
        data['low'].rolling(window=10).min() + eps
    )
    
    data['macro_vol'] = (
        data['high'].rolling(window=20).max() - 
        data['low'].rolling(window=20).min()
    ) / (
        data['high'].rolling(window=40).max() - 
        data['low'].rolling(window=40).min() + eps
    )
    
    data['volume_regime'] = data['volume'] / (data['volume'].rolling(window=5).mean() + eps)
    
    # Regime Classification
    def classify_regime(row):
        if row['micro_vol'] > 1:
            vol_class = "High"
        elif row['meso_vol'] > 1:
            vol_class = "Medium"
        else:
            vol_class = "Low"
        
        if row['volume_regime'] > 1:
            vol_regime = "_HighVol"
        else:
            vol_regime = "_LowVol"
        
        return vol_class + vol_regime
    
    data['regime_class'] = data.apply(classify_regime, axis=1)
    
    # Regime-Specific Momentum Construction
    data['high_vol_momentum'] = (data['close'] - data['close'].shift(1)) / (data['high'] - data['low'] + eps)
    data['medium_vol_momentum'] = (data['close'] - data['close'].rolling(window=5).mean()) / (
        data['high'].rolling(window=5).max() - data['low'].rolling(window=5).min() + eps
    )
    data['low_vol_momentum'] = (data['close'] - data['close'].shift(5)) / (
        data['high'].rolling(window=20).max() - data['low'].rolling(window=20).min() + eps
    )
    
    def get_regime_momentum(row):
        regime = row['regime_class']
        if regime in ["High_HighVol", "Medium_HighVol"]:
            return row['high_vol_momentum']
        elif regime in ["Low_HighVol", "High_LowVol"]:
            return row['medium_vol_momentum']
        else:  # "Medium_LowVol", "Low_LowVol"
            return row['low_vol_momentum']
    
    data['regime_momentum'] = data.apply(get_regime_momentum, axis=1)
    
    # Price-Efficiency Asymmetry
    data['opening_efficiency'] = (data['close'] - data['open']) / (data['high'] - data['low'] + eps)
    data['closing_efficiency'] = (data['close'] - data['open']) / (
        data['high'].rolling(window=3).max().shift(1) - 
        data['low'].rolling(window=3).min().shift(1) + eps
    )
    data['intraday_pressure'] = ((data['high'] - data['close']) - (data['close'] - data['low'])) / (data['high'] - data['low'] + eps)
    data['efficiency_asymmetry'] = data['opening_efficiency'] * data['closing_efficiency'] * data['intraday_pressure']
    
    # Volume-Price Convergence Dynamics
    data['volume_momentum'] = data['volume'] / (data['volume'].shift(1) + eps) - 1
    data['price_momentum'] = (data['close'] - data['close'].shift(1)) / (data['close'].shift(1) + eps)
    data['price_volume_divergence'] = data['price_momentum'] - data['volume_momentum']
    
    data['convergence_strength'] = np.abs(data['price_volume_divergence']) / (
        data['close'].rolling(window=5).std() / data['close'].rolling(window=5).mean() + eps
    )
    data['convergence_signal'] = np.sign(data['price_volume_divergence']) * data['convergence_strength']
    
    # Multi-Timeframe Momentum Alignment
    data['short_momentum'] = (data['close'] - data['close'].shift(1)) / (data['close'].shift(1) + eps)
    data['medium_momentum'] = (data['close'] - data['close'].shift(5)) / (data['close'].shift(5) + eps)
    data['long_momentum'] = (data['close'] - data['close'].shift(20)) / (data['close'].shift(20) + eps)
    
    data['momentum_alignment'] = (
        np.sign(data['short_momentum']) * np.sign(data['medium_momentum']) * np.sign(data['long_momentum']) *
        (np.abs(data['short_momentum']) + np.abs(data['medium_momentum']) + np.abs(data['long_momentum']))
    )
    
    # Regime-Adaptive Factor Construction
    def get_adaptive_core(row):
        regime = row['regime_class']
        if regime in ["High_HighVol", "Medium_HighVol"]:
            return row['regime_momentum'] * row['efficiency_asymmetry'] * row['convergence_signal']
        elif regime in ["Low_HighVol", "High_LowVol"]:
            return row['regime_momentum'] * row['momentum_alignment'] * row['efficiency_asymmetry']
        else:  # "Medium_LowVol", "Low_LowVol"
            return row['regime_momentum'] * row['momentum_alignment'] * row['convergence_signal']
    
    data['adaptive_core'] = data.apply(get_adaptive_core, axis=1)
    
    # Persistence and Quality Enhancement
    def calculate_persistence(series, window):
        return series.rolling(window=window).apply(
            lambda x: np.sum(np.sign(x) == np.sign(x.shift(1))) / (window - 1) if len(x) == window else np.nan
        )
    
    data['momentum_persistence'] = calculate_persistence(data['regime_momentum'], 6)
    data['efficiency_persistence'] = calculate_persistence(data['efficiency_asymmetry'], 4)
    data['convergence_persistence'] = calculate_persistence(data['convergence_signal'], 6)
    
    data['quality_multiplier'] = data['momentum_persistence'] * data['efficiency_persistence'] * data['convergence_persistence']
    
    # Liquidity-Adjusted Momentum
    data['trade_efficiency'] = data['amount'] / (data['volume'] * (data['high'] - data['low']) + eps)
    
    price_change_volume = (data['close'] - data['close'].shift(1)) * data['volume']
    data['liquidity_momentum'] = price_change_volume / (price_change_volume.rolling(window=5).mean() + eps)
    
    data['liquidity_adjustment'] = data['trade_efficiency'] * data['liquidity_momentum']
    
    # Final Alpha Construction
    data['core_factor'] = data['adaptive_core'] * data['quality_multiplier']
    data['liquidity_enhancement'] = data['core_factor'] * data['liquidity_adjustment']
    data['regime_confirmation'] = (
        data['liquidity_enhancement'] * 
        np.sign(data['regime_momentum']) * 
        np.sign(data['momentum_alignment'])
    )
    
    data['final_alpha'] = data['core_factor'] * data['liquidity_enhancement'] * data['regime_confirmation']
    
    return data['final_alpha']
