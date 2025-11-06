import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Copy dataframe to avoid modifying original
    data = df.copy()
    
    # Multi-Scale Fractal Efficiency
    # Ultra-Short Efficiency (2-day)
    data['ultra_short_eff'] = np.abs(data['close'] - data['close'].shift(2)) / (
        data['high'].rolling(window=2).max() - data['low'].rolling(window=2).min()
    )
    
    # Short-Term Efficiency (5-day)
    data['short_term_eff'] = np.abs(data['close'] - data['close'].shift(5)) / (
        data['high'].rolling(window=5).max() - data['low'].rolling(window=5).min()
    )
    
    # 20-day efficiency for decay calculation
    data['medium_term_eff'] = np.abs(data['close'] - data['close'].shift(20)) / (
        data['high'].rolling(window=20).max() - data['low'].rolling(window=20).min()
    )
    
    # Efficiency Decay
    data['efficiency_decay'] = (data['short_term_eff'] - data['ultra_short_eff']) * np.sign(
        data['medium_term_eff'] - data['short_term_eff']
    )
    
    # Fractal Price Efficiency
    data['fractal_price_eff'] = np.abs(data['close'] - data['open']) / (data['high'] - data['low'])
    
    # Asymmetric Fractal Momentum
    # Upside and Downside Volatility (5-day)
    data['returns'] = data['close'].pct_change()
    data['upside_vol'] = data['returns'].rolling(window=5).apply(
        lambda x: np.std(x[x > 0]) if len(x[x > 0]) > 1 else 0
    )
    data['downside_vol'] = data['returns'].rolling(window=5).apply(
        lambda x: np.std(x[x < 0]) if len(x[x < 0]) > 1 else 0
    )
    data['volatility_asymmetry'] = data['upside_vol'] / (data['downside_vol'] + 1e-8)
    
    # Upside and Downside Momentum (5-day)
    data['upside_momentum'] = data['returns'].rolling(window=5).apply(
        lambda x: np.mean(x[x > 0]) if len(x[x > 0]) > 0 else 0
    )
    data['downside_momentum'] = data['returns'].rolling(window=5).apply(
        lambda x: np.mean(x[x < 0]) if len(x[x < 0]) > 0 else 0
    )
    data['momentum_asymmetry'] = data['upside_momentum'] / (
        data['upside_momentum'] + np.abs(data['downside_momentum']) + 1e-8
    )
    
    # Fractal Volatility Skew
    data['fractal_vol_skew'] = (
        (data['high'] - data['open']) / (data['high'] - data['low'] + 1e-8) - 
        (data['open'] - data['low']) / (data['high'] - data['low'] + 1e-8)
    )
    
    # Volatility-Momentum Integration
    data['vol_momentum_integration'] = data['momentum_asymmetry'] * data['volatility_asymmetry']
    
    # Fractal Microstructure Dynamics
    # Opening Fracture
    data['opening_fracture'] = (
        (data['open'] - data['low']) - (data['high'] - data['open'])
    ) * (
        np.abs(data['close'] - data['open']) / (np.abs(data['open'] - data['close'].shift(1)) + 1e-8)
    )
    
    # Fractal Closing Acceleration
    data['fractal_closing_accel'] = (
        (data['close'] - data['open']) / (data['close'].shift(1) - data['open'].shift(1) + 1e-8)
    )
    
    # Price Fractal Divergence
    # Short-term Fractal (3-day)
    data['short_fractal'] = np.abs(data['close'] - data['close'].shift(3)) / (
        data['high'].rolling(window=3).max() - data['low'].rolling(window=3).min()
    )
    # Medium-term Fractal (8-day)
    data['medium_fractal'] = np.abs(data['close'] - data['close'].shift(8)) / (
        data['high'].rolling(window=8).max() - data['low'].rolling(window=8).min()
    )
    data['price_fractal_divergence'] = np.abs(data['short_fractal'] - data['medium_fractal'])
    
    # Fractal Auction Imbalance
    data['fractal_auction_imbalance'] = (data['open'] - data['low']) - (data['high'] - data['open'])
    
    # Volume Fractal Behavior
    # Volume Asymmetry
    data['up_volume'] = np.where(data['returns'] > 0, data['volume'], 0)
    data['down_volume'] = np.where(data['returns'] < 0, data['volume'], 0)
    data['volume_asymmetry'] = (
        data['up_volume'].rolling(window=5).sum() / 
        (data['down_volume'].rolling(window=5).sum() + 1e-8)
    )
    
    # Fractal Volume Persistence
    data['volume_increase'] = data['volume'] > data['volume'].shift(1)
    data['fractal_volume_persistence'] = data['volume_increase'].rolling(window=3).sum()
    
    # Fractal Volume Exhaustion
    data['fractal_volume_exhaustion'] = np.where(
        data['volume'] > 2 * data['volume'].shift(1), -1, 1
    )
    
    # Fractal Flow Imbalance
    data['positive_amount_flow'] = np.where(data['returns'] > 0, data['amount'], 0)
    data['negative_amount_flow'] = np.where(data['returns'] < 0, data['amount'], 0)
    data['fractal_flow_imbalance'] = (
        (data['positive_amount_flow'].rolling(window=5).sum() - 
         data['negative_amount_flow'].rolling(window=5).sum()) / 
        (data['positive_amount_flow'].rolling(window=5).sum() + 
         data['negative_amount_flow'].rolling(window=5).sum() + 1e-8)
    ) * data['fractal_volume_persistence']
    
    # Integration Framework
    # Efficiency-Momentum Core
    data['efficiency_momentum_core'] = data['efficiency_decay'] * data['vol_momentum_integration']
    
    # Microstructure-Volume Alignment
    data['microstructure_volume_alignment'] = (
        (1 - data['price_fractal_divergence']) * 
        data['volume_asymmetry'] * 
        data['fractal_volume_persistence']
    )
    
    # Fractal Position Signal
    data['fractal_position_signal'] = (
        (data['close'] - data['low'].rolling(window=3).min()) / 
        (data['high'].rolling(window=3).max() - data['low'].rolling(window=3).min() + 1e-8)
    ) * data['momentum_asymmetry']
    
    # Dynamic Multiplier
    data['dynamic_multiplier'] = data['fractal_volume_exhaustion'] * data['fractal_price_eff']
    
    # Base Signal
    data['base_signal'] = (
        data['efficiency_momentum_core'] * 
        data['microstructure_volume_alignment'] * 
        data['fractal_position_signal']
    )
    
    # Regime Classification
    conditions = [
        (data['fractal_price_eff'] > 0.7) & (data['efficiency_decay'] > 0),
        (data['fractal_price_eff'] >= 0.3) & (data['fractal_price_eff'] <= 0.7),
        (data['fractal_price_eff'] < 0.3) & (data['efficiency_decay'] < 0),
        (data['fractal_volume_exhaustion'] == -1) & (data['fractal_price_eff'] < 0.2)
    ]
    choices = [1.2, 1.0, 0.8, 0.6]
    data['regime_multiplier'] = np.select(conditions, choices, default=1.0)
    
    # Final Alpha
    data['final_alpha'] = (
        data['base_signal'] * 
        data['regime_multiplier'] * 
        data['dynamic_multiplier']
    )
    
    # Return the final alpha factor
    return data['final_alpha']
