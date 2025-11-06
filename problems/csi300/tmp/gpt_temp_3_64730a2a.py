import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(data):
    df = data.copy()
    
    # Multi-Scale Fractal Efficiency
    # Ultra-Short Efficiency (2-day)
    df['ultra_short_eff'] = abs(df['close'] - df['close'].shift(2)) / (
        df['high'].rolling(window=2).max() - df['low'].rolling(window=2).min()
    )
    
    # Short-Term Efficiency (5-day)
    df['short_term_eff'] = abs(df['close'] - df['close'].shift(5)) / (
        df['high'].rolling(window=5).max() - df['low'].rolling(window=5).min()
    )
    
    # 20-day efficiency for decay calculation
    df['twenty_day_eff'] = abs(df['close'] - df['close'].shift(20)) / (
        df['high'].rolling(window=20).max() - df['low'].rolling(window=20).min()
    )
    
    # Efficiency Decay
    df['efficiency_decay'] = (df['short_term_eff'] - df['ultra_short_eff']) * np.sign(
        df['twenty_day_eff'] - df['short_term_eff']
    )
    
    # Fractal Price Efficiency
    df['fractal_price_eff'] = abs(df['close'] - df['open']) / (df['high'] - df['low'])
    
    # Asymmetric Fractal Momentum
    # Upside and Downside Volatility (5-day)
    df['returns'] = df['close'].pct_change()
    df['upside_vol'] = df['returns'].rolling(window=5).apply(
        lambda x: np.std(x[x > 0]) if len(x[x > 0]) > 0 else 0
    )
    df['downside_vol'] = df['returns'].rolling(window=5).apply(
        lambda x: np.std(x[x < 0]) if len(x[x < 0]) > 0 else 0
    )
    df['volatility_asymmetry'] = df['upside_vol'] / (df['downside_vol'] + 1e-8)
    
    # Momentum Asymmetry
    df['upside_momentum'] = df['returns'].rolling(window=5).apply(
        lambda x: x[x > 0].sum() if len(x[x > 0]) > 0 else 0
    )
    df['downside_momentum'] = df['returns'].rolling(window=5).apply(
        lambda x: x[x < 0].sum() if len(x[x < 0]) > 0 else 0
    )
    df['momentum_asymmetry'] = df['upside_momentum'] / (
        df['upside_momentum'] + abs(df['downside_momentum']) + 1e-8
    )
    
    # Fractal Volatility Skew
    df['fractal_vol_skew'] = (
        (df['high'] - df['open']) / (df['high'] - df['low'] + 1e-8) - 
        (df['open'] - df['low']) / (df['high'] - df['low'] + 1e-8)
    )
    
    # Volatility-Momentum Integration
    df['vol_momentum_integration'] = df['momentum_asymmetry'] * df['volatility_asymmetry']
    
    # Fractal Microstructure Dynamics
    # Opening Fracture
    df['opening_fracture'] = (
        (df['open'] - df['low']) - (df['high'] - df['open'])
    ) * (abs(df['close'] - df['open']) / (abs(df['open'] - df['close'].shift(1)) + 1e-8))
    
    # Fractal Closing Acceleration
    df['fractal_closing_accel'] = (df['close'] - df['open']) / (
        df['close'].shift(1) - df['open'].shift(1) + 1e-8
    )
    
    # Price Fractal Divergence
    df['short_fractal'] = (df['close'] - df['low'].rolling(window=3).min()) / (
        df['high'].rolling(window=3).max() - df['low'].rolling(window=3).min() + 1e-8
    )
    df['medium_fractal'] = (df['close'] - df['low'].rolling(window=8).min()) / (
        df['high'].rolling(window=8).max() - df['low'].rolling(window=8).min() + 1e-8
    )
    df['price_fractal_divergence'] = abs(df['short_fractal'] - df['medium_fractal'])
    
    # Fractal Auction Imbalance
    df['fractal_auction_imbalance'] = (df['open'] - df['low']) - (df['high'] - df['open'])
    
    # Volume Fractal Behavior
    # Volume Asymmetry
    df['up_volume'] = np.where(df['returns'] > 0, df['volume'], 0)
    df['down_volume'] = np.where(df['returns'] < 0, df['volume'], 0)
    df['volume_asymmetry'] = (
        df['up_volume'].rolling(window=5).sum() / 
        (df['down_volume'].rolling(window=5).sum() + 1e-8)
    )
    
    # Fractal Volume Persistence
    df['volume_increase'] = df['volume'] > df['volume'].shift(1)
    df['fractal_volume_persistence'] = df['volume_increase'].rolling(window=3).sum()
    
    # Fractal Volume Exhaustion
    df['fractal_volume_exhaustion'] = np.where(
        df['volume'] > 2 * df['volume'].shift(1), -1, 1
    )
    
    # Fractal Flow Imbalance
    df['positive_amount_flow'] = np.where(df['returns'] > 0, df['amount'], 0)
    df['negative_amount_flow'] = np.where(df['returns'] < 0, df['amount'], 0)
    df['fractal_flow_imbalance'] = (
        (df['positive_amount_flow'] - df['negative_amount_flow']) / 
        (df['positive_amount_flow'] + df['negative_amount_flow'] + 1e-8)
    ) * df['fractal_volume_persistence']
    
    # Integration Framework
    # Efficiency-Momentum Core
    df['efficiency_momentum_core'] = df['efficiency_decay'] * df['vol_momentum_integration']
    
    # Microstructure-Volume Alignment
    df['microstructure_volume_alignment'] = (
        (1 - df['price_fractal_divergence']) * 
        df['volume_asymmetry'] * 
        df['fractal_volume_persistence']
    )
    
    # Fractal Position Signal
    df['fractal_position_signal'] = (
        (df['close'] - df['low'].rolling(window=3).min()) / 
        (df['high'].rolling(window=3).max() - df['low'].rolling(window=3).min() + 1e-8)
    ) * df['momentum_asymmetry']
    
    # Dynamic Multiplier
    df['dynamic_multiplier'] = df['fractal_volume_exhaustion'] * df['fractal_price_eff']
    
    # Base Signal
    df['base_signal'] = (
        df['efficiency_momentum_core'] * 
        df['microstructure_volume_alignment'] * 
        df['fractal_position_signal']
    )
    
    # Regime Classification
    conditions = [
        (df['fractal_price_eff'] > 0.7) & (df['efficiency_decay'] > 0),
        (df['fractal_price_eff'] >= 0.3) & (df['fractal_price_eff'] <= 0.7),
        (df['fractal_price_eff'] < 0.3) & (df['efficiency_decay'] < 0),
        (df['fractal_volume_exhaustion'] == -1) & (df['fractal_price_eff'] < 0.2)
    ]
    choices = [1.2, 1.0, 0.8, 0.6]
    df['regime_multiplier'] = np.select(conditions, choices, default=1.0)
    
    # Final Alpha
    df['final_alpha'] = df['base_signal'] * df['regime_multiplier'] * df['dynamic_multiplier']
    
    return df['final_alpha']
