import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Multi-Scale Fractal Efficiency
    # Ultra-Short Efficiency (3-day)
    data['ultra_short_eff'] = np.abs(data['close'] - data['close'].shift(3)) / (
        data['high'].rolling(window=3).max() - data['low'].rolling(window=3).min()
    )
    
    # Short-Term Efficiency (7-day)
    data['short_term_eff'] = np.abs(data['close'] - data['close'].shift(7)) / (
        data['high'].rolling(window=7).max() - data['low'].rolling(window=7).min()
    )
    
    # Efficiency Momentum
    data['efficiency_momentum'] = (data['short_term_eff'] - data['ultra_short_eff']) * np.sign(data['close'] - data['close'].shift(1))
    
    # Fractal Opening Efficiency
    data['fractal_opening_eff'] = np.abs(data['open'] - data['close'].shift(1)) / (data['high'] - data['low'])
    
    # Asymmetric Fractal Momentum
    # Volatility Asymmetry
    data['volatility_asymmetry'] = (data['high'].rolling(window=5).max() - data['close'].shift(4)) / (
        data['close'].shift(4) - data['low'].rolling(window=5).min()
    )
    
    # Momentum Asymmetry
    returns = data['close'].pct_change()
    positive_returns = returns.rolling(window=7).apply(lambda x: x[x > 0].sum(), raw=False)
    negative_returns = returns.rolling(window=7).apply(lambda x: x[x < 0].sum(), raw=False)
    data['momentum_asymmetry'] = positive_returns / (positive_returns + np.abs(negative_returns))
    
    # Fractal Opening Skew
    data['fractal_opening_skew'] = ((data['high'] - data['open']) / (data['high'] - data['low'])) - (
        (data['open'] - data['low']) / (data['high'] - data['low'])
    )
    
    # Volatility-Momentum Integration
    data['vol_momentum_integration'] = data['momentum_asymmetry'] * data['volatility_asymmetry'] * np.sign(data['close'] - data['open'])
    
    # Fractal Microstructure Dynamics
    # Opening Fracture Momentum
    data['opening_fracture_momentum'] = ((data['open'] - data['low']) - (data['high'] - data['open'])) * (
        data['close'] - data['close'].shift(1)
    ) / np.abs(data['close'] - data['close'].shift(1))
    
    # Fractal Closing Momentum
    data['fractal_closing_momentum'] = (data['close'] - data['open']) / (
        data['high'].rolling(window=2).max() - data['low'].rolling(window=2).min()
    )
    
    # Price Fractal Convergence
    short_fractal = np.abs(data['close'] - data['close'].shift(5)) / (
        data['high'].rolling(window=5).max() - data['low'].rolling(window=5).min()
    )
    medium_fractal = np.abs(data['close'] - data['close'].shift(10)) / (
        data['high'].rolling(window=10).max() - data['low'].rolling(window=10).min()
    )
    data['price_fractal_convergence'] = np.abs(short_fractal - medium_fractal) * -1
    
    # Fractal Auction Pressure
    data['fractal_auction_pressure'] = (data['open'] - data['low'].shift(1)) - (data['high'].shift(1) - data['open'])
    
    # Volume Fractal Behavior
    # Volume Asymmetry Momentum
    up_volume = data['volume'].where(data['close'] > data['close'].shift(1), 0)
    down_volume = data['volume'].where(data['close'] < data['close'].shift(1), 0)
    data['volume_asymmetry_momentum'] = (
        up_volume.rolling(window=7).sum() / down_volume.rolling(window=7).sum()
    ) * np.sign(data['close'] - data['close'].shift(1))
    
    # Fractal Volume Momentum
    volume_increase_count = data['volume'].rolling(window=5).apply(
        lambda x: sum(x[i] > x[i-1] for i in range(1, len(x))), raw=False
    )
    data['fractal_volume_momentum'] = volume_increase_count * np.sign(data['volume'] - data['volume'].shift(1))
    
    # Fractal Volume Shock
    volume_ratio = data['volume'] / data['volume'].shift(1)
    data['fractal_volume_shock'] = np.where(
        volume_ratio > 1.8, -1, np.where(volume_ratio < 0.6, 1, 0)
    )
    
    # Fractal Flow Pressure
    positive_amount = data['amount'].where(data['close'] > data['close'].shift(1), 0)
    negative_amount = data['amount'].where(data['close'] < data['close'].shift(1), 0)
    data['fractal_flow_pressure'] = (
        (positive_amount.rolling(window=5).sum() - negative_amount.rolling(window=5).sum()) / 
        (positive_amount.rolling(window=5).sum() + negative_amount.rolling(window=5).sum())
    ) * (data['volume'] / data['volume'].shift(1))
    
    # Integration Framework
    # Efficiency-Momentum Core
    data['efficiency_momentum_core'] = data['efficiency_momentum'] * data['vol_momentum_integration']
    
    # Microstructure-Volume Alignment
    data['microstructure_volume_alignment'] = (1 - data['price_fractal_convergence']) * data['volume_asymmetry_momentum'] * data['fractal_volume_momentum']
    
    # Fractal Position Signal
    data['fractal_position_signal'] = (
        (data['close'] - data['low'].rolling(window=4).min()) / 
        (data['high'].rolling(window=4).max() - data['low'].rolling(window=4).min())
    ) * data['momentum_asymmetry'] * np.sign(data['close'] - data['open'])
    
    # Dynamic Multiplier
    data['dynamic_multiplier'] = data['fractal_volume_shock'] * data['fractal_opening_eff']
    
    # Base Signal
    data['base_signal'] = data['efficiency_momentum_core'] * data['microstructure_volume_alignment'] * data['fractal_position_signal']
    
    # Regime Classification
    regime_multiplier = np.where(
        (data['fractal_opening_eff'] > 0.6) & (data['efficiency_momentum'] > 0), 1.3,
        np.where(
            (data['fractal_opening_eff'] >= 0.4) & (data['fractal_opening_eff'] <= 0.6), 1.0,
            np.where(
                (data['fractal_opening_eff'] < 0.4) & (data['efficiency_momentum'] < 0), 0.7,
                np.where(
                    (data['fractal_volume_shock'] == -1) & (data['fractal_opening_eff'] < 0.3), 0.5, 1.0
                )
            )
        )
    )
    
    # Final Alpha
    data['final_alpha'] = data['base_signal'] * regime_multiplier * data['dynamic_multiplier']
    
    return data['final_alpha']
