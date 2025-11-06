import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Hierarchical Fractal Volatility with Asymmetric Order Flow alpha factor
    """
    data = df.copy()
    
    # Multi-Timeframe Volatility-Fractal Alignment
    # Fractal Momentum with Volatility Context
    # Short-Term Fractal Momentum (5-day)
    upper_shadow_momentum = (
        (data['high'].rolling(window=5).apply(lambda x: ((x - data.loc[x.index, 'close']) / 
                                                       (data.loc[x.index, 'high'] - data.loc[x.index, 'low'])).sum(), 
                                           raw=False))
    )
    lower_shadow_momentum = (
        (data['high'].rolling(window=5).apply(lambda x: ((data.loc[x.index, 'close'] - data.loc[x.index, 'low']) / 
                                                       (data.loc[x.index, 'high'] - data.loc[x.index, 'low'])).sum(), 
                                           raw=False))
    )
    net_fractal_momentum = lower_shadow_momentum - upper_shadow_momentum
    
    # Volatility-Weighted Fractal Signals
    trading_range_volatility = (data['high'] - data['low']) / data['close']
    volatility_adjusted_momentum = net_fractal_momentum / (1 + trading_range_volatility)
    fractal_persistence = np.sign(net_fractal_momentum) * np.abs(net_fractal_momentum)
    
    # Cross-Timeframe Volatility Alignment
    short_term_volatility = (data['high'] - data['low']) / data['close']
    medium_term_volatility = ((data['high'] - data['low']) / data['close']).rolling(window=5).mean()
    volatility_momentum = short_term_volatility / medium_term_volatility
    
    # Volume-Volatility Asymmetry Analysis
    # Asymmetric Volume Efficiency
    bullish_efficiency = np.where(data['close'] > data['open'], 
                                data['volume'] / (data['high'] - data['low']), 0)
    bearish_efficiency = np.where(data['close'] < data['open'], 
                                data['volume'] / (data['high'] - data['low']), 0)
    efficiency_asymmetry = bullish_efficiency / np.where(bearish_efficiency == 0, 1, bearish_efficiency)
    
    # Volume Concentration Patterns
    volume_skewness = (data['volume'] - data['volume'].rolling(window=5).mean()) / data['volume'].rolling(window=5).std()
    volume_clustering = data['volume'] / data['volume'].rolling(window=5).max()
    volume_persistence = np.sign(data['volume'] - data['volume'].shift(1)) * volume_clustering
    
    # Price-Volume Divergence with Volatility
    # Volatility-adjusted divergence
    short_term_divergence = (data['volume'] / data['volume'].rolling(window=3).mean()) - (data['close'] / data['close'].shift(3) - 1)
    medium_term_divergence = (data['volume'] / data['volume'].rolling(window=8).mean()) - (data['close'] / data['close'].shift(8) - 1)
    divergence_magnitude = np.abs(short_term_divergence) + np.abs(medium_term_divergence)
    
    # Fractal Order Flow with Volatility
    upper_fractal_flow = np.where(data['close'] < data['high'].rolling(window=5).max(),
                                np.abs(data['close'] - data['high'].rolling(window=5).max()) / ((data['high'] - data['low'])/2), 0)
    lower_fractal_flow = np.where(data['close'] > data['low'].rolling(window=5).min(),
                                np.abs(data['close'] - data['low'].rolling(window=5).min()) / ((data['high'] - data['low'])/2), 0)
    net_fractal_flow = lower_fractal_flow - upper_fractal_flow
    
    # Efficiency-Weighted Volatility Signals
    range_efficiency = np.abs(data['close'] / data['open'] - 1) / (data['high'] - data['low'])
    volume_efficiency = data['volume'] / (data['high'] - data['low'])
    combined_efficiency = range_efficiency * volume_efficiency
    
    # Microstructure Noise and Trade Dynamics
    # Trade Size Distribution Analysis
    avg_trade_size = data['amount'] / data['volume']
    trade_size_momentum = avg_trade_size / (data['amount'].rolling(window=5).mean() / data['volume'].rolling(window=5).mean())
    trade_size_concentration = data['volume'] / (avg_trade_size * 1000)
    
    # Market Friction Measurement
    spread_pressure = np.where(data['close'] > data['open'], (data['high'] - data['open']) / data['close'], 0)
    price_impact = (data['high'] - data['low']) / data['volume']
    transaction_asymmetry = np.abs(data['close'] - data['open']) / (data['high'] - data['low'])
    
    # Volume Stress Detection
    volume_stress = data['volume'] / data['volume'].rolling(window=20).max()
    volume_acceleration = data['volume'] / data['volume'].rolling(window=5).mean()
    volume_exhaustion = np.where(data['volume'].shift(1) > data['volume'].rolling(window=5).mean().shift(1),
                               data['volume'] / data['volume'].shift(1), 1)
    
    # Volatility-Regime Adaptive Synthesis
    # Multi-Dimensional Regime Detection
    # Volatility Regime Classification
    high_volatility = trading_range_volatility > trading_range_volatility.rolling(window=20).mean()
    low_volatility = trading_range_volatility <= trading_range_volatility.rolling(window=20).mean()
    volatility_transition = np.abs(trading_range_volatility / trading_range_volatility.rolling(window=5).mean() - 1)
    
    # Efficiency Regime Assessment
    high_efficiency = efficiency_asymmetry > 1
    low_efficiency = efficiency_asymmetry <= 1
    efficiency_stability = np.abs(efficiency_asymmetry - 1)
    
    # Volume Regime Identification
    high_volume_stress = volume_stress > 1.5
    normal_volume = volume_stress <= 1.5
    volume_regime_strength = volume_stress * volume_acceleration
    
    # Component definitions
    momentum_component = volatility_adjusted_momentum * fractal_persistence
    volume_component = efficiency_asymmetry * volume_persistence
    microstructure_component = trade_size_momentum * transaction_asymmetry
    divergence_component = net_fractal_flow * divergence_magnitude
    
    # Regime-Based Component Weighting
    regime_factor = pd.Series(index=data.index, dtype=float)
    
    # High Volatility & High Efficiency Regime
    mask1 = high_volatility & high_efficiency
    regime_factor[mask1] = (momentum_component * 0.4 + volume_component * 0.3 + 
                          microstructure_component * 0.2 + divergence_component * 0.1)
    
    # Low Volatility & Low Efficiency Regime
    mask2 = low_volatility & low_efficiency
    regime_factor[mask2] = (momentum_component * 0.2 + volume_component * 0.3 + 
                          microstructure_component * 0.3 + divergence_component * 0.2)
    
    # Transition Regime (High Volume Stress)
    mask3 = high_volume_stress
    regime_factor[mask3] = (momentum_component * 0.3 + volume_component * 0.2 + 
                          microstructure_component * 0.4 + divergence_component * 0.1)
    
    # Default regime (catch remaining cases)
    mask_default = ~(mask1 | mask2 | mask3)
    regime_factor[mask_default] = (momentum_component * 0.3 + volume_component * 0.3 + 
                                 microstructure_component * 0.2 + divergence_component * 0.2)
    
    # Final Alpha Construction
    # Regime-adaptive factor combination
    base_factor = regime_factor
    volatility_adjusted_factor = base_factor * (1 + volatility_momentum)
    volume_confirmed_factor = volatility_adjusted_factor * volume_regime_strength
    
    # Efficiency-based signal refinement
    efficiency_scaled_factor = volume_confirmed_factor / (1 + efficiency_stability)
    range_efficiency_boosted = efficiency_scaled_factor * (1 + range_efficiency)
    volume_efficiency_filtered = range_efficiency_boosted * volume_efficiency
    
    # Dynamic momentum enhancement
    fractal_momentum_reinforcement = net_fractal_momentum * volatility_adjusted_momentum
    divergence_momentum = divergence_component * volume_component
    microstructure_momentum = microstructure_component * trade_size_concentration
    
    final_alpha = (volume_efficiency_filtered + 
                  fractal_momentum_reinforcement + 
                  divergence_momentum + 
                  microstructure_momentum)
    
    return final_alpha
