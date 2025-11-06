import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Multi-Scale Entropy-Momentum with Pressure-Regime Adaptive Dynamics
    """
    data = df.copy()
    
    # Entropy-Momentum Asymmetry
    # Momentum divergence: sign(Short-term momentum) × (Short-term momentum - Medium-term momentum)
    short_momentum = data['close'] / data['close'].shift(5) - 1
    medium_momentum = data['close'] / data['close'].shift(20) - 1
    momentum_divergence = np.sign(short_momentum) * (short_momentum - medium_momentum)
    
    # Nonlinear acceleration: (Close_t / Close_{t-1}) - (Close_{t-1} / Close_{t-2})
    price_ratio_t = data['close'] / data['close'].shift(1)
    price_ratio_t1 = data['close'].shift(1) / data['close'].shift(2)
    nonlinear_acceleration = price_ratio_t - price_ratio_t1
    
    # Pressure-Entropy Dynamics
    # Buy-side pressure: sum of volume when close > open
    # Sell-side pressure: sum of volume when close < open
    buy_pressure = np.where(data['close'] > data['open'], data['volume'], 0)
    sell_pressure = np.where(data['close'] < data['open'], data['volume'], 0)
    
    # Net pressure ratio: (Buy-side pressure - Sell-side pressure) / (Buy-side pressure + Sell-side pressure)
    net_pressure_ratio = (buy_pressure - sell_pressure) / (buy_pressure + sell_pressure + 1e-8)
    
    # Price-Volume Entropy: -Σ[(Price_i - Avg_Price)² * Volume_i] / Total_Volume for i=t-4 to t
    def calculate_entropy(window):
        prices = window['close']
        volumes = window['volume']
        avg_price = prices.mean()
        weighted_variance = ((prices - avg_price) ** 2 * volumes).sum()
        total_volume = volumes.sum()
        if total_volume == 0:
            return 0
        return -weighted_variance / total_volume
    
    price_volume_entropy = data.rolling(window=5, min_periods=3).apply(
        lambda x: calculate_entropy(pd.DataFrame({
            'close': x[:len(x)//2], 
            'volume': x[len(x)//2:]
        })), raw=False
    ).iloc[:, 0]
    
    # Volatility-Entropy Classification
    # Price path volatility: (|High_t - Close_t| + |Close_t - Low_t|) / (High_t - Low_t)
    price_path_volatility = (np.abs(data['high'] - data['close']) + np.abs(data['close'] - data['low'])) / (data['high'] - data['low'] + 1e-8)
    
    # Volume-weighted dispersion: (High_t - Low_t) × Volume_t / Amount_t
    volume_weighted_dispersion = (data['high'] - data['low']) * data['volume'] / (data['amount'] + 1e-8)
    
    # Fractal Asymmetry Structure
    # Micro-macro coherence: Short-term momentum / Medium-term momentum
    micro_macro_coherence = short_momentum / (medium_momentum + 1e-8)
    
    # Entropy-momentum correlation: Price-Volume Entropy × Momentum divergence
    entropy_momentum_correlation = price_volume_entropy * momentum_divergence
    
    # Order Flow Breakout
    # Breakout asymmetry: (Buy-side pressure - Sell-side pressure) / (High_t - Low_t)
    breakout_asymmetry = (buy_pressure - sell_pressure) / (data['high'] - data['low'] + 1e-8)
    
    # Range compression: (High_t - Low_t) / Close_{t-1}
    range_compression = (data['high'] - data['low']) / data['close'].shift(1)
    
    # Final Composite Alpha with Regime Adaptation
    # Entropy-pressure alignment
    entropy_pressure_alignment = price_volume_entropy * net_pressure_ratio
    
    # Regime classification
    volatility_regime = price_path_volatility.rolling(window=10, min_periods=5).mean()
    high_vol_threshold = volatility_regime.quantile(0.7)
    low_vol_threshold = volatility_regime.quantile(0.3)
    
    # Composite alpha components
    breakout_regime_component = breakout_asymmetry * entropy_pressure_alignment
    high_vol_component = momentum_divergence * volume_weighted_dispersion
    low_vol_component = nonlinear_acceleration * range_compression
    default_component = micro_macro_coherence * price_volume_entropy
    
    # Regime-adaptive final alpha
    final_alpha = np.where(
        volatility_regime > high_vol_threshold, high_vol_component,
        np.where(
            volatility_regime < low_vol_threshold, low_vol_component,
            np.where(
                np.abs(breakout_asymmetry) > breakout_asymmetry.rolling(window=10).std(),
                breakout_regime_component,
                default_component
            )
        )
    )
    
    return pd.Series(final_alpha, index=data.index)
