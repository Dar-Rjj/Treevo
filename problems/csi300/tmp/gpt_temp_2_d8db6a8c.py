import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Generate novel alpha factor through crossover synthesis of multiple components
    """
    data = df.copy()
    
    # Component 1: Momentum-Volume Efficiency Divergence
    # Price Momentum Efficiency
    price_momentum = data['close'] / data['close'].shift(5) - 1
    max_possible_gain = (data['high'].rolling(5).max() / data['close'].shift(5)) - 1
    movement_efficiency = np.where(max_possible_gain > 0, price_momentum / max_possible_gain, 0)
    
    # Volume Divergence Adjustment
    volume_momentum = data['volume'] / data['volume'].shift(5) - 1
    momentum_divergence = price_momentum / (volume_momentum + 1e-8)
    efficiency_weighted_divergence = momentum_divergence * movement_efficiency
    
    # Component 2: Intraday Range Persistence with Volume Breakout
    # Range Persistence Analysis
    true_range = data['high'] - data['low']
    range_autocorr = true_range.rolling(20).apply(lambda x: x.autocorr(), raw=False)
    range_persistence = range_autocorr.fillna(0)
    
    # Volume Breakout Confirmation
    volume_ma = data['volume'].rolling(20).mean()
    volume_std = data['volume'].rolling(20).std()
    volume_breakout = (data['volume'] - volume_ma) / (volume_std + 1e-8)
    volume_acceleration = data['volume'].pct_change(3)
    
    overnight_gap = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
    range_volume_signal = range_persistence * volume_breakout * (1 + np.abs(overnight_gap))
    
    # Component 3: Amount-Weighted Price Acceleration with Volatility Regulation
    # Price Acceleration Component
    price_velocity = data['close'].pct_change(3)
    price_acceleration = price_velocity.diff(3)
    
    # Amount and Volatility Integration
    amount_trend = data['amount'].pct_change(5)
    price_trend = data['close'].pct_change(5)
    amount_price_divergence = amount_trend - price_trend
    
    amount_volume_efficiency = data['amount'] / (data['volume'] + 1e-8)
    amount_flow_signal = amount_price_divergence * amount_volume_efficiency.pct_change(3)
    
    # Volatility Regulation
    daily_range = data['high'] - data['low']
    volatility = daily_range.rolling(10).std()
    risk_adjusted_acceleration = price_acceleration / (volatility + 1e-8)
    
    amount_weighted_acceleration = risk_adjusted_acceleration * amount_flow_signal
    
    # Component 4: Fractal Momentum with Volume Regime Detection
    # Fractal Price Analysis (simplified Hurst approximation)
    def hurst_approximation(series):
        lags = range(2, 8)
        tau = [np.std(np.subtract(series[lag:], series[:-lag])) for lag in lags]
        poly = np.polyfit(np.log(lags), np.log(tau), 1)
        return poly[0]
    
    hurst_values = data['close'].rolling(30).apply(hurst_approximation, raw=False)
    fractal_momentum = hurst_values * data['close'].pct_change(5)
    
    # Volume Regime Enhancement
    volume_quantiles = data['volume'].rolling(20).quantile(0.7)
    high_volume_regime = (data['volume'] > volume_quantiles).astype(int)
    volume_regime_stability = high_volume_regime.rolling(5).std()
    
    regime_adjusted_fractal = fractal_momentum * (1 + high_volume_regime * 0.5)
    regime_weighted_fractal = regime_adjusted_fractal / (1 + volume_regime_stability)
    
    # Component 5: Liquidity-Confirmed Reversal with Pressure Accumulation
    # Price Reversal Detection
    short_term_return = data['close'].pct_change(3)
    medium_term_return = data['close'].pct_change(8)
    reversal_signal = -short_term_return * (1 + np.sign(short_term_return) * np.sign(medium_term_return))
    
    # Pressure Accumulation Component
    buying_pressure = ((data['close'] > data['close'].shift(1)) * data['volume']).rolling(5).sum()
    selling_pressure = ((data['close'] < data['close'].shift(1)) * data['volume']).rolling(5).sum()
    net_pressure = (buying_pressure - selling_pressure) / (buying_pressure + selling_pressure + 1e-8)
    
    # Liquidity Confirmation
    participation_intensity = data['amount'] / (data['volume'] + 1e-8)
    liquidity_confirmed_reversal = reversal_signal * net_pressure * participation_intensity.pct_change(3)
    
    # Component 6: Efficiency-Weighted Breakout with Range Confirmation
    # Breakout Efficiency Analysis
    trading_range_high = data['high'].rolling(10).max()
    trading_range_low = data['low'].rolling(10).min()
    range_width = trading_range_high - trading_range_low
    
    breakout_attempt = (data['close'] - trading_range_low) / (range_width + 1e-8)
    breakout_efficiency = breakout_attempt * (1 - data['close'].pct_change(3).abs())
    
    # Multi-Dimensional Confirmation
    volume_clustering = (data['volume'] / data['volume'].rolling(5).mean()).rolling(3).std()
    range_expansion = (data['high'] - data['low']).pct_change(3)
    
    enhanced_breakout = breakout_efficiency * (1 + volume_clustering) * (1 + range_expansion)
    
    # Final Alpha Factor Synthesis
    # Combine all components with equal weighting and normalization
    components = [
        efficiency_weighted_divergence,
        range_volume_signal,
        amount_weighted_acceleration,
        regime_weighted_fractal,
        liquidity_confirmed_reversal,
        enhanced_breakout
    ]
    
    # Normalize each component
    normalized_components = []
    for component in components:
        mean_val = component.rolling(20).mean()
        std_val = component.rolling(20).std()
        normalized = (component - mean_val) / (std_val + 1e-8)
        normalized_components.append(normalized)
    
    # Equal-weighted combination
    final_alpha = sum(normalized_components) / len(normalized_components)
    
    return final_alpha
