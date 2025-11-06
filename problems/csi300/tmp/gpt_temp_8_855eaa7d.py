import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    data = df.copy()
    
    # Calculate basic price changes
    data['close_prev'] = data['close'].shift(1)
    data['price_change'] = data['close'] / data['close_prev'] - 1
    
    # 1. Asymmetric Price-Volume Momentum
    # Directional Volume Accumulation
    up_condition = data['close'] > data['close_prev']
    down_condition = data['close'] < data['close_prev']
    
    up_volume = data['volume'].rolling(window=5).apply(
        lambda x: np.sum(x[up_condition.iloc[-5:].values]), raw=False
    )
    down_volume = data['volume'].rolling(window=5).apply(
        lambda x: np.sum(x[down_condition.iloc[-5:].values]), raw=False
    )
    
    volume_asymmetry = (up_volume - down_volume) / (up_volume + down_volume + 1e-8)
    
    # Price Momentum Regimes
    ma_5 = data['close'].rolling(window=5).mean()
    ma_10 = data['close'].rolling(window=10).mean()
    
    bull_regime = (data['close'] > ma_5) & (data['close'] > ma_10)
    bear_regime = (data['close'] < ma_5) & (data['close'] < ma_10)
    transition_regime = ~bull_regime & ~bear_regime
    
    # Regime-Weighted Momentum
    momentum_5d = data['close'] / data['close'].shift(5) - 1
    momentum_10d = data['close'] / data['close'].shift(10) - 1
    
    bull_momentum = momentum_5d * volume_asymmetry
    bear_momentum = momentum_10d * (1 - np.abs(volume_asymmetry))
    transition_momentum = (bull_momentum + bear_momentum) / 2
    
    regime_momentum = np.where(bull_regime, bull_momentum,
                              np.where(bear_regime, bear_momentum, transition_momentum))
    
    # 2. Efficiency-Based Reversal Detection
    # Multi-Scale Efficiency Ratios
    intraday_efficiency = np.abs(data['close'] - data['open']) / (data['high'] - data['low'] + 1e-8)
    
    # 3-day efficiency
    price_range_3d = data['high'].rolling(window=3).max() - data['low'].rolling(window=3).min()
    efficiency_3d = np.abs(data['close'] - data['close'].shift(2)) / (price_range_3d + 1e-8)
    
    # 10-day efficiency
    price_range_10d = data['high'].rolling(window=10).max() - data['low'].rolling(window=10).min()
    efficiency_10d = np.abs(data['close'] - data['close'].shift(9)) / (price_range_10d + 1e-8)
    
    # Efficiency Reversal Patterns
    high_efficiency_reversal = (intraday_efficiency > 0.8) & (efficiency_3d < 0.3)
    low_efficiency_continuation = (intraday_efficiency < 0.2) & (efficiency_3d > 0.6)
    efficiency_divergence = (efficiency_3d / (efficiency_10d + 1e-8)) - 1
    
    # Efficiency-Based Signals
    reversal_signal = high_efficiency_reversal.astype(float) * volume_asymmetry
    continuation_signal = low_efficiency_continuation.astype(float) * (1 - volume_asymmetry)
    divergence_signal = efficiency_divergence * regime_momentum
    
    efficiency_signal = reversal_signal + continuation_signal + divergence_signal
    
    # 3. Amount-Based Order Flow Imbalance
    # Order Flow Components
    amount_avg_5d = data['amount'].rolling(window=5).mean()
    amount_concentration = data['amount'] / (amount_avg_5d + 1e-8)
    volume_adjusted_amount = data['amount'] / (data['volume'] + 1e-8)
    
    # Order flow persistence
    above_avg_amount = data['amount'] > amount_avg_5d
    order_flow_persistence = above_avg_amount.rolling(window=10).sum()
    
    # Order Flow Regimes
    high_concentration_regime = (amount_concentration > 1.5) & (order_flow_persistence > 3)
    low_concentration_regime = (amount_concentration < 0.7) & (order_flow_persistence < 2)
    
    # Order Flow Signals
    high_concentration_signal = volume_adjusted_amount * regime_momentum
    low_concentration_signal = (1 - amount_concentration) * reversal_signal
    
    # Regime-weighted order flow
    order_flow_signal = np.where(high_concentration_regime, high_concentration_signal,
                                np.where(low_concentration_regime, low_concentration_signal,
                                        (high_concentration_signal + low_concentration_signal) / 2))
    
    order_flow_signal = order_flow_signal * (order_flow_persistence / 10)
    
    # 4. Volatility-Adaptive Signal Combination
    # Adaptive Volatility Measures
    # Regime-specific volatility
    returns = data['close'].pct_change()
    bull_volatility = returns.rolling(window=10).std()
    bear_volatility = returns.rolling(window=10).std()
    transition_volatility = returns.rolling(window=10).std()
    
    regime_volatility = np.where(bull_regime, bull_volatility,
                                np.where(bear_regime, bear_volatility, transition_volatility))
    
    # True Range and efficiency-adjusted volatility
    tr1 = data['high'] - data['low']
    tr2 = np.abs(data['high'] - data['close'].shift(1))
    tr3 = np.abs(data['low'] - data['close'].shift(1))
    true_range = np.maximum(tr1, np.maximum(tr2, tr3))
    
    tr_avg_5d = true_range.rolling(window=5).mean()
    efficiency_adjusted_vol = true_range / (tr_avg_5d + 1e-8)
    
    # Volume-volatility correlation
    volume_change = data['volume'].pct_change()
    volume_vol_corr = volume_change.rolling(window=10).corr(true_range.pct_change())
    
    # Signal Robustness Scores
    momentum_robustness = np.abs(regime_momentum) / (regime_volatility + 1e-8)
    efficiency_robustness = efficiency_divergence / (efficiency_adjusted_vol + 1e-8)
    order_flow_robustness = order_flow_signal * (1 + volume_vol_corr)
    
    # Adaptive Weighting
    momentum_weighted = regime_momentum * momentum_robustness
    efficiency_weighted = efficiency_signal * efficiency_robustness
    order_flow_weighted = order_flow_signal * order_flow_robustness
    
    # 5. Final Multi-Regime Factor Synthesis
    # Regime-Based Signal Integration
    bull_weight = np.where(bull_regime, 0.5, 0.2)
    bear_weight = np.where(bear_regime, 0.5, 0.2)
    transition_weight = np.where(transition_regime, 0.6, 0.3)
    
    # Volatility adjustment
    vol_adjustment = 1 / (regime_volatility + 0.1)
    
    # Combine signals with regime-dependent weights
    bull_component = (momentum_weighted * 0.4 + order_flow_weighted * 0.4 + efficiency_weighted * 0.2) * bull_weight
    bear_component = (efficiency_weighted * 0.4 + momentum_weighted * 0.3 + order_flow_weighted * 0.3) * bear_weight
    transition_component = (momentum_weighted * 0.33 + efficiency_weighted * 0.33 + order_flow_weighted * 0.34) * transition_weight
    
    final_factor = (bull_component + bear_component + transition_component) * vol_adjustment
    
    # Generate confidence scores
    signal_strength = np.abs(momentum_weighted) + np.abs(efficiency_weighted) + np.abs(order_flow_weighted)
    regime_consistency = (bull_regime.astype(int) + bear_regime.astype(int) + transition_regime.astype(int))
    
    confidence_score = signal_strength * regime_consistency * (1 - regime_volatility)
    
    # Final alpha factor with confidence adjustment
    alpha_factor = final_factor * np.tanh(confidence_score)
    
    return pd.Series(alpha_factor, index=data.index)
