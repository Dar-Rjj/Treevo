import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Market Microstructure Regime Detection
    # Bid-Ask Spread Proxy
    data['mid_price'] = (data['high'] + data['low']) / 2
    data['effective_spread'] = 2 * np.abs(data['mid_price'] - data['close']) / data['mid_price']
    data['spread_momentum'] = data['effective_spread'] - data['effective_spread'].shift(1)
    data['spread_volatility'] = data['effective_spread'].rolling(window=10, min_periods=5).std()
    
    # Volume Concentration Analysis
    data['large_trade_concentration'] = data['amount'] / (data['volume'] * data['close'])
    data['volume_clustering'] = data['large_trade_concentration'].rolling(window=5, min_periods=3).std()
    data['microstructure_noise_ratio'] = data['volume_clustering'] / data['spread_volatility']
    
    # Regime Detection based on microstructure noise
    conditions = [
        data['microstructure_noise_ratio'] > data['microstructure_noise_ratio'].rolling(window=20, min_periods=10).quantile(0.7),
        data['microstructure_noise_ratio'] < data['microstructure_noise_ratio'].rolling(window=20, min_periods=10).quantile(0.3)
    ]
    choices = ['high_noise', 'low_noise']
    data['regime'] = np.select(conditions, choices, default='normal')
    
    # Regime-Dependent Price Impact Modeling
    # Permanent vs Transient Impact
    data['permanent_impact'] = data['close'].rolling(window=10, min_periods=5).corr(data['volume'] * (data['close'] - data['open']))
    data['transient_impact'] = (data['close'] - data['open']) / np.sqrt(data['volume'])
    data['impact_ratio'] = data['permanent_impact'] / np.abs(data['transient_impact'])
    
    # Regime-Adaptive Impact Signals
    data['regime_impact_signal'] = np.where(
        data['regime'] == 'high_noise',
        data['impact_ratio'] * data['microstructure_noise_ratio'],
        np.where(
            data['regime'] == 'low_noise',
            data['impact_ratio'] / data['microstructure_noise_ratio'],
            data['impact_ratio']
        )
    )
    
    # Order Flow Imbalance Dynamics
    # Intraday Pressure Indicators
    data['opening_pressure'] = (data['open'] - data['close'].shift(1)) / (data['high'] - data['low'])
    data['closing_pressure'] = (data['close'] - data['open']) / (data['high'] - data['low'])
    data['pressure_divergence'] = data['opening_pressure'] - data['closing_pressure']
    
    # Flow Acceleration Patterns
    data['volume_acceleration'] = data['volume'] / data['volume'].shift(1) - 1
    data['amount_acceleration'] = data['amount'] / data['amount'].shift(1) - 1
    data['flow_momentum'] = data['volume_acceleration'] * data['amount_acceleration']
    
    # Multi-Timeframe Regime Consistency
    # Short-term Regime Stability
    regime_persistence = []
    for i in range(len(data)):
        if i < 2:
            regime_persistence.append(0)
        else:
            current_regime = data['regime'].iloc[i]
            past_regimes = [data['regime'].iloc[i-1], data['regime'].iloc[i-2]]
            persistence = sum(1 for regime in past_regimes if regime == current_regime)
            regime_persistence.append(persistence)
    
    data['regime_persistence'] = regime_persistence
    data['regime_transition_prob'] = 1 / (1 + data['regime_persistence'])
    data['short_term_confidence'] = 1 - data['regime_transition_prob']
    
    # Long-term Regime Memory
    regime_entropy = []
    for i in range(len(data)):
        if i < 9:
            regime_entropy.append(0)
        else:
            recent_regimes = data['regime'].iloc[i-9:i+1]
            regime_counts = recent_regimes.value_counts()
            probabilities = regime_counts / len(recent_regimes)
            entropy = -np.sum(probabilities * np.log(probabilities + 1e-10))
            max_entropy = np.log(len(probabilities))
            normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
            regime_entropy.append(normalized_entropy)
    
    data['regime_entropy'] = regime_entropy
    data['regime_predictability'] = 1 - data['regime_entropy']
    data['multi_scale_consistency'] = data['short_term_confidence'] * data['regime_predictability']
    
    # Adaptive Alpha Synthesis
    # Core Signal Construction
    microstructure_component = data['regime_impact_signal']
    flow_dynamics = data['pressure_divergence'] * data['flow_momentum']
    regime_quality = data['multi_scale_consistency']
    
    # Dynamic Weighting Scheme
    signal_reliability_weighting = microstructure_component * flow_dynamics * regime_quality
    noise_adjusted_combination = (microstructure_component * flow_dynamics) / data['microstructure_noise_ratio']
    
    # Final alpha
    final_alpha = noise_adjusted_combination * signal_reliability_weighting
    
    # Return the alpha factor series
    return final_alpha
