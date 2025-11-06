import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Cross-Association Momentum & Liquidity Regime Synthesis Alpha Factor
    """
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Liquidity Regime Identification & Characterization
    # Trade-Level Liquidity Metrics
    data['bid_ask_spread_proxy'] = (data['high'] - data['low']) / data['close']
    data['price_impact_proxy'] = abs(data['close'] - data['close'].shift(1)) / data['amount']
    data['market_depth_proxy'] = data['volume'] / (data['high'] - data['low'])
    data['liquidity_composite'] = data['market_depth_proxy'] / (data['bid_ask_spread_proxy'] * data['price_impact_proxy'])
    
    # Liquidity Regime Classification
    data['liquidity_composite_median_5d'] = data['liquidity_composite'].rolling(window=5, min_periods=3).median()
    data['market_depth_median_5d'] = data['market_depth_proxy'].rolling(window=5, min_periods=3).median()
    data['price_impact_median_5d'] = data['price_impact_proxy'].rolling(window=5, min_periods=3).median()
    
    conditions = [
        (data['liquidity_composite'] > 1.2 * data['liquidity_composite_median_5d']) & 
        (data['market_depth_proxy'] > 1.1 * data['market_depth_median_5d']),
        (data['liquidity_composite'] < 0.8 * data['liquidity_composite_median_5d']) & 
        (data['price_impact_proxy'] > 1.1 * data['price_impact_median_5d'])
    ]
    choices = [2, 0]  # 2: High, 0: Low, 1: Normal
    data['liquidity_regime'] = np.select(conditions, choices, default=1)
    
    # Cross-Association Momentum Patterns
    # Price-Volume Association Dynamics
    data['volume_price_correlation'] = np.sign(data['close'] - data['close'].shift(1)) * (data['volume'] / data['volume'].shift(1))
    data['acceleration_association'] = ((data['close']/data['close'].shift(1) - data['close'].shift(1)/data['close'].shift(2)) * 
                                      (data['volume']/data['volume'].shift(1) - data['volume'].shift(1)/data['volume'].shift(2)))
    
    # Association Persistence
    price_changes = np.sign(data['close'] - data['close'].shift(1))
    volume_changes = np.sign(data['volume'] - data['volume'].shift(1))
    same_sign_count = pd.Series(index=data.index, dtype=float)
    for i in range(2, len(data)):
        if i >= 2:
            window_start = max(0, i-2)
            window_end = i+1
            same_sign = (price_changes.iloc[window_start:window_end] * volume_changes.iloc[window_start:window_end] > 0)
            same_sign_count.iloc[i] = same_sign.sum()
    data['association_persistence'] = same_sign_count
    
    # Amount-Price Interaction Patterns
    data['amount_median_5d'] = data['amount'].rolling(window=5, min_periods=3).median()
    data['large_trade_dominance'] = data['amount'] / data['amount_median_5d']
    data['price_efficiency_per_amount'] = abs(data['close'] - data['close'].shift(1)) / data['amount']
    data['amount_momentum_divergence'] = ((data['close']/data['close'].shift(1) - 1) / data['large_trade_dominance'])
    
    # Multi-timeframe Association Synthesis
    data['short_term_strength'] = data['volume_price_correlation'].rolling(window=1).mean()
    data['medium_term_consistency'] = data['volume_price_correlation'].rolling(window=3, min_periods=2).std()
    data['association_trend'] = data['short_term_strength'] / (data['medium_term_consistency'] + 1e-8)
    
    # Regime-Adaptive Momentum Enhancement
    momentum_signals = pd.Series(index=data.index, dtype=float)
    
    # High Liquidity Regime Processing
    high_liquidity_mask = data['liquidity_regime'] == 2
    momentum_signals[high_liquidity_mask] = (
        data.loc[high_liquidity_mask, 'volume_price_correlation'] * 
        data.loc[high_liquidity_mask, 'association_persistence'] * 
        np.where(data.loc[high_liquidity_mask, 'acceleration_association'] > 0, 1, 0) *
        data.loc[high_liquidity_mask, 'market_depth_proxy']
    )
    
    # Low Liquidity Regime Processing
    low_liquidity_mask = data['liquidity_regime'] == 0
    momentum_signals[low_liquidity_mask] = (
        data.loc[low_liquidity_mask, 'large_trade_dominance'] *
        (-data.loc[low_liquidity_mask, 'price_efficiency_per_amount']) *  # Inverted for mean reversion
        0.7  # Regime penalty
    )
    
    # Normal Liquidity Regime Processing
    normal_liquidity_mask = data['liquidity_regime'] == 1
    momentum_signals[normal_liquidity_mask] = (
        (data.loc[normal_liquidity_mask, 'volume_price_correlation'] +
         data.loc[normal_liquidity_mask, 'acceleration_association'] +
         data.loc[normal_liquidity_mask, 'amount_momentum_divergence']) *
        data.loc[normal_liquidity_mask, 'association_trend']
    )
    
    # Microstructure Quality & Persistence Assessment
    # Trade Flow Quality Metrics
    data['volume_amount_ratio'] = data['volume'] / data['amount']
    data['volume_amount_median_5d'] = data['volume_amount_ratio'].rolling(window=5, min_periods=3).median()
    data['volume_amount_consistency'] = data['volume_amount_ratio'] / data['volume_amount_median_5d']
    
    # Price Continuity
    price_direction = np.sign(data['close'] - data['close'].shift(1))
    price_continuity = pd.Series(index=data.index, dtype=float)
    for i in range(len(data)):
        if i == 0:
            price_continuity.iloc[i] = 0
        else:
            count = 0
            for j in range(min(5, i), 0, -1):
                if price_direction.iloc[i] == price_direction.iloc[i-j]:
                    count += 1
                else:
                    break
            price_continuity.iloc[i] = min(count, 5)
    
    data['price_continuity'] = price_continuity
    
    # Trade Size Distribution
    data['trade_size'] = data['amount'] / data['volume']
    data['trade_size_median_5d'] = data['trade_size'].rolling(window=5, min_periods=3).median()
    data['trade_size_distribution'] = data['trade_size'] / data['trade_size_median_5d']
    
    # Momentum Sustainability Indicators
    data['association_stability'] = data['volume_price_correlation'].rolling(window=5, min_periods=3).var()
    
    # Regime Persistence
    regime_persistence = pd.Series(index=data.index, dtype=float)
    current_regime = None
    persistence_count = 0
    for i in range(len(data)):
        if data['liquidity_regime'].iloc[i] == current_regime:
            persistence_count += 1
        else:
            current_regime = data['liquidity_regime'].iloc[i]
            persistence_count = 1
        regime_persistence.iloc[i] = persistence_count
    data['regime_persistence'] = regime_persistence
    
    data['quality_score'] = (1 / (data['association_stability'] + 1e-8)) * data['regime_persistence']
    
    # Signal Integration & Risk Adjustment
    # Base Signal Construction
    trade_flow_confirmation = data['volume_amount_consistency'] * data['price_continuity']
    base_signal = (momentum_signals * data['quality_score']) + trade_flow_confirmation
    
    # Risk-Aware Signal Refinement
    volatility_adjustment = 1 / (data['bid_ask_spread_proxy'] + 1e-8)
    regime_confidence = 1 / (1 + data['regime_persistence'])
    
    # Apply filters and adjustments
    final_signal = base_signal.copy()
    final_signal[data['association_stability'] < 0] = 0  # Consistency filter
    final_signal = final_signal * volatility_adjustment * regime_confidence * data['trade_size_distribution']
    
    return final_signal
