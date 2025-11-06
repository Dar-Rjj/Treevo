import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Momentum-Decay Adjusted Order Flow Imbalance alpha factor
    Combines order flow analysis with momentum decay framework to predict future returns
    """
    
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # 1. Order Flow Imbalance Calculation
    # Intraday Price-Volume Signature using OHLC data
    data['price_change'] = data['close'] - data['open']
    data['range'] = data['high'] - data['low']
    
    # Classify volume as buyer/seller initiated based on price action
    data['buy_volume'] = np.where(data['price_change'] > 0, data['volume'], 0)
    data['sell_volume'] = np.where(data['price_change'] < 0, data['volume'], 0)
    
    # Net order flow imbalance
    data['net_flow'] = (data['buy_volume'] - data['sell_volume']) / (data['buy_volume'] + data['sell_volume'] + 1e-8)
    
    # Volume-weighted price impact
    data['price_impact'] = data['price_change'] / (data['volume'] + 1e-8)
    data['buy_impact'] = np.where(data['price_change'] > 0, data['price_impact'], 0)
    data['sell_impact'] = np.where(data['price_change'] < 0, -data['price_impact'], 0)
    
    # Intraday flow persistence (autocorrelation)
    data['flow_persistence'] = data['net_flow'].rolling(window=3, min_periods=1).apply(
        lambda x: np.corrcoef(x[:-1], x[1:])[0,1] if len(x) > 1 and not np.isnan(x).any() else 0
    )
    
    # 2. Multi-Timeframe Flow Accumulation
    # Short-term flow (1-day)
    data['short_term_flow'] = data['net_flow'].rolling(window=3, min_periods=1).mean()
    data['flow_intensity'] = data['volume'].rolling(window=3, min_periods=1).mean() / data['volume'].rolling(window=20, min_periods=1).mean()
    
    # Medium-term flow (5-day)
    data['medium_term_flow'] = data['net_flow'].rolling(window=5, min_periods=1).mean()
    data['flow_trend'] = data['net_flow'].rolling(window=5, min_periods=1).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) >= 2 else 0
    )
    
    # Flow divergence detection
    data['flow_divergence'] = data['short_term_flow'] - data['medium_term_flow']
    
    # 3. Momentum Decay Framework
    # Recent return momentum (3-day)
    data['returns_3d'] = data['close'].pct_change(periods=3)
    data['momentum_strength_3d'] = data['returns_3d'].abs()
    
    # Medium-term momentum (10-day)
    data['returns_10d'] = data['close'].pct_change(periods=10)
    data['momentum_strength_10d'] = data['returns_10d'].abs()
    
    # Momentum decay rate
    data['momentum_decay'] = (data['returns_3d'] - data['returns_10d'].shift(7)).fillna(0)
    
    # 4. Volume-Momentum Interaction
    # Volume confirmation of momentum
    data['volume_confirmation'] = (data['momentum_strength_3d'] * data['flow_intensity']).fillna(0)
    
    # Momentum exhaustion detection
    data['momentum_exhaustion'] = np.where(
        (data['returns_3d'] > 0) & (data['flow_intensity'] < data['flow_intensity'].rolling(window=5, min_periods=1).mean()),
        -1, 0
    ) + np.where(
        (data['returns_3d'] < 0) & (data['flow_intensity'] < data['flow_intensity'].rolling(window=5, min_periods=1).mean()),
        1, 0
    )
    
    # 5. Decay-Adjusted Flow Imbalance
    # Momentum-adjusted flow strength
    data['flow_efficiency'] = data['net_flow'] / (data['momentum_strength_3d'] + 1e-8)
    
    # Flow resilience during decay
    data['flow_resilience'] = data['net_flow'].rolling(window=3, min_periods=1).std() / (data['momentum_decay'].abs() + 1e-8)
    
    # Decay-compensated flow
    data['decay_compensated_flow'] = data['net_flow'] - (data['momentum_decay'] * data['flow_efficiency'])
    
    # 6. Multi-Timeframe Flow Decay
    # Short-term flow decay (3-day)
    data['flow_decay_3d'] = data['net_flow'].diff(periods=3) / (data['net_flow'].rolling(window=3, min_periods=1).std() + 1e-8)
    
    # Medium-term flow decay (10-day)
    data['flow_decay_10d'] = data['net_flow'].diff(periods=10) / (data['net_flow'].rolling(window=10, min_periods=1).std() + 1e-8)
    
    # Flow decay acceleration
    data['flow_decay_accel'] = data['flow_decay_3d'] - data['flow_decay_10d'].shift(7).fillna(0)
    
    # 7. Predictive Signal Generation
    # Flow-momentum convergence
    data['flow_momentum_convergence'] = (
        np.sign(data['net_flow']) * np.sign(data['returns_3d']) * 
        (data['momentum_strength_3d'] * data['net_flow'].abs())
    )
    
    # Divergent flow-momentum
    data['flow_momentum_divergence'] = (
        np.sign(data['net_flow']) * np.sign(data['returns_3d']) * -1 * 
        (data['momentum_strength_3d'] * data['net_flow'].abs())
    )
    
    # Convergence-divergence momentum
    data['convergence_strength'] = data['flow_momentum_convergence'].rolling(window=5, min_periods=1).mean()
    data['divergence_intensity'] = data['flow_momentum_divergence'].rolling(window=5, min_periods=1).mean()
    
    # 8. Composite Alpha Construction
    # Momentum-decay flow imbalance score
    momentum_decay_weight = 1 / (1 + np.exp(-data['momentum_decay']))
    data['momentum_decay_flow_score'] = (
        data['decay_compensated_flow'] * momentum_decay_weight +
        data['convergence_strength'] * 0.3 +
        data['divergence_intensity'] * 0.2
    )
    
    # Flow efficiency forecast
    data['flow_efficiency_forecast'] = (
        data['flow_efficiency'].rolling(window=5, min_periods=1).mean() +
        data['flow_decay_accel'] * 0.5
    )
    
    # Final alpha factor - Momentum-Decay Adjusted Order Flow Imbalance
    alpha = (
        data['momentum_decay_flow_score'] * 0.6 +
        data['flow_efficiency_forecast'] * 0.4 +
        data['flow_resilience'] * 0.2 -
        data['momentum_exhaustion'] * 0.3
    )
    
    # Normalize the final alpha factor
    alpha_normalized = (alpha - alpha.rolling(window=20, min_periods=1).mean()) / (alpha.rolling(window=20, min_periods=1).std() + 1e-8)
    
    return alpha_normalized
