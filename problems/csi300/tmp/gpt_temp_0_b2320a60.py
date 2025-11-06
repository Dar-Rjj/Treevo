import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Generate alpha factor combining regime-adaptive microstructure reversion,
    volume-efficiency range breakout, skewness-order flow divergence, and 
    multi-scale microstructure reversion signals.
    """
    data = df.copy()
    
    # 1. Regime-Adaptive Microstructure Reversion
    # Volatility regime detection
    data['intraday_range'] = (data['high'] - data['low']) / data['close']
    data['overnight_gap'] = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
    data['volatility_regime'] = data['intraday_range'].rolling(window=5).std() / data['overnight_gap'].abs().rolling(window=5).std()
    
    # Multi-timeframe momentum
    data['momentum_3d'] = data['close'] / data['close'].shift(3) - 1
    data['momentum_8d'] = data['close'] / data['close'].shift(8) - 1
    data['momentum_15d'] = data['close'] / data['close'].shift(15) - 1
    
    # Order flow imbalance (simplified using volume and price movement)
    data['price_change'] = data['close'] - data['open']
    data['order_flow_imbalance'] = np.where(data['price_change'] > 0, 
                                          data['volume'], 
                                          -data['volume']) / data['volume'].rolling(window=10).mean()
    
    # Regime-dependent reversion weighting
    data['reversion_signal'] = -data['momentum_3d'] * (1 + data['volatility_regime'])
    
    # 2. Volume-Efficiency Range Breakout
    # Movement efficiency using true range
    data['true_range'] = np.maximum(data['high'] - data['low'], 
                                  np.maximum(abs(data['high'] - data['close'].shift(1)), 
                                           abs(data['low'] - data['close'].shift(1))))
    data['movement_efficiency'] = (data['close'] - data['open']) / data['true_range']
    
    # Volume regime
    data['volume_regime'] = data['volume'] / data['volume'].rolling(window=20).mean()
    
    # Order flow persistence
    data['volume_trend'] = data['volume'].rolling(window=5).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0])
    data['order_flow_persistence'] = data['order_flow_imbalance'].rolling(window=3).mean()
    
    # Range breakout detection
    data['range_breakout'] = ((data['close'] > data['high'].rolling(window=5).max()) | 
                            (data['close'] < data['low'].rolling(window=5).min())).astype(int)
    data['volume_breakout'] = data['range_breakout'] * data['volume_regime']
    
    # 3. Skewness-Order Flow Divergence
    # Regime-adjusted skewness
    data['returns_5d'] = data['close'].pct_change(periods=5)
    data['skewness'] = data['returns_5d'].rolling(window=20).skew()
    data['regime_skewness'] = data['skewness'] * data['volatility_regime']
    
    # Volume surge detection
    data['volume_surge'] = data['volume'] / data['volume'].rolling(window=10).mean()
    
    # Order flow divergence
    data['momentum_diff'] = data['momentum_3d'] - data['momentum_8d']
    data['order_flow_divergence'] = data['order_flow_imbalance'] * data['momentum_diff']
    
    # Microstructure stress (simplified using intraday volatility)
    data['microstructure_stress'] = data['intraday_range'] * data['volume_surge']
    
    # 4. Multi-Scale Microstructure Reversion
    # Multi-scale reversion signals
    data['short_term_reversion'] = -(data['close'] / data['close'].shift(3) - 1)
    data['medium_term_reversion'] = -(data['close'] / data['close'].shift(8) - 1)
    data['long_term_reversion'] = -(data['close'] / data['close'].shift(15) - 1)
    
    # Volume expansion/contraction
    data['volume_expansion'] = data['volume'] / data['volume'].rolling(window=10).mean() - 1
    
    # Combine all signals with regime-adaptive weighting
    data['combined_signal'] = (
        data['reversion_signal'] * 0.3 +
        data['volume_breakout'] * data['movement_efficiency'] * 0.25 +
        data['order_flow_divergence'] * data['regime_skewness'] * 0.2 +
        (data['short_term_reversion'] * 0.4 + 
         data['medium_term_reversion'] * 0.3 + 
         data['long_term_reversion'] * 0.3) * data['volume_expansion'] * 0.25
    )
    
    # Final factor with normalization
    factor = data['combined_signal'].rolling(window=10).apply(
        lambda x: (x.iloc[-1] - x.mean()) / x.std() if x.std() > 0 else 0
    )
    
    return factor
