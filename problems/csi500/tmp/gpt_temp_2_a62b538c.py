import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Regime-Adaptive Momentum-Efficiency Divergence Alpha Factor
    """
    data = df.copy()
    
    # Multi-Timeframe Momentum Analysis
    # Short-term momentum (5-day)
    data['momentum_5d'] = data['close'] / data['close'].shift(5) - 1
    
    # 3-day directional consistency count
    data['daily_ret'] = data['close'].pct_change()
    data['pos_ret_count'] = data['daily_ret'].rolling(window=3, min_periods=1).apply(
        lambda x: np.sum(x > 0), raw=True
    )
    data['neg_ret_count'] = data['daily_ret'].rolling(window=3, min_periods=1).apply(
        lambda x: np.sum(x < 0), raw=True
    )
    data['direction_consistency'] = np.where(
        data['daily_ret'] > 0, 
        data['pos_ret_count'] / 3, 
        data['neg_ret_count'] / 3
    )
    
    # Medium-term momentum (10-day)
    data['momentum_10d'] = data['close'] / data['close'].shift(10) - 1
    data['momentum_acceleration'] = data['momentum_5d'] - data['momentum_10d']
    
    # Momentum persistence
    data['momentum_sign'] = np.sign(data['daily_ret'])
    data['momentum_persistence'] = 0
    for i in range(1, len(data)):
        if data['momentum_sign'].iloc[i] == data['momentum_sign'].iloc[i-1]:
            data.loc[data.index[i], 'momentum_persistence'] = data['momentum_persistence'].iloc[i-1] + 1
        else:
            data.loc[data.index[i], 'momentum_persistence'] = 1
    
    # Volatility-Liquidity Regime Framework
    # Volatility regime component
    data['true_range'] = np.maximum(
        data['high'] - data['low'],
        np.maximum(
            abs(data['high'] - data['close'].shift(1)),
            abs(data['low'] - data['close'].shift(1))
        )
    )
    data['atr_5d'] = data['true_range'].rolling(window=5, min_periods=1).mean()
    data['volatility_momentum'] = data['atr_5d'] / data['atr_5d'].shift(3) - 1
    
    # Liquidity regime component
    data['liquidity'] = data['volume'] * abs(data['close'] - data['open'])
    data['avg_liquidity_5d'] = data['liquidity'].rolling(window=5, min_periods=1).mean()
    data['liquidity_momentum'] = data['avg_liquidity_5d'] / data['avg_liquidity_5d'].shift(3) - 1
    
    # Regime classification
    data['volatility_regime'] = np.where(
        data['atr_5d'] > data['atr_5d'].rolling(window=20, min_periods=1).median(), 
        1, -1  # 1: high volatility, -1: low volatility
    )
    data['liquidity_regime'] = np.where(
        data['avg_liquidity_5d'] > data['avg_liquidity_5d'].rolling(window=20, min_periods=1).median(),
        1, -1  # 1: high liquidity, -1: low liquidity
    )
    
    # Regime interaction analysis
    data['regime_alignment'] = data['volatility_regime'] * data['liquidity_regime']
    data['regime_interaction_strength'] = data['volatility_momentum'] * data['liquidity_momentum']
    
    # Efficiency-Divergence Core Engine
    # Price efficiency measurement
    data['efficiency_ratio'] = abs(data['close'] - data['open']) / data['true_range']
    data['efficiency_ratio'] = data['efficiency_ratio'].replace([np.inf, -np.inf], 0).fillna(0)
    
    # Efficiency momentum dynamics
    data['efficiency_momentum'] = data['efficiency_ratio'] - data['efficiency_ratio'].shift(5)
    data['efficiency_trend'] = np.sign(data['efficiency_ratio'] - data['efficiency_ratio'].shift(3))
    
    # Divergence detection system
    data['momentum_efficiency_divergence'] = 0
    bullish_condition = (data['momentum_5d'] < 0) & (data['efficiency_momentum'] > 0)
    bearish_condition = (data['momentum_5d'] > 0) & (data['efficiency_momentum'] < 0)
    
    data.loc[bullish_condition, 'momentum_efficiency_divergence'] = (
        abs(data['momentum_5d']) * data['efficiency_momentum']
    )
    data.loc[bearish_condition, 'momentum_efficiency_divergence'] = (
        -abs(data['momentum_5d']) * data['efficiency_momentum']
    )
    
    # Volume confirmation integration
    data['volume_momentum'] = data['volume'] / data['volume'].shift(3) - 1
    data['volume_breakout_alignment'] = np.sign(data['volume_momentum']) * np.sign(data['momentum_5d'])
    
    # Volume persistence
    data['volume_sign'] = np.sign(data['volume'] - data['volume'].shift(1))
    data['volume_persistence'] = 0
    for i in range(1, len(data)):
        if data['volume_sign'].iloc[i] == data['volume_sign'].iloc[i-1]:
            data.loc[data.index[i], 'volume_persistence'] = data['volume_persistence'].iloc[i-1] + 1
        else:
            data.loc[data.index[i], 'volume_persistence'] = 1
    
    data['volume_momentum_consistency'] = data['volume_breakout_alignment'] * (data['volume_persistence'] / 5)
    
    # Divergence quality assessment
    data['divergence_quality'] = (
        data['momentum_efficiency_divergence'] * 
        data['volume_momentum_consistency'] * 
        data['direction_consistency']
    )
    
    # Regime-Enhanced Signal Construction
    # Volatility regime scaling
    data['volatility_scaling'] = np.where(
        data['volatility_regime'] == 1,
        1.5,  # Enhanced weight during high volatility
        0.8   # Reduced weight during low volatility
    )
    
    # Liquidity confirmation layer
    data['volume_acceleration'] = (
        data['volume_momentum'] - data['volume_momentum'].shift(5)
    )
    data['gap_analysis'] = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
    data['gap_persistence'] = data['gap_analysis'].rolling(window=3, min_periods=1).apply(
        lambda x: np.sum(np.sign(x) == np.sign(x[-1])) / len(x), raw=True
    )
    
    # Efficiency filter application
    data['efficiency_filter'] = np.where(
        data['efficiency_ratio'] > data['efficiency_ratio'].rolling(window=10, min_periods=1).median(),
        1.2,  # Boost signals during high efficiency
        0.7   # Reduce noise during low efficiency
    )
    
    # Composite Alpha Generation
    # Core divergence component
    core_divergence = (
        data['momentum_efficiency_divergence'] * 
        data['divergence_quality'] * 
        data['volume_momentum_consistency']
    )
    
    # Regime adaptation layer
    regime_adaptation = (
        data['volatility_scaling'] * 
        data['liquidity_momentum'] * 
        data['regime_interaction_strength']
    )
    
    # Breakout enhancement
    breakout_enhancement = (
        data['volume_acceleration'] * 
        data['momentum_acceleration'] * 
        data['efficiency_trend']
    )
    
    # Final alpha factor
    alpha_factor = (
        core_divergence * 
        regime_adaptation * 
        breakout_enhancement * 
        data['efficiency_filter']
    )
    
    # Clean and normalize
    alpha_factor = alpha_factor.replace([np.inf, -np.inf], 0).fillna(0)
    
    return alpha_factor
