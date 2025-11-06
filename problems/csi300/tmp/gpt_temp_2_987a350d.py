import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Quantum Range Efficiency with Liquidity Acceleration factor
    """
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Quantum Range State Analysis
    # Range Efficiency as Quantum Coherence
    data['range_efficiency'] = (data['close'] - data['open']) / (data['high'] - data['low'])
    data['range_efficiency'] = data['range_efficiency'].replace([np.inf, -np.inf], np.nan)
    
    # Efficiency Persistence: 5-Day Efficiency Trend Slope
    data['efficiency_trend'] = data['range_efficiency'].rolling(window=5).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 5 else np.nan
    )
    
    # Range Expansion Quality
    data['range_expansion'] = (data['high'] - data['low']) / (data['high'] - data['low']).shift(1)
    data['range_expansion'] = data['range_expansion'].replace([np.inf, -np.inf], np.nan)
    
    # Expansion Consistency: Consecutive Range Expansion Days
    data['range_expansion_flag'] = (data['range_expansion'] > 1).astype(int)
    data['expansion_consecutive'] = data['range_expansion_flag'] * (data['range_expansion_flag'].groupby(
        (data['range_expansion_flag'] != data['range_expansion_flag'].shift()).cumsum()
    ).cumcount() + 1)
    
    # Liquidity Acceleration Field
    # Volume Acceleration Dynamics
    data['volume_acceleration'] = (data['volume'] / data['volume'].shift(1)) - 1
    data['volume_acceleration'] = data['volume_acceleration'].replace([np.inf, -np.inf], np.nan)
    
    # Volume Spike Persistence: 3-Day Volume Trend Slope
    data['volume_trend'] = data['volume'].rolling(window=3).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 3 else np.nan
    )
    
    # Amount Field Intensity
    # Large Trade Ratio (simplified as top quantile of daily amount distribution)
    data['amount_rank'] = data['amount'].rolling(window=20).rank(pct=True)
    data['large_trade_ratio'] = data['amount_rank'].rolling(window=5).mean()
    
    # Amount Momentum: Daily Change in Large Trade Ratio
    data['amount_momentum'] = data['large_trade_ratio'].diff()
    
    # Quantum State Transition Patterns
    # High Coherence Breakouts
    data['high_coherence'] = (data['range_efficiency'] > data['range_efficiency'].rolling(window=10).mean()).astype(int)
    data['range_expanding'] = (data['range_expansion'] > 1).astype(int)
    data['volume_accelerating'] = (data['volume_acceleration'] > 0).astype(int)
    
    data['breakout_signal'] = data['high_coherence'] * data['range_expanding'] * data['volume_accelerating']
    
    # Low Coherence Reversals
    data['low_coherence'] = (data['range_efficiency'] < data['range_efficiency'].rolling(window=10).mean()).astype(int)
    data['range_compressing'] = (data['range_expansion'] < 1).astype(int)
    data['volume_decelerating'] = (data['volume_acceleration'] < 0).astype(int)
    
    data['reversal_signal'] = data['low_coherence'] * data['range_compressing'] * data['volume_decelerating']
    
    # Field Interaction Dynamics
    # Range-Liquidity Alignment
    data['positive_alignment'] = data['range_expanding'] * (data['volume_acceleration'] > 0).astype(int)
    data['negative_alignment'] = data['range_compressing'] * (data['volume_acceleration'] < 0).astype(int)
    
    # Amount Field Coupling
    data['high_large_trade'] = (data['large_trade_ratio'] > data['large_trade_ratio'].rolling(window=10).mean()).astype(int)
    data['low_large_trade'] = (data['large_trade_ratio'] < data['large_trade_ratio'].rolling(window=10).mean()).astype(int)
    
    data['strong_coupling'] = data['high_large_trade'] * data['range_expanding']
    data['weak_coupling'] = data['low_large_trade'] * data['range_compressing']
    
    # Alpha Signal Generation
    # Quantum Efficiency Momentum
    data['efficiency_momentum'] = data['range_efficiency'] * data['volume_acceleration']
    
    # Persistence-Weighted Expansion Quality
    data['weighted_expansion'] = data['expansion_consecutive'] * data['range_expansion']
    
    # Liquidity-Enhanced State Transitions
    data['amount_weighted_coherence'] = data['large_trade_ratio'] * data['range_efficiency']
    data['acceleration_adjusted_breakout'] = data['breakout_signal'] * (1 + data['volume_acceleration'])
    
    # Final Alpha Factor Construction
    # Combine components with appropriate weights
    alpha = (
        0.3 * data['efficiency_momentum'] +
        0.25 * data['weighted_expansion'] +
        0.2 * data['amount_weighted_coherence'] +
        0.15 * data['acceleration_adjusted_breakout'] +
        0.1 * data['strong_coupling'] -
        0.1 * data['weak_coupling'] -
        0.05 * data['reversal_signal']
    )
    
    # Normalize the final factor
    alpha_normalized = (alpha - alpha.rolling(window=20).mean()) / alpha.rolling(window=20).std()
    
    return alpha_normalized
