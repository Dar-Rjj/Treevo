import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Multi-Scale Volatility-Volume Momentum Efficiency Factor
    Combines momentum alignment, volume-volatility regimes, gap efficiency, and volume-price dynamics
    """
    # Copy dataframe to avoid modifying original
    data = df.copy()
    
    # Calculate True Range
    data['TR'] = np.maximum(
        data['high'] - data['low'],
        np.maximum(
            abs(data['high'] - data['close'].shift(1)),
            abs(data['low'] - data['close'].shift(1))
        )
    )
    
    # Multi-Timeframe Momentum-Volatility Alignment
    # Short-term momentum (2-day)
    data['momentum_short'] = data['close'] - data['close'].shift(2)
    # Medium-term momentum (5-day)
    data['momentum_medium'] = data['close'] - data['close'].shift(5)
    
    # Momentum-Volatility Ratios
    data['momentum_vol_ratio_short'] = data['momentum_short'] / data['TR'].rolling(window=3, min_periods=1).mean()
    data['momentum_vol_ratio_medium'] = data['momentum_medium'] / data['TR'].rolling(window=6, min_periods=1).mean()
    
    # Momentum Alignment Score (sign consistency)
    data['momentum_alignment'] = np.where(
        (data['momentum_short'] * data['momentum_medium']) > 0,
        np.sign(data['momentum_short']) * (abs(data['momentum_vol_ratio_short']) + abs(data['momentum_vol_ratio_medium'])) / 2,
        0
    )
    
    # Volume-Volatility Regime Classification
    # Volume percentile (15-day lookback)
    data['volume_percentile'] = data['volume'].rolling(window=15, min_periods=1).apply(
        lambda x: (x[-1] > np.percentile(x[:-1], 70)) if len(x) > 1 else 0
    )
    
    # Volatility regime (True Range vs 10-day average)
    data['volatility_regime'] = np.where(
        data['TR'] > data['TR'].rolling(window=10, min_periods=1).mean(),
        1,  # High volatility
        0   # Low volatility
    )
    
    # Volume-Volatility Convergence
    data['vol_vol_convergence'] = data['volume_percentile'] * data['volatility_regime']
    
    # Gap-Momentum Efficiency Component
    # Overnight gap
    data['overnight_gap'] = data['open'] - data['close'].shift(1)
    
    # Gap fill efficiency (gap closure speed)
    data['gap_fill_efficiency'] = np.where(
        data['overnight_gap'] != 0,
        (data['close'] - data['open']) / data['overnight_gap'],
        0
    )
    
    # Gap momentum persistence (post-gap price movement)
    data['gap_momentum_persistence'] = np.where(
        data['overnight_gap'] != 0,
        (data['close'] - data['open']) * np.sign(data['overnight_gap']),
        0
    )
    
    # Gap efficiency score
    data['gap_efficiency'] = np.where(
        abs(data['overnight_gap']) > data['TR'].rolling(window=5, min_periods=1).mean() * 0.1,
        (1 - abs(data['gap_fill_efficiency'])) * np.sign(data['gap_momentum_persistence']),
        0
    )
    
    # Volume-Price Efficiency Dynamics
    # Price movement efficiency
    data['price_efficiency'] = abs(data['close'] - data['close'].shift(1)) / data['TR']
    
    # Volume-weighted momentum
    data['volume_weighted_momentum'] = data['momentum_short'] * data['volume'] / data['volume'].rolling(window=10, min_periods=1).mean()
    
    # Order flow imbalance (simplified using price movement and volume)
    data['order_flow_imbalance'] = np.where(
        data['close'] > data['close'].shift(1),
        data['volume'],
        -data['volume']
    ) / data['volume'].rolling(window=10, min_periods=1).mean()
    
    # Liquidity absorption (large move efficiency)
    data['liquidity_absorption'] = np.where(
        data['TR'] > data['TR'].rolling(window=10, min_periods=1).mean(),
        data['price_efficiency'] * data['volume'] / data['volume'].rolling(window=10, min_periods=1).mean(),
        0
    )
    
    # Regime-Adaptive Signal Synthesis
    # Weight momentum by volatility regime
    volatility_adjustment = np.where(
        data['volatility_regime'] == 1,  # High volatility
        data['momentum_alignment'] * 0.7,  # Reduce signal in high volatility
        data['momentum_alignment'] * 1.3   # Amplify in low volatility
    )
    
    # Adjust by volume confirmation
    volume_confirmation = np.where(
        data['volume_percentile'] == 1,
        volatility_adjustment * 1.2,  # Reinforce with high volume
        volatility_adjustment * 0.8   # Reduce without volume confirmation
    )
    
    # Apply gap efficiency filter
    gap_filtered = np.where(
        data['gap_efficiency'] > 0,
        volume_confirmation * 1.1,  # Maintain sustainable gaps
        np.where(
            data['gap_efficiency'] < 0,
            volume_confirmation * 0.9,  # Reduce filling gaps
            volume_confirmation
        )
    )
    
    # Final factor synthesis
    final_factor = (
        gap_filtered * 
        (1 + data['liquidity_absorption'] * 0.1) *  # Incorporate liquidity dynamics
        (1 + data['order_flow_imbalance'] * 0.05)   # Incorporate order flow
    )
    
    # Normalize and return
    factor_series = pd.Series(final_factor, index=data.index)
    factor_series = (factor_series - factor_series.rolling(window=20, min_periods=1).mean()) / factor_series.rolling(window=20, min_periods=1).std()
    
    return factor_series.fillna(0)
