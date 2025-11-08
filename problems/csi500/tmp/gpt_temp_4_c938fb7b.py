import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Regime-Adaptive Microstructure Momentum with Volatility-Liquidity Alignment
    """
    data = df.copy()
    
    # Multi-Dimensional Regime Classification
    # Volatility-Liquidity Regime
    data['compression'] = (data['high'] - data['low']) / data['close']
    data['liquidity_efficiency'] = data['amount'] / data['volume']
    
    # Calculate rolling percentiles for regime classification
    data['compression_rank'] = data['compression'].rolling(window=20, min_periods=10).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )
    data['liquidity_rank'] = data['liquidity_efficiency'].rolling(window=20, min_periods=10).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )
    
    # Regime states
    conditions = [
        (data['compression_rank'] < 0.3) & (data['liquidity_rank'] > 0.7),  # Accumulation
        (data['compression_rank'] < 0.3) & (data['liquidity_rank'] <= 0.7), # Indecision
        (data['compression_rank'] >= 0.3) & (data['liquidity_rank'] > 0.7), # Trending
        (data['compression_rank'] >= 0.3) & (data['liquidity_rank'] <= 0.7) # Distribution
    ]
    choices = ['accumulation', 'indecision', 'trending', 'distribution']
    data['vol_liquidity_regime'] = np.select(conditions, choices, default='indecision')
    
    # Momentum Regime
    data['momentum_3d'] = data['close'] / data['close'].shift(3) - 1
    data['momentum_5d'] = data['close'] / data['close'].shift(5) - 1
    
    # Momentum regime classification
    momentum_threshold = 0.005
    momentum_conditions = [
        (data['momentum_3d'] > momentum_threshold) & (data['momentum_5d'] > momentum_threshold) & 
        (data['momentum_5d'] > data['momentum_3d']),  # Strong
        (data['momentum_3d'] > momentum_threshold) & (data['momentum_5d'] > momentum_threshold) & 
        (data['momentum_5d'] <= data['momentum_3d']), # Weak
        (data['momentum_3d'] * data['momentum_5d'] < 0),  # Reversal
        (abs(data['momentum_3d']) <= momentum_threshold) & (abs(data['momentum_5d']) <= momentum_threshold) # Consolidation
    ]
    momentum_choices = ['strong', 'weak', 'reversal', 'consolidation']
    data['momentum_regime'] = np.select(momentum_conditions, momentum_choices, default='weak')
    
    # Microstructure Efficiency Analysis
    # Intraday Price Efficiency
    data['intraday_efficiency'] = (data['close'] - data['open']) / (data['high'] - data['low']).replace(0, np.nan)
    
    # Gap persistence (simplified)
    data['gap_persistence'] = np.where(
        data['open'] > data['close'].shift(1),
        (data['high'] - data['open']) / (data['open'] - data['close'].shift(1)).replace(0, np.nan),
        0
    )
    
    # Breakout quality
    data['high_10d'] = data['high'].rolling(window=10, min_periods=5).max()
    data['low_10d'] = data['low'].rolling(window=10, min_periods=5).min()
    data['breakout_quality'] = (data['close'] - data['high_10d']) / (data['high_10d'] - data['low_10d']).replace(0, np.nan)
    
    # Volume Distribution Patterns (simplified proxies)
    data['volume_concentration'] = data['volume'].rolling(window=5, min_periods=3).max() / data['volume'].rolling(window=5, min_periods=3).sum()
    
    # Closing momentum proxy
    data['closing_momentum'] = (data['close'] - data['open']) / (data['close'] - data['open']).replace(0, np.nan)
    
    # Volume-price alignment
    data['volume_price_alignment'] = np.sign(data['close'] - data['close'].shift(5)) * np.sign(data['volume'] - data['volume'].shift(5))
    
    # Volatility-Microstructure Interaction
    # Asymmetric Volatility Patterns
    up_mask = data['close'] > data['close'].shift(1)
    down_mask = data['close'] < data['close'].shift(1)
    
    data['up_compression'] = np.where(up_mask, data['compression'], np.nan)
    data['down_compression'] = np.where(down_mask, data['compression'], np.nan)
    
    data['volatility_skew'] = (
        data['up_compression'].rolling(window=10, min_periods=5).mean() / 
        data['down_compression'].rolling(window=10, min_periods=5).mean()
    ).replace([np.inf, -np.inf], np.nan)
    
    # Compression-Liquidity Momentum
    data['compression_momentum'] = data['compression'] - data['compression'].shift(1)
    data['liquidity_momentum'] = data['liquidity_efficiency'] - data['liquidity_efficiency'].shift(1)
    data['momentum_alignment'] = np.sign(data['compression_momentum']) * np.sign(data['liquidity_momentum'])
    
    # Regime-Specific Signal Generation
    # Accumulation Regime Signals
    accumulation_signal = data['gap_persistence'] * data['compression']
    accumulation_signal = np.where(
        data['volume_price_alignment'] > 0,
        accumulation_signal * 1.2,
        np.where(data['volume_price_alignment'] < 0, accumulation_signal * 0.8, accumulation_signal)
    )
    
    # Trending Regime Signals
    data['price_acceleration'] = ((data['close'] - data['open']) / data['open']) - (
        (data['close'].shift(1) - data['open'].shift(1)) / data['open'].shift(1)
    )
    data['volume_20d_avg'] = data['volume'].rolling(window=20, min_periods=10).mean()
    trending_signal = data['price_acceleration'] * (data['volume'] / data['volume_20d_avg']) * data['liquidity_efficiency']
    
    # Distribution Regime Signals
    distribution_signal = data['breakout_quality'] * (1 - data['compression'])
    distribution_signal = np.where(
        data['volume_price_alignment'] < 0,
        distribution_signal * 1.2,
        distribution_signal
    )
    distribution_signal = distribution_signal * np.where(data['volatility_skew'] > 1, 1.1, 0.9)
    
    # Indecision Regime Signals
    data['compression_momentum_abs'] = abs(data['compression_momentum'])
    indecision_signal = data['intraday_efficiency'] * data['compression_momentum_abs'] * data['volume_concentration']
    
    # Combine regime signals
    regime_signals = pd.Series(index=data.index, dtype=float)
    for regime in ['accumulation', 'trending', 'distribution', 'indecision']:
        mask = data['vol_liquidity_regime'] == regime
        if regime == 'accumulation':
            regime_signals[mask] = accumulation_signal[mask]
        elif regime == 'trending':
            regime_signals[mask] = trending_signal[mask]
        elif regime == 'distribution':
            regime_signals[mask] = distribution_signal[mask]
        else:  # indecision
            regime_signals[mask] = indecision_signal[mask]
    
    # Adaptive Alpha Factor Construction
    # Base Momentum Component
    data['momentum_strength'] = np.sqrt(data['momentum_3d']**2 + data['momentum_5d']**2)
    
    # Regime score multiplier
    regime_multiplier = pd.Series(index=data.index, dtype=float)
    regime_multiplier[data['momentum_regime'] == 'strong'] = 2.0
    regime_multiplier[data['momentum_regime'] == 'weak'] = 1.0
    regime_multiplier[data['momentum_regime'] == 'reversal'] = -1.0
    regime_multiplier[data['momentum_regime'] == 'consolidation'] = 0.5
    
    base_momentum = regime_signals * data['momentum_strength'] * regime_multiplier
    
    # Microstructure Efficiency Adjustment
    efficiency_multiplier = data['intraday_efficiency'] * 2.0
    
    # Volume distribution adjustment
    volume_adjustment = pd.Series(1.0, index=data.index)
    volume_adjustment[data['volume_concentration'] > 0.3] = 1.3
    volume_adjustment[data['closing_momentum'] > 0.8] = 1.4
    
    efficiency_adjustment = efficiency_multiplier * volume_adjustment
    
    # Volatility-Liquidity Alignment
    alignment_multiplier = pd.Series(1.0, index=data.index)
    alignment_multiplier[data['momentum_alignment'] > 0] = 1.3
    alignment_multiplier[data['momentum_alignment'] < 0] = 0.8
    
    # Asymmetric volatility adjustment
    volatility_adjustment = pd.Series(1.0, index=data.index)
    volatility_adjustment[data['volatility_skew'] > 1.2] = 1.2
    
    # Volatility cluster detection
    data['high_compression'] = data['compression_rank'] < 0.3
    data['compression_cluster'] = data['high_compression'].rolling(window=3).sum()
    volatility_adjustment[data['compression_cluster'] >= 3] = 1.5
    
    alignment_component = alignment_multiplier * volatility_adjustment
    
    # Multi-Timeframe Confirmation
    data['compression_3d'] = data['compression'].rolling(window=3, min_periods=2).mean()
    compression_convergence = data['compression'] * data['compression_3d']
    
    momentum_persistence = data['momentum_3d'] * data['momentum_5d']
    
    confirmation = compression_convergence * momentum_persistence
    
    # Final factor construction
    final_factor = (
        base_momentum * 
        efficiency_adjustment * 
        alignment_component * 
        confirmation
    )
    
    # Clean and return
    final_factor = final_factor.replace([np.inf, -np.inf], np.nan)
    final_factor = final_factor.fillna(0)
    
    return final_factor
