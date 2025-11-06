import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Multi-Scale Efficiency-Liquidity Momentum Divergence Factor
    """
    data = df.copy()
    
    # Basic price calculations
    data['prev_close'] = data['close'].shift(1)
    data['true_range'] = np.maximum(
        data['high'] - data['low'],
        np.maximum(
            abs(data['high'] - data['prev_close']),
            abs(data['low'] - data['prev_close'])
        )
    )
    
    # Intraday Efficiency-Liquidity Assessment
    data['range_efficiency'] = (data['high'] - data['low']) / data['close']
    data['gap_efficiency'] = abs(data['open'] - data['prev_close']) / data['prev_close']
    data['close_position_strength'] = (data['close'] - data['low']) / (data['high'] - data['low']).replace(0, np.nan)
    data['vwap_efficiency'] = ((data['close'] - data['open']) * data['volume']) / (data['high'] - data['low']).replace(0, np.nan)
    data['liquidity_absorption'] = data['volume'] / abs(data['close'] - data['prev_close']).replace(0, np.nan)
    
    # Multi-Timeframe Momentum Divergence
    # Ultra-Short Term
    data['opening_gap_momentum'] = abs(data['open'] - data['prev_close']) / (data['high'] - data['low']).replace(0, np.nan)
    mid_price = (data['high'] + data['low']) / 2
    data['closing_momentum_persistence'] = (data['close'] - mid_price) / (data['high'] - data['low']).replace(0, np.nan)
    
    # Short-Term (3-5 days)
    data['price_volume_momentum'] = ((data['close'] - data['prev_close']) / data['prev_close'] + 
                                   (data['close'] - data['close'].shift(5)) / data['close'].shift(5))
    data['volume_momentum'] = data['volume'] / data['volume'].shift(5)
    data['volatility_adj_momentum'] = (data['close'] - data['close'].shift(3)) / data['true_range'].rolling(window=3, min_periods=1).mean()
    
    # Medium-Term (10-20 days)
    data['compression_ratio'] = (
        (data['high'].rolling(window=7, min_periods=1).max() - data['low'].rolling(window=7, min_periods=1).min()) /
        (data['high'].rolling(window=20, min_periods=1).max() - data['low'].rolling(window=20, min_periods=1).min()).replace(0, np.nan)
    )
    
    returns_15d = data['close'].pct_change(periods=1)
    data['trend_quality'] = returns_15d.rolling(window=15, min_periods=1).apply(lambda x: (x > 0).sum() / len(x))
    
    data['drawdown_resilience'] = (
        (data['close'] - data['low'].rolling(window=15, min_periods=1).min()) /
        (data['high'].rolling(window=15, min_periods=1).max() - data['low'].rolling(window=15, min_periods=1).min()).replace(0, np.nan)
    )
    
    # Volatility-Regime Adaptive Framework
    data['volatility_regime'] = (
        data['true_range'] / 
        data['true_range'].rolling(window=5, min_periods=1).mean()
    )
    data['pressure_asymmetry'] = (data['high'] - data['close'] - data['close'] + data['low']) / data['true_range'].replace(0, np.nan)
    
    # Liquidity Regime Detection
    data['volume_acceleration'] = data['volume'] / data['volume'].shift(3)
    data['market_depth'] = data['volume'] / data['true_range'].replace(0, np.nan)
    
    # Divergence-Convergence Detection
    price_momentum_5d = data['close'].pct_change(periods=5)
    volume_momentum_5d = data['volume'] / data['volume'].shift(5)
    data['efficiency_liquidity_divergence'] = data['range_efficiency'] * (price_momentum_5d - volume_momentum_5d)
    
    # Timeframe Momentum Divergence
    data['ultra_short_vs_short'] = np.sign(data['opening_gap_momentum']) * np.sign(data['price_volume_momentum'])
    data['short_vs_medium'] = np.sign(data['price_volume_momentum']) * np.sign(data['trend_quality'])
    
    convergence_signals = pd.concat([
        np.sign(data['opening_gap_momentum']),
        np.sign(data['price_volume_momentum']),
        np.sign(data['trend_quality'])
    ], axis=1)
    data['convergence_score'] = convergence_signals.apply(lambda x: (x == x.iloc[0]).sum() if not x.isna().any() else 0, axis=1)
    
    # Volume-Price Alignment
    data['volume_clustering_efficiency'] = (data['volume'] / data['volume'].shift(1)) * np.sign(data['close'] - data['open'])
    
    # Adaptive Signal Integration
    # Volatility regime adjustments
    high_vol_mask = data['volatility_regime'] > 1.2
    low_vol_mask = data['volatility_regime'] < 0.8
    
    # Base signal components
    data['alignment_score'] = (
        data['close_position_strength'] * data['volume_clustering_efficiency'] * 
        data['convergence_score'] / 3.0
    )
    
    data['liquidity_weighted_momentum'] = (
        data['price_volume_momentum'] * data['liquidity_absorption'] / 
        data['liquidity_absorption'].rolling(window=20, min_periods=1).mean()
    )
    
    # Base signal
    data['base_signal'] = data['alignment_score'] * data['liquidity_weighted_momentum']
    
    # Divergence enhancement
    efficiency_momentum_mismatch = abs(data['range_efficiency'] - data['price_volume_momentum'])
    data['divergence_enhanced'] = data['base_signal'] * (1 + efficiency_momentum_mismatch)
    
    # Timeframe convergence
    data['timeframe_converged'] = data['divergence_enhanced'] * data['convergence_score']
    
    # Quality filters
    data['quality_filtered'] = (
        data['timeframe_converged'] * 
        data['trend_quality'] * 
        data['drawdown_resilience'] * 
        (1 - (data['high'] - data['low']) / data['close'])
    )
    
    # Volatility regime adjustments
    data['volatility_adjusted'] = data['quality_filtered'].copy()
    data.loc[high_vol_mask, 'volatility_adjusted'] = data.loc[high_vol_mask, 'quality_filtered'] * 1.4
    data.loc[low_vol_mask, 'volatility_adjusted'] = data.loc[low_vol_mask, 'quality_filtered'] * 0.8
    
    # Final alpha factor
    alpha_factor = (
        data['efficiency_liquidity_divergence'] * 0.3 +
        data['volatility_adjusted'] * 0.5 +
        data['compression_ratio'] * data['volume_acceleration'] * 0.2
    )
    
    # Normalize and clean
    alpha_factor = alpha_factor.replace([np.inf, -np.inf], np.nan)
    alpha_factor = (alpha_factor - alpha_factor.rolling(window=20, min_periods=1).mean()) / alpha_factor.rolling(window=20, min_periods=1).std()
    
    return alpha_factor
