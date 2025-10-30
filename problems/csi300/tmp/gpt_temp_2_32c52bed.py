import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Multi-Timeframe Volatility-Adjusted Momentum with Liquidity Anomaly Detection
    """
    data = df.copy()
    
    # Calculate Volatility-Adjusted Momentum
    # Short-term volatility (5-day)
    data['high_5d_max'] = data['high'].rolling(window=5, min_periods=5).max()
    data['low_5d_min'] = data['low'].rolling(window=5, min_periods=5).min()
    data['short_term_vol'] = (data['high_5d_max'] - data['low_5d_min']) / data['close'].shift(5)
    
    # Momentum acceleration
    data['momentum_3d'] = data['close'] / data['close'].shift(3)
    data['momentum_5d'] = data['close'] / data['close'].shift(5)
    data['momentum_acceleration'] = data['momentum_3d'] / data['momentum_5d']
    
    # Volatility-adjusted momentum signal
    data['vol_adj_momentum'] = data['momentum_acceleration'] / (data['short_term_vol'] + 1e-8)
    
    # Liquidity Anomaly Detection
    # Volume-price divergence
    data['price_range_efficiency'] = np.abs(data['close'] - data['open']) / (data['high'] - data['low'] + 1e-8)
    data['volume_efficiency'] = data['volume'] / (data['volume'].shift(5) + 1e-8)
    data['volume_price_divergence'] = data['price_range_efficiency'] / (data['volume_efficiency'] + 1e-8)
    
    # Liquidity regime shifts
    data['amount_per_trade'] = data['amount'] / (data['volume'] + 1e-8)
    data['trade_size_ratio'] = data['amount_per_trade'] / (data['close'] + 1e-8)
    data['liquidity_signal'] = data['volume_price_divergence'] * data['trade_size_ratio']
    
    # Intraday Pattern Recognition
    # Opening gap persistence
    data['opening_gap'] = data['open'] / data['close'].shift(1)
    data['gap_persistence'] = np.abs(data['close'] - data['open']) / (np.abs(data['open'] - data['close'].shift(1)) + 1e-8)
    
    # Midday reversal patterns
    data['high_proximity'] = (data['high'] - data['close']) / (data['high'] - data['low'] + 1e-8)
    data['low_proximity'] = (data['close'] - data['low']) / (data['high'] - data['low'] + 1e-8)
    data['reversal_pattern'] = np.where(
        ((data['high_proximity'] > 0.8) | (data['low_proximity'] > 0.8)), 
        -1, 1
    )
    
    # Pattern signal
    data['intraday_pattern'] = data['gap_persistence'] * data['reversal_pattern'] * data['price_range_efficiency']
    
    # Multi-Timeframe Signal Integration
    # Volatility-momentum regime classification
    vol_momentum_quantile = data['vol_adj_momentum'].rolling(window=20, min_periods=20).apply(
        lambda x: pd.qcut(x, 3, labels=False, duplicates='drop').iloc[-1] if len(x) == 20 else np.nan, 
        raw=False
    )
    
    # Liquidity regime classification
    liquidity_quantile = data['liquidity_signal'].rolling(window=20, min_periods=20).apply(
        lambda x: pd.qcut(x, 3, labels=False, duplicates='drop').iloc[-1] if len(x) == 20 else np.nan, 
        raw=False
    )
    
    # Timeframe consistency scoring
    data['momentum_consistency'] = data['vol_adj_momentum'].rolling(window=5, min_periods=5).std()
    data['liquidity_consistency'] = data['liquidity_signal'].rolling(window=5, min_periods=5).std()
    data['pattern_consistency'] = data['intraday_pattern'].rolling(window=5, min_periods=5).std()
    
    # Composite alpha factor
    data['composite_alpha'] = (
        data['vol_adj_momentum'] / (data['momentum_consistency'] + 1e-8) * 0.4 +
        data['liquidity_signal'] / (data['liquidity_consistency'] + 1e-8) * 0.35 +
        data['intraday_pattern'] / (data['pattern_consistency'] + 1e-8) * 0.25
    )
    
    # Risk-Adjusted Alpha Generation
    # Signal quality assessment
    recent_volatility = data['close'].pct_change().rolling(window=10, min_periods=10).std()
    data['signal_to_noise'] = data['composite_alpha'].abs() / (recent_volatility + 1e-8)
    
    # Position sizing framework
    momentum_strength = data['vol_adj_momentum'].rolling(window=10, min_periods=10).apply(
        lambda x: np.percentile(x, 75) - np.percentile(x, 25), raw=True
    )
    
    # Final predictive factor
    data['alpha_factor'] = (
        data['composite_alpha'] * 
        np.tanh(data['signal_to_noise']) * 
        (1 + np.tanh(momentum_strength))
    )
    
    return data['alpha_factor']
