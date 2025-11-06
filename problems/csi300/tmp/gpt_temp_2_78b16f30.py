import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Multi-Scale Fractal Entropy Momentum with Volume-Volatility Regime Integration
    """
    # Copy dataframe to avoid modifying original
    data = df.copy()
    
    # Multi-Timeframe Fractal Efficiency Calculation
    # Short-term (5-day) fractal efficiency
    data['cumulative_abs_5d'] = data['close'].diff().abs().rolling(window=5, min_periods=3).sum()
    data['net_movement_5d'] = data['close'].diff(5)
    data['fractal_eff_5d'] = data['net_movement_5d'] / (data['cumulative_abs_5d'] + 1e-8)
    
    # Medium-term (10-day) fractal efficiency
    data['cumulative_abs_10d'] = data['close'].diff().abs().rolling(window=10, min_periods=5).sum()
    data['net_movement_10d'] = data['close'].diff(10)
    data['fractal_eff_10d'] = data['net_movement_10d'] / (data['cumulative_abs_10d'] + 1e-8)
    
    # Long-term (15-day) fractal efficiency
    data['cumulative_abs_15d'] = data['close'].diff().abs().rolling(window=15, min_periods=8).sum()
    data['net_movement_15d'] = data['close'].diff(15)
    data['fractal_eff_15d'] = data['net_movement_15d'] / (data['cumulative_abs_15d'] + 1e-8)
    
    # Microstructure entropy measurement
    # Order flow imbalance entropy (simplified using volume and price movement)
    data['price_direction'] = np.where(data['close'] > data['open'], 1, -1)
    data['buy_volume'] = np.where(data['price_direction'] == 1, data['volume'], 0)
    data['sell_volume'] = np.where(data['price_direction'] == -1, data['volume'], 0)
    
    # Calculate rolling entropy of volume imbalance (5-day window)
    def calculate_volume_entropy(series):
        if len(series) < 2:
            return np.nan
        p_buy = np.sum(series > 0) / len(series)
        p_sell = 1 - p_buy
        if p_buy == 0 or p_sell == 0:
            return 0
        return - (p_buy * np.log2(p_buy) + p_sell * np.log2(p_sell))
    
    data['volume_entropy'] = data['buy_volume'].rolling(window=5, min_periods=3).apply(
        calculate_volume_entropy, raw=False
    )
    
    # Entropy-weighted fractal momentum
    data['entropy_weight'] = 1 / (1 + data['volume_entropy'].fillna(0))
    data['weighted_fractal_5d'] = data['fractal_eff_5d'] * data['entropy_weight']
    data['weighted_fractal_10d'] = data['fractal_eff_10d'] * data['entropy_weight']
    data['weighted_fractal_15d'] = data['fractal_eff_15d'] * data['entropy_weight']
    
    # Multi-scale momentum divergence
    data['momentum_divergence'] = (
        data['weighted_fractal_5d'] - data['weighted_fractal_10d']
    ) + (data['weighted_fractal_10d'] - data['weighted_fractal_15d'])
    
    # Volume-Volatility Regime Integration
    # Volume concentration analysis
    data['volume_skewness'] = data['volume'].rolling(window=20, min_periods=10).skew()
    data['volume_trend'] = data['volume'].rolling(window=5, min_periods=3).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0], raw=False
    )
    
    # Volatility regime characterization
    data['daily_range'] = data['high'] - data['low']
    data['avg_range_10d'] = data['daily_range'].rolling(window=10, min_periods=5).mean()
    data['volatility_regime'] = data['daily_range'] / (data['avg_range_10d'] + 1e-8)
    
    # Volume-volatility asymmetric weighting
    data['volume_regime'] = np.where(
        (data['volume'] > data['volume'].rolling(window=20, min_periods=10).mean()) & 
        (data['close'] > data['open']), 'high_volume_up',
        np.where(
            (data['volume'] > data['volume'].rolling(window=20, min_periods=10).mean()) & 
            (data['close'] <= data['open']), 'high_volume_down',
            np.where(
                data['close'] > data['open'], 'low_volume_up', 'low_volume_down'
            )
        )
    )
    
    # Volatility-based signal adjustment
    data['volatility_weight'] = np.where(
        data['volatility_regime'] < 0.8, 1.2,  # Enhanced weight for low volatility
        np.where(data['volatility_regime'] > 1.2, 0.8, 1.0)  # Reduced weight for high volatility
    )
    
    # Volume concentration validation
    data['volume_concentration_weight'] = np.where(
        abs(data['volume_skewness']) > 1.0, 1.3,  # Strong weight for concentrated patterns
        np.where(abs(data['volume_skewness']) > 0.5, 1.1, 1.0)  # Moderate weight
    )
    
    # Price-Volume Fractal Alignment
    data['volume_fractal_alignment'] = (
        data['weighted_fractal_5d'] * np.sign(data['volume_trend'])
    )
    
    # Entropy-enhanced volume-momentum validation
    data['entropy_confirmation'] = np.where(
        data['volume_entropy'] < 0.3, 1.5,  # Strong confirmation for low entropy
        np.where(data['volume_entropy'] > 0.7, 0.7, 1.0)  # Reduced confirmation for high entropy
    )
    
    # Regime transition timing signals
    data['entropy_volatility_convergence'] = (
        data['volume_entropy'].rolling(window=5, min_periods=3).std() / 
        data['volatility_regime'].rolling(window=5, min_periods=3).std()
    )
    
    # Composite Factor Generation
    data['composite_factor'] = (
        data['momentum_divergence'] * 
        data['volatility_weight'] * 
        data['volume_concentration_weight'] * 
        data['entropy_confirmation'] * 
        data['volume_fractal_alignment'] * 
        (1 / (1 + abs(data['entropy_volatility_convergence'])))
    )
    
    # Final factor normalization
    factor = data['composite_factor'].fillna(0)
    
    return factor
