import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Cross-Market Microstructure Volatility Transmission Alpha
    Combines volatility transmission dynamics, price-volume convergence, and breakout confirmation
    with cross-market microstructure effects.
    """
    data = df.copy()
    
    # Multi-Timeframe Volatility Transmission Components
    # Short-term volatility transmission (3-day)
    data['volatility_1d'] = (data['high'] - data['low']) / data['open']
    data['volatility_3d'] = data['volatility_1d'].rolling(window=3, min_periods=3).mean()
    data['volatility_acceleration'] = (data['volatility_1d'] - data['volatility_3d']) / data['volatility_3d']
    
    # Volatility spillover persistence (consecutive same-direction volatility changes)
    data['volatility_change'] = data['volatility_1d'].diff()
    data['volatility_spillover_persistence'] = data['volatility_change'].rolling(window=3).apply(
        lambda x: np.sum((x > 0) & (x.shift(1) > 0)) if len(x) == 3 else np.nan
    )
    
    # Medium-term transmission networks (8-day)
    data['volatility_range_8d'] = (data['high'].rolling(window=8).max() / 
                                  data['low'].rolling(window=8).min()) - 1
    
    # Volatility cascade detection (clustering patterns)
    data['volatility_clustering'] = data['volatility_1d'].rolling(window=8).std() / data['volatility_1d'].rolling(window=8).mean()
    
    # Volatility regime classification (15-day percentile)
    data['volatility_regime'] = data['volatility_1d'].rolling(window=15).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) == 15 else np.nan
    )
    
    # Price-Volume-Transmission Convergence Components
    # Price efficiency with transmission effects
    data['price_efficiency'] = (data['close'] - data['open']) / (data['high'] - data['low'] + 1e-8)
    data['price_efficiency_transmission'] = data['price_efficiency'] * (1 + data['volatility_spillover_persistence'])
    
    # Gap persistence with transmission effects
    data['open_gap'] = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
    data['gap_persistence'] = data['open_gap'].rolling(window=3).apply(
        lambda x: np.sum((x > 0) & (x.shift(1) > 0)) if len(x) == 3 else np.nan
    )
    
    # Price momentum stability with transmission
    data['returns_5d'] = data['close'].pct_change(periods=5)
    data['price_momentum_stability'] = data['returns_5d'].rolling(window=5).std() * (1 + data['volatility_spillover_persistence'])
    
    # Volume intensity with transmission
    data['volume_ma_5d'] = data['volume'].rolling(window=5).mean()
    data['volume_intensity'] = data['volume'] / (data['volume_ma_5d'] + 1e-8)
    data['volume_intensity_transmission'] = data['volume_intensity'] * (1 + data['volatility_spillover_persistence'])
    
    # Volume persistence with transmission
    data['volume_change'] = data['volume'].pct_change()
    data['volume_persistence'] = data['volume_change'].rolling(window=3).apply(
        lambda x: np.sum((x > 0) & (x.shift(1) > 0)) if len(x) == 3 else np.nan
    )
    
    # Convergence Detection
    # Price-Volume-Transmission Convergence Score
    data['convergence_score'] = (
        data['price_efficiency_transmission'].fillna(0) * 
        data['volume_intensity_transmission'].fillna(0) * 
        (1 + data['volatility_spillover_persistence'].fillna(0))
    )
    
    # Transmission Divergence Patterns
    data['price_transmission_divergence'] = data['price_efficiency'] - data['volatility_spillover_persistence']
    data['volume_transmission_divergence'] = data['volume_intensity'] - data['volatility_spillover_persistence']
    
    # Convergence quality with transmission
    data['convergence_quality'] = data['convergence_score'].rolling(window=3).mean() * data['volatility_spillover_persistence']
    
    # Breakout Confirmation with Transmission
    # Price breakout with transmission
    data['high_5d'] = data['high'].rolling(window=5).max()
    data['price_breakout'] = (data['close'] - data['high_5d'].shift(1)) / data['high_5d'].shift(1)
    data['price_breakout_transmission'] = data['price_breakout'] * (1 + data['volatility_spillover_persistence'])
    
    # Volume breakout with transmission
    data['volume_breakout'] = (data['volume'] - data['volume_ma_5d']) / data['volume_ma_5d']
    data['volume_breakout_transmission'] = data['volume_breakout'] * (1 + data['volatility_spillover_persistence'])
    
    # Combined breakout assessment
    data['combined_breakout'] = (
        data['price_breakout_transmission'].fillna(0) * 
        data['volume_breakout_transmission'].fillna(0) * 
        data['volatility_acceleration'].fillna(0)
    )
    
    # Composite Transmission Alpha Construction
    # Core transmission convergence component
    core_convergence = (
        data['convergence_score'].fillna(0) * 
        data['convergence_quality'].fillna(0) * 
        (1 + data['volatility_acceleration'].fillna(0))
    )
    
    # Transmission divergence enhancement
    divergence_enhancement = (
        data['price_transmission_divergence'].fillna(0) * 
        data['volume_transmission_divergence'].fillna(0) * 
        data['volatility_clustering'].fillna(0)
    )
    
    # Transmission breakout confirmation overlay
    breakout_overlay = data['combined_breakout'].fillna(0) * data['volatility_regime'].fillna(0)
    
    # Final Transmission Alpha Signal
    alpha_signal = (
        core_convergence * 0.4 + 
        divergence_enhancement * 0.3 + 
        breakout_overlay * 0.3
    )
    
    # Normalize the final signal
    alpha_signal_normalized = (alpha_signal - alpha_signal.rolling(window=20).mean()) / (alpha_signal.rolling(window=20).std() + 1e-8)
    
    return alpha_signal_normalized
