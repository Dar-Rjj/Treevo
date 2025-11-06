import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Generate novel alpha factor using fractal market dynamics with multi-dimensional analysis
    """
    # Create copy to avoid modifying original data
    data = df.copy()
    
    # Multi-Scale Momentum Fracture with Regime Switching
    # Calculate Fractal Momentum Structure
    data['mom_3d'] = data['close'] / data['close'].shift(3) - 1
    data['mom_8d'] = data['close'] / data['close'].shift(8) - 1
    data['mom_13d'] = data['close'] / data['close'].shift(13) - 1
    
    # Detect Momentum Fracture Points
    data['mom_fracture'] = np.where(
        (data['mom_3d'] * data['mom_8d'] < 0) & (abs(data['mom_3d']) > 0.02),
        data['mom_3d'] * -1,  # Signal opposite to short-term momentum
        0
    )
    
    # Regime classification using price range characteristics
    data['daily_range'] = (data['high'] - data['low']) / data['close']
    data['range_ma_5d'] = data['daily_range'].rolling(window=5).mean()
    data['trending_regime'] = (data['daily_range'] > data['range_ma_5d']).astype(int)
    
    # Regime-dependent momentum fracture signal
    data['momentum_signal'] = data['mom_fracture'] * (1 + 0.5 * data['trending_regime'])
    
    # Volume-Price Fractal Dimension with Liquidity Clustering
    # Calculate Fractal Dimension Components
    price_diff = abs(data['close'] - data['open'])
    price_diff = np.where(price_diff == 0, 0.001, price_diff)  # Avoid division by zero
    data['price_fractal'] = (data['high'] - data['low']) / price_diff
    
    data['volume_fractal'] = data['volume'] / data['volume'].shift(1)
    data['volume_fractal'] = data['volume_fractal'].fillna(1)
    
    data['combined_dimension'] = data['price_fractal'] * data['volume_fractal']
    
    # Liquidity Clustering Analysis
    data['volume_ma_5d'] = data['volume'].rolling(window=5).mean()
    data['volume_cluster'] = data['volume'] / data['volume_ma_5d']
    data['liquidity_persistence'] = (data['volume_cluster'] > 1.2).rolling(window=5).sum() / 5
    
    # Fractal Breakout Detection
    data['dimension_ma_3d'] = data['combined_dimension'].rolling(window=3).mean()
    data['fractal_breakout'] = np.where(
        (data['combined_dimension'] > data['dimension_ma_3d'] * 1.5) & 
        (data['volume_cluster'] > 1.2),
        data['combined_dimension'] * data['liquidity_persistence'],
        0
    )
    
    # Temporal Asymmetry in Price Discovery
    # Since we only have daily OHLCV, approximate intraday patterns using daily structure
    daily_range = data['high'] - data['low']
    daily_range = np.where(daily_range == 0, 0.001, daily_range)
    
    # Morning pressure approximation (first half of day)
    data['morning_pressure'] = (data['high'] - data['open']) / daily_range
    
    # Afternoon pressure approximation (second half of day)  
    data['afternoon_pressure'] = (data['close'] - data['low']) / daily_range
    
    data['discovery_asymmetry'] = data['morning_pressure'] - data['afternoon_pressure']
    
    # Multi-Day Persistence Scoring
    data['asymmetry_persistence'] = (data['discovery_asymmetry'].rolling(window=5).apply(
        lambda x: np.sum(x > 0) if len(x) == 5 else 0
    ) / 5)
    
    data['temporal_signal'] = data['discovery_asymmetry'] * data['asymmetry_persistence']
    
    # Price Elasticity with Volume Impulse
    # Calculate Price Elasticity Metrics
    daily_range_elastic = data['high'] - data['low']
    daily_range_elastic = np.where(daily_range_elastic == 0, 0.001, daily_range_elastic)
    data['daily_elasticity'] = abs(data['close'] - data['open']) / daily_range_elastic
    
    data['elasticity_persistence'] = data['daily_elasticity'].rolling(window=3).mean()
    data['elasticity_ma_5d'] = data['daily_elasticity'].rolling(window=5).mean()
    data['elasticity_change'] = data['daily_elasticity'] / data['elasticity_ma_5d'] - 1
    
    # Volume Impulse Integration
    data['volume_ma_3d'] = data['volume'].rolling(window=3).mean()
    data['volume_impulse'] = data['volume'] / data['volume_ma_3d']
    
    # Elastic Regime Classification
    data['elastic_regime'] = (data['daily_elasticity'] > data['elasticity_ma_5d']).astype(int)
    
    data['elasticity_signal'] = data['elasticity_change'] * data['volume_impulse'] * (1 + 0.3 * data['elastic_regime'])
    
    # Multi-Fractal Volatility Clustering with Price Memory
    # Fractal Volatility Structure
    data['vol_3d'] = data['close'].pct_change().rolling(window=3).std()
    data['vol_8d'] = data['close'].pct_change().rolling(window=8).std()
    data['vol_13d'] = data['close'].pct_change().rolling(window=13).std()
    
    data['vol_ratio_short'] = data['vol_3d'] / data['vol_8d']
    data['vol_ratio_long'] = data['vol_8d'] / data['vol_13d']
    
    # Price Memory Effects
    data['high_5d'] = data['high'].rolling(window=5).max()
    data['low_5d'] = data['low'].rolling(window=5).min()
    
    data['price_memory'] = (data['close'] - data['low_5d']) / (data['high_5d'] - data['low_5d'])
    data['price_memory'] = data['price_memory'].fillna(0.5)
    
    # Memory persistence (how consistent is the position in recent range)
    data['memory_persistence'] = abs(data['price_memory'] - 0.5).rolling(window=5).mean()
    
    # Adaptive Signal Generation
    data['volatility_signal'] = (
        data['vol_ratio_short'] * data['price_memory'] * data['memory_persistence'] * 
        np.where(data['price_memory'] > 0.7, -1, np.where(data['price_memory'] < 0.3, 1, 0))
    )
    
    # Cross-Dimensional Market Microstructure Alignment
    # Normalize all component signals
    signals = ['momentum_signal', 'fractal_breakout', 'temporal_signal', 
               'elasticity_signal', 'volatility_signal']
    
    for signal in signals:
        # Robust normalization using median and IQR
        signal_median = data[signal].rolling(window=20, min_periods=10).median()
        signal_iqr = data[signal].rolling(window=20, min_periods=10).apply(
            lambda x: np.percentile(x, 75) - np.percentile(x, 25) if len(x) >= 10 else 1
        )
        data[f'{signal}_norm'] = (data[signal] - signal_median) / (signal_iqr + 1e-8)
    
    # Calculate cross-dimensional consistency
    normalized_signals = [f'{signal}_norm' for signal in signals]
    data['signal_std'] = data[normalized_signals].std(axis=1)
    data['signal_consistency'] = 1 / (1 + data['signal_std'])
    
    # Generate composite alpha with cross-dimensional confirmation
    composite_alpha = (
        data['momentum_signal_norm'] * 0.25 +
        data['fractal_breakout_norm'] * 0.20 +
        data['temporal_signal_norm'] * 0.15 +
        data['elasticity_signal_norm'] * 0.20 +
        data['volatility_signal_norm'] * 0.20
    ) * data['signal_consistency']
    
    # Final robust normalization
    alpha_median = composite_alpha.rolling(window=20, min_periods=10).median()
    alpha_iqr = composite_alpha.rolling(window=20, min_periods=10).apply(
        lambda x: np.percentile(x, 75) - np.percentile(x, 25) if len(x) >= 10 else 1
    )
    final_alpha = (composite_alpha - alpha_median) / (alpha_iqr + 1e-8)
    
    return final_alpha
