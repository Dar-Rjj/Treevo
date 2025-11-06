import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Novel alpha factor combining momentum regimes, microstructure reversal, 
    behavioral patterns, and information flow analysis.
    """
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate returns
    data['returns'] = data['close'].pct_change()
    
    # 1. Momentum-Regime Adaptive Factor
    # Multi-timeframe momentum calculation
    data['momentum_3d'] = data['close'] / data['close'].shift(3) - 1
    data['momentum_8d'] = data['close'] / data['close'].shift(8) - 1
    data['momentum_15d'] = data['close'] / data['close'].shift(15) - 1
    
    # Volatility regime detection
    data['volatility_10d'] = data['returns'].rolling(window=10, min_periods=5).std()
    volatility_threshold = data['volatility_10d'].rolling(window=30, min_periods=15).median()
    data['high_vol_regime'] = (data['volatility_10d'] > volatility_threshold).astype(int)
    
    # Regime-adaptive combination
    data['momentum_factor'] = np.where(
        data['high_vol_regime'] == 1,
        data['momentum_3d'] * 0.6 + data['momentum_8d'] * 0.3 + data['momentum_15d'] * 0.1,
        data['momentum_3d'] * 0.2 + data['momentum_8d'] * 0.3 + data['momentum_15d'] * 0.5
    )
    
    # 2. Microstructure Reversal Factor
    # Price efficiency measurement
    data['true_range'] = np.maximum(
        data['high'] - data['low'],
        np.maximum(
            abs(data['high'] - data['close'].shift(1)),
            abs(data['low'] - data['close'].shift(1))
        )
    )
    data['price_efficiency'] = abs(data['close'] - data['close'].shift(1)) / data['true_range']
    
    # Volume confirmation
    data['volume_ratio'] = data['volume'] / data['volume'].shift(1)
    data['volume_confirmation'] = np.where(
        (data['returns'] > 0) & (data['volume_ratio'] > 1), 1,
        np.where((data['returns'] < 0) & (data['volume_ratio'] > 1), -1, 0)
    )
    
    # Reversal signal generation
    data['reversal_factor'] = -data['price_efficiency'] * data['volume_confirmation']
    
    # 3. Behavioral Temporal Factor
    # Intraday pattern extraction
    data['opening_gap'] = data['open'] / data['close'].shift(1) - 1
    data['midday_price'] = (data['high'] + data['low']) / 2
    data['closing_effect'] = data['close'] / data['midday_price'] - 1
    
    # Momentum divergence
    data['momentum_3d_smooth'] = data['momentum_3d'].rolling(window=3, min_periods=2).mean()
    data['momentum_7d'] = data['close'] / data['close'].shift(7) - 1
    data['momentum_divergence'] = data['momentum_3d_smooth'] - data['momentum_7d']
    
    # Volume trend consistency
    data['volume_trend'] = data['volume'].rolling(window=5, min_periods=3).apply(
        lambda x: 1 if (x.diff().dropna() > 0).sum() > 2 else -1, raw=False
    )
    
    # Pattern-enhanced signal
    data['behavioral_factor'] = (
        data['opening_gap'] * 0.3 + 
        data['closing_effect'] * 0.4 + 
        data['momentum_divergence'] * data['volume_trend'] * 0.3
    )
    
    # 4. Information Flow Factor
    # Market information content
    data['price_significance'] = abs(data['returns']) / data['volatility_10d']
    data['volume_surprise'] = (data['volume'] - data['volume'].rolling(window=10, min_periods=5).mean()) / data['volume'].rolling(window=10, min_periods=5).std()
    
    # Volatility regime adaptation
    data['info_vol_weight'] = np.where(
        data['high_vol_regime'] == 1,
        0.7,  # Higher weight in high volatility
        0.3   # Lower weight in low volatility
    )
    
    # Information-based prediction
    data['information_factor'] = (
        data['price_significance'] * data['volume_surprise'] * data['info_vol_weight']
    )
    
    # Final factor combination with normalization
    factors = ['momentum_factor', 'reversal_factor', 'behavioral_factor', 'information_factor']
    
    # Z-score normalization for each factor
    for factor in factors:
        data[f'{factor}_z'] = (
            data[factor] - data[factor].rolling(window=30, min_periods=15).mean()
        ) / data[factor].rolling(window=30, min_periods=15).std()
    
    # Weighted combination
    final_factor = (
        data['momentum_factor_z'] * 0.25 +
        data['reversal_factor_z'] * 0.30 +
        data['behavioral_factor_z'] * 0.25 +
        data['information_factor_z'] * 0.20
    )
    
    # Remove any remaining NaN values
    final_factor = final_factor.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    return final_factor
