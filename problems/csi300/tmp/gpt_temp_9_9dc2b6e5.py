import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # 1. Regime-Adaptive Momentum
    # Multi-timeframe momentum calculations
    data['mom_3d'] = data['close'] / data['close'].shift(3) - 1
    data['mom_8d'] = data['close'] / data['close'].shift(8) - 1
    data['mom_15d'] = data['close'] / data['close'].shift(15) - 1
    
    # Volatility regime using 10-day rolling standard deviation
    data['volatility_10d'] = data['close'].pct_change().rolling(window=10).std()
    volatility_regime = data['volatility_10d'].rolling(window=20).apply(
        lambda x: 1 if x.iloc[-1] > x.median() else 0, raw=False
    )
    
    # Regime-weighted momentum combination
    data['regime_momentum'] = (
        volatility_regime * data['mom_3d'] + 
        (1 - volatility_regime) * (0.4 * data['mom_8d'] + 0.6 * data['mom_15d'])
    )
    
    # 2. Microstructure Reversal
    # Price efficiency: |Close-Close_prev|/TrueRange
    true_range = np.maximum(
        data['high'] - data['low'],
        np.maximum(
            abs(data['high'] - data['close'].shift(1)),
            abs(data['low'] - data['close'].shift(1))
        )
    )
    data['price_efficiency'] = abs(data['close'] - data['close'].shift(1)) / true_range
    
    # Volume confirmation
    data['volume_confirmation'] = data['volume'] / data['volume'].shift(1)
    
    # Contrarian signal generation
    data['reversal_signal'] = -data['mom_3d'] * data['price_efficiency'] * np.log(data['volume_confirmation'])
    
    # 3. Behavioral Temporal
    # Intraday patterns
    data['opening_gap'] = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
    data['closing_effect'] = (data['close'] - data['open']) / data['open']
    
    # Momentum divergence (3d vs 7d)
    data['mom_7d'] = data['close'] / data['close'].shift(7) - 1
    data['momentum_divergence'] = data['mom_3d'] - data['mom_7d']
    
    # Pattern-weighted signal
    data['behavioral_signal'] = (
        np.sign(data['opening_gap']) * data['closing_effect'] * 
        np.tanh(data['momentum_divergence'])
    )
    
    # 4. Information Flow
    # Price significance (abnormal returns)
    data['price_significance'] = abs(data['close'].pct_change()) / data['close'].pct_change().rolling(window=20).std()
    
    # Volume surprise
    data['volume_surprise'] = (data['volume'] - data['volume'].rolling(window=20).mean()) / data['volume'].rolling(window=20).std()
    
    # Volatility adaptation for information weighting
    vol_weight = 1 / (1 + data['volatility_10d'])
    
    # Information-based prediction
    data['information_signal'] = (
        vol_weight * data['price_significance'] * 
        np.tanh(data['volume_surprise']) * data['mom_3d']
    )
    
    # Final alpha factor combination
    alpha_factor = (
        0.3 * data['regime_momentum'] +
        0.25 * data['reversal_signal'] +
        0.25 * data['behavioral_signal'] +
        0.2 * data['information_signal']
    )
    
    # Normalize the final factor
    alpha_factor = (alpha_factor - alpha_factor.rolling(window=50).mean()) / alpha_factor.rolling(window=50).std()
    
    return alpha_factor
