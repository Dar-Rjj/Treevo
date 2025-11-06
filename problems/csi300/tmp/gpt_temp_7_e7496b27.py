import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import entropy

def heuristics_v2(df):
    data = df.copy()
    
    # Calculate necessary intermediate series
    data['prev_close'] = data['close'].shift(1)
    data['true_range'] = np.maximum(
        data['high'] - data['low'],
        np.maximum(
            abs(data['high'] - data['prev_close']),
            abs(data['low'] - data['prev_close'])
        )
    )
    
    # Volatility Regime Classification
    data['vol_5d'] = data['close'].pct_change().rolling(window=5).std()
    data['vol_20d'] = data['close'].pct_change().rolling(window=20).std()
    data['vol_ratio'] = data['vol_5d'] / data['vol_20d']
    
    def classify_regime(ratio):
        if ratio > 1.2:
            return 2  # High volatility
        elif ratio < 0.8:
            return 0  # Low volatility
        else:
            return 1  # Transition
    
    data['vol_regime'] = data['vol_ratio'].apply(classify_regime)
    
    # Entropy Asymmetry Detection
    def calculate_entropy(series, window=8):
        ranks = series.rolling(window=window).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
        )
        # Calculate histogram-based entropy
        hist_entropy = ranks.rolling(window=window).apply(
            lambda x: entropy(np.histogram(x.dropna(), bins=5, density=True)[0]), 
            raw=False
        )
        return hist_entropy
    
    data['volatility_entropy'] = calculate_entropy(data['true_range'])
    data['volume_entropy'] = calculate_entropy(data['volume'])
    data['entropy_asymmetry'] = data['volatility_entropy'] - data['volume_entropy']
    
    # Adaptive Momentum System
    data['mom_3d'] = (data['close'] - data['close'].shift(3)) / data['close'].shift(3)
    data['mom_6d'] = (data['close'] - data['close'].shift(6)) / data['close'].shift(6)
    data['acceleration'] = data['mom_3d'] - data['mom_6d']
    
    # 8-day persistence
    def count_consecutive_sign(series, window=8):
        def count_func(x):
            if len(x) < 2:
                return 0
            signs = np.sign(x)
            count = 0
            for i in range(1, len(signs)):
                if signs[i] == signs[i-1] and signs[i] != 0:
                    count += 1
                else:
                    break
            return count
        return series.rolling(window=window).apply(count_func, raw=True)
    
    data['momentum_sign'] = np.sign(data['close'].pct_change())
    data['persistence'] = count_consecutive_sign(data['momentum_sign'])
    
    # 15-day reversal
    data['mom_15d_8d'] = (data['close'].shift(8) - data['close'].shift(15)) / data['close'].shift(15)
    data['mom_recent_7d'] = (data['close'] - data['close'].shift(7)) / data['close'].shift(7)
    data['reversal_strength'] = abs(data['mom_15d_8d'] - data['mom_recent_7d'])
    data['reversal_strength'] = data['reversal_strength'].replace(0, 0.0001)  # Avoid division by zero
    
    data['combined_momentum'] = (data['acceleration'] * data['persistence']) / data['reversal_strength']
    
    # Volume Anomaly Detection
    data['vol_5d_avg'] = data['volume'].rolling(window=5).mean()
    data['vol_20d_avg'] = data['volume'].rolling(window=20).mean()
    data['volume_ratio'] = data['vol_5d_avg'] / data['vol_20d_avg']
    
    # Price-volume phase angle
    data['price_change_5d'] = data['close'].pct_change(5)
    data['volume_change_5d'] = data['volume'].pct_change(5)
    
    def calculate_phase_angle(price_chg, vol_chg):
        if price_chg == 0 and vol_chg == 0:
            return 0
        return np.arctan2(vol_chg, price_chg)
    
    data['phase_angle'] = data.apply(
        lambda row: calculate_phase_angle(row['price_change_5d'], row['volume_change_5d']), 
        axis=1
    )
    
    # Phase angle consistency (rolling standard deviation)
    data['phase_consistency'] = 1 / (1 + data['phase_angle'].rolling(window=5).std())
    data['volume_anomaly'] = data['volume_ratio'] * data['phase_consistency']
    
    # Factor Synthesis
    # Regime weights: high vol = 1.5, transition = 1.0, low vol = 0.7
    regime_weights = {2: 1.5, 1: 1.0, 0: 0.7}
    data['regime_weight'] = data['vol_regime'].map(regime_weights)
    
    # Combined factor
    data['combined_entropy_momentum'] = data['entropy_asymmetry'] * data['combined_momentum']
    data['regime_weighted'] = data['combined_entropy_momentum'] * data['regime_weight']
    data['final_factor'] = data['regime_weighted'] * data['volume_anomaly']
    
    # Apply momentum direction sign
    data['momentum_direction'] = np.sign(data['mom_3d'])
    data['final_factor'] = data['final_factor'] * data['momentum_direction']
    
    return data['final_factor']
