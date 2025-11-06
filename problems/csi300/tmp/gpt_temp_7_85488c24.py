import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Regime-Adaptive Entropy-Volume Momentum factor
    """
    data = df.copy()
    
    # 1. Regime Classification
    # Volatility regime
    data['volatility'] = (data['high'] - data['low']) / data['close']
    
    # Trend regime
    data['trend_short'] = np.sign(data['close'] - data['close'].shift(5))
    data['trend_medium'] = np.sign(data['close'] - data['close'].shift(10))
    data['trend'] = data['trend_short'] * data['trend_medium']
    
    # Liquidity regime
    data['volume_ma_20'] = data['volume'].rolling(window=20, min_periods=10).mean()
    data['liquidity'] = data['volume'] / data['volume_ma_20']
    
    # 2. Entropy-Volume Dynamics
    # True Range calculation
    data['tr1'] = data['high'] - data['low']
    data['tr2'] = abs(data['high'] - data['close'].shift(1))
    data['tr3'] = abs(data['low'] - data['close'].shift(1))
    data['true_range'] = data[['tr1', 'tr2', 'tr3']].max(axis=1)
    
    # Price Entropy (Shannon entropy of True Range percentile ranks)
    def calculate_entropy(series, window=20):
        entropy_values = []
        for i in range(len(series)):
            if i < window:
                entropy_values.append(np.nan)
                continue
            window_data = series.iloc[i-window:i]
            ranks = window_data.rank(pct=True)
            # Calculate Shannon entropy
            hist, _ = np.histogram(ranks, bins=10, range=(0, 1), density=True)
            hist = hist[hist > 0]  # Remove zeros for log calculation
            entropy = -np.sum(hist * np.log(hist))
            entropy_values.append(entropy)
        return pd.Series(entropy_values, index=series.index)
    
    data['price_entropy'] = calculate_entropy(data['true_range'])
    
    # Volume-Price Asymmetry (Buy pressure - Sell pressure)
    data['typical_price'] = (data['high'] + data['low'] + data['close']) / 3
    data['buy_pressure'] = np.where(data['close'] > data['open'], 
                                   data['volume'] * (data['close'] - data['open']), 0)
    data['sell_pressure'] = np.where(data['close'] < data['open'], 
                                    data['volume'] * (data['open'] - data['close']), 0)
    data['volume_asymmetry'] = (data['buy_pressure'] - data['sell_pressure']) / \
                              (data['buy_pressure'] + data['sell_pressure'] + 1e-8)
    
    # Entropy-Volume Coupling
    data['entropy_volume_coupling'] = data['price_entropy'] * data['volume_asymmetry']
    
    # 3. Adaptive Momentum
    # Short-term momentum
    data['momentum_short'] = (data['close'] - data['close'].shift(3)) / data['close'].shift(3)
    
    # Medium-term momentum
    data['momentum_medium'] = (data['close'] - data['close'].shift(8)) / data['close'].shift(8)
    
    # Memory-weighted momentum (exponential average)
    alpha = 0.3
    data['momentum_weighted'] = data['momentum_short'].ewm(alpha=alpha, adjust=False).mean()
    
    # 4. Volume-Price Phase
    # Volume Anomaly
    data['volume_median_20'] = data['volume'].rolling(window=20, min_periods=10).median()
    data['volume_anomaly'] = data['volume'] / data['volume_median_20']
    
    # Phase Angle (angle between 5-day price and volume vectors)
    def calculate_phase_angle(price_series, volume_series, window=5):
        angles = []
        for i in range(len(price_series)):
            if i < window:
                angles.append(np.nan)
                continue
            price_vec = price_series.iloc[i-window:i].values
            volume_vec = volume_series.iloc[i-window:i].values
            
            # Normalize vectors
            price_vec = (price_vec - np.mean(price_vec)) / (np.std(price_vec) + 1e-8)
            volume_vec = (volume_vec - np.mean(volume_vec)) / (np.std(volume_vec) + 1e-8)
            
            # Calculate cosine similarity and convert to angle
            dot_product = np.dot(price_vec, volume_vec)
            norm_product = np.linalg.norm(price_vec) * np.linalg.norm(volume_vec)
            cosine_sim = dot_product / (norm_product + 1e-8)
            angle = np.arccos(np.clip(cosine_sim, -1, 1))
            angles.append(angle)
        return pd.Series(angles, index=price_series.index)
    
    data['phase_angle'] = calculate_phase_angle(data['close'], data['volume'])
    
    # Phase Consistency
    data['phase_consistency'] = data['volume_anomaly'] * (1 - data['phase_angle'] / np.pi)
    
    # 5. Signal Synthesis
    # Regime weights
    data['volatility_weight'] = 1 / (1 + data['volatility'].rolling(window=10).std())
    data['trend_weight'] = abs(data['trend'])
    data['liquidity_weight'] = np.tanh(data['liquidity'])
    
    # Component combination with regime weighting
    momentum_component = (data['momentum_short'] * 0.4 + 
                         data['momentum_medium'] * 0.3 + 
                         data['momentum_weighted'] * 0.3)
    
    entropy_component = data['entropy_volume_coupling'] * data['phase_consistency']
    
    # Cross-regime validation
    regime_filter = ((data['volatility'] < data['volatility'].rolling(window=20).quantile(0.8)) &
                    (data['liquidity'] > data['liquidity'].rolling(window=20).quantile(0.2)))
    
    # Final alpha generation
    alpha_signal = (momentum_component * data['volatility_weight'] + 
                   entropy_component * data['liquidity_weight']) * regime_filter
    
    # Normalize and clean
    alpha_signal = alpha_signal.replace([np.inf, -np.inf], np.nan)
    alpha_signal = (alpha_signal - alpha_signal.rolling(window=50, min_periods=20).mean()) / \
                   (alpha_signal.rolling(window=50, min_periods=20).std() + 1e-8)
    
    return alpha_signal
