import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Fractal Entropy Momentum with Asymmetric Microstructure
    """
    # Copy data to avoid modifying original
    data = df.copy()
    
    # Helper function for Shannon entropy
    def shannon_entropy(series, window):
        """Calculate Shannon entropy of percentile distribution"""
        if len(series) < window:
            return pd.Series([np.nan] * len(series), index=series.index)
        
        entropy_values = []
        for i in range(len(series)):
            if i < window - 1:
                entropy_values.append(np.nan)
                continue
                
            window_data = series.iloc[i-window+1:i+1]
            # Calculate percentiles (0-100)
            percentiles = pd.cut(window_data, bins=10, labels=False, duplicates='drop')
            if len(percentiles.dropna()) == 0:
                entropy_values.append(np.nan)
                continue
                
            # Calculate probability distribution
            value_counts = percentiles.value_counts(normalize=True)
            # Shannon entropy
            entropy = -np.sum(value_counts * np.log(value_counts + 1e-10))
            entropy_values.append(entropy)
            
        return pd.Series(entropy_values, index=series.index)
    
    # 1. Multi-Scale Entropy Asymmetry
    # Volatility Regime Entropy
    # True Range calculation
    data['TR'] = np.maximum(
        data['high'] - data['low'],
        np.maximum(
            abs(data['high'] - data['close'].shift(1)),
            abs(data['low'] - data['close'].shift(1))
        )
    )
    
    # Volatility entropy for different timeframes
    data['vol_entropy_2d'] = shannon_entropy(data['TR'], 2)
    data['vol_entropy_10d'] = shannon_entropy(data['TR'], 10)
    
    # Volatility entropy asymmetry
    data['vol_entropy_asym'] = data['vol_entropy_10d'] - data['vol_entropy_2d']
    data['vol_entropy_asym_ma'] = data['vol_entropy_asym'].rolling(window=8, min_periods=1).mean()
    data['vol_entropy_direction'] = np.where(data['vol_entropy_asym'] > data['vol_entropy_asym_ma'], 1, -1)
    
    # Volume-Amount Fractal Entropy
    data['volume_entropy'] = shannon_entropy(data['volume'], 8)
    data['amount_entropy'] = shannon_entropy(data['amount'], 8)
    
    # Volume-Amount entropy divergence
    data['vol_amt_entropy_div'] = data['volume_entropy'] - data['amount_entropy']
    data['vol_amt_div_ma'] = data['vol_amt_entropy_div'].rolling(window=8, min_periods=1).mean()
    data['vol_amt_regime'] = np.where(data['vol_amt_entropy_div'] > data['vol_amt_div_ma'], 1, -1)
    
    # 2. Asymmetric Momentum Fractals
    # Multi-Timeframe Momentum Efficiency
    # Ultra-short term (intraday momentum)
    data['intraday_ret'] = (data['close'] - data['open']) / data['open']
    data['intraday_momentum'] = data['intraday_ret'].rolling(window=3, min_periods=1).mean()
    
    # Short-term momentum (2-3 days)
    data['ret_3d'] = data['close'].pct_change(periods=3)
    data['momentum_accel'] = data['ret_3d'] - data['ret_3d'].shift(3)
    data['momentum_persistence'] = data['ret_3d'].rolling(window=3, min_periods=1).apply(
        lambda x: np.sum(np.sign(x) == np.sign(x.iloc[-1])) if len(x.dropna()) > 0 else np.nan
    )
    
    # Medium-term momentum (5-7 days)
    data['ret_5d'] = data['close'].pct_change(periods=5)
    data['ret_7d'] = data['close'].pct_change(periods=7)
    data['momentum_reversal'] = np.where(
        data['ret_5d'] * data['ret_7d'] < 0,
        abs(data['ret_5d'] - data['ret_7d']),
        0
    )
    
    # Path-Dependent Momentum Coupling
    # Price-Volume Phase Angle
    data['price_change_5d'] = data['close'] - data['close'].shift(5)
    data['volume_change_5d'] = data['volume'] - data['volume'].shift(5)
    
    # Calculate phase angle using dot product
    def calculate_phase_angle(row):
        price_vec = row['price_change_5d']
        volume_vec = row['volume_change_5d']
        if pd.isna(price_vec) or pd.isna(volume_vec) or price_vec == 0 or volume_vec == 0:
            return np.nan
        dot_product = price_vec * volume_vec
        magnitude_price = abs(price_vec)
        magnitude_volume = abs(volume_vec)
        cosine_sim = dot_product / (magnitude_price * magnitude_volume + 1e-10)
        cosine_sim = np.clip(cosine_sim, -1, 1)
        return np.arccos(cosine_sim)
    
    data['phase_angle'] = data.apply(calculate_phase_angle, axis=1)
    
    # Phase consistency
    data['phase_variance'] = data['phase_angle'].rolling(window=8, min_periods=1).std()
    data['phase_stability'] = 1 / (data['phase_variance'] + 1e-10)
    
    # Combine momentum with path efficiency
    data['momentum_efficiency'] = (
        data['intraday_momentum'] * 
        data['momentum_persistence'] * 
        data['phase_angle'] * 
        data['phase_stability']
    )
    
    # 3. Microstructure Asymmetric Pressure
    # Opening Gap Pressure Analysis
    data['high_open_asym'] = (data['high'] - data['open']) / data['open']
    data['open_low_asym'] = (data['open'] - data['low']) / data['open']
    data['gap_pressure'] = data['high_open_asym'] - data['open_low_asym']
    
    # Volume-Amount Regime Mismatch
    data['range'] = (data['high'] - data['low']) / data['close']
    data['volume_range_ratio'] = data['volume'] / (data['range'] + 1e-10)
    data['vol_vol_mismatch'] = data['volume_range_ratio'].rolling(window=5, min_periods=1).apply(
        lambda x: (x.iloc[-1] - x.mean()) / (x.std() + 1e-10) if len(x.dropna()) > 1 else np.nan
    )
    
    # 4. Regime-Adaptive Signal Synthesis
    # Volatility regime classification
    vol_ma = data['TR'].rolling(window=20, min_periods=1).mean()
    vol_std = data['TR'].rolling(window=20, min_periods=1).std()
    data['vol_regime'] = np.where(
        data['TR'] > vol_ma + vol_std, 2,  # High volatility
        np.where(data['TR'] < vol_ma - vol_std, 0, 1)  # Low volatility
    )
    
    # Multi-Scale Factor Integration
    # Combine entropy asymmetry with momentum fractals
    entropy_momentum = (
        data['vol_entropy_direction'] * 
        data['momentum_efficiency'] * 
        data['vol_amt_regime']
    )
    
    # Apply microstructure pressure adjustment
    microstructure_adjustment = (
        data['gap_pressure'] * 
        data['vol_vol_mismatch']
    )
    
    # Regime-specific coefficients
    regime_coeff = np.where(
        data['vol_regime'] == 2, 0.7,  # High volatility - reduce sensitivity
        np.where(data['vol_regime'] == 0, 1.3, 1.0)  # Low volatility - enhance sensitivity
    )
    
    # Final alpha signal
    alpha_signal = (
        entropy_momentum * 
        microstructure_adjustment * 
        regime_coeff
    )
    
    # Apply logarithmic transformation for normalization
    final_alpha = np.sign(alpha_signal) * np.log1p(abs(alpha_signal))
    
    return final_alpha
