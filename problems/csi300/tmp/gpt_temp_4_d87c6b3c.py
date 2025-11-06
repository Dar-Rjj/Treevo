import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Multi-Timeframe Price-Volume Fractality with Microstructure Entropy alpha factor
    """
    data = df.copy()
    
    # 1. Price Fractality Metrics
    # Hurst exponent estimation using high-low range
    def hurst_exponent(series, window):
        lags = range(2, min(6, window))
        tau = []
        for lag in lags:
            rs = (series.rolling(lag).max() - series.rolling(lag).min()) / series.rolling(lag).std()
            tau.append(np.log(rs.mean() if not np.isnan(rs.mean()) else 1.0))
        
        if len(tau) > 1:
            hurst = np.polyfit(np.log(lags), tau, 1)[0]
            return hurst
        return 0.5
    
    # Calculate multi-timeframe fractal dimensions
    high_low_range = (data['high'] - data['low']) / data['close']
    
    # 5-day fractal dimension
    fractal_5d = 2 - high_low_range.rolling(5).apply(
        lambda x: hurst_exponent(x, 5), raw=False
    ).fillna(1.5)
    
    # 10-day fractal dimension  
    fractal_10d = 2 - high_low_range.rolling(10).apply(
        lambda x: hurst_exponent(x, 10), raw=False
    ).fillna(1.5)
    
    # Fractal consistency
    fractal_consistency = (fractal_5d + fractal_10d) / 2
    
    # 2. Volume Distribution Fractality
    vol_5d_avg = data['volume'].rolling(5).mean()
    vol_concentration = data['volume'] / vol_5d_avg
    
    # Volume fractal dimension using daily patterns
    vol_range = data['volume'].rolling(5).std() / data['volume'].rolling(5).mean()
    vol_fractal = 2 - vol_range.fillna(1.0)
    
    # Price-Volume Fractal Alignment
    fractal_divergence = fractal_5d - vol_fractal
    
    # 3. Microstructure Entropy
    # Session efficiency
    morning_efficiency = (data['high'] - data['open']) / (data['high'] - data['low']).replace(0, 1e-10)
    afternoon_efficiency = (data['close'] - data['low']) / (data['high'] - data['low']).replace(0, 1e-10)
    
    intraday_entropy = abs(morning_efficiency - afternoon_efficiency)
    entropy_persistence = intraday_entropy.rolling(5).mean()
    
    # Volume Flow Asymmetry
    directional_flow = np.sign(data['close'] - data['open']) * data['volume']
    cum_flow_3d = directional_flow.rolling(3).sum()
    
    # Morning vs afternoon flow (simplified)
    morning_flow = (data['high'] - data['open']) * data['volume']
    afternoon_flow = (data['close'] - data['low']) * data['volume']
    flow_asymmetry = morning_flow / afternoon_flow.replace(0, 1e-10)
    
    # 4. Gap Fractality
    gap_magnitude = abs(data['open'] - data['close'].shift(1)) / data['close'].shift(1)
    gap_direction_persistence = np.sign(data['open'] - data['close'].shift(1)) * np.sign(data['close'] - data['open'])
    
    # Gap fractal dimension (simplified)
    gap_fractal = 2 - gap_magnitude.rolling(5).std().fillna(0.1)
    
    # Intraday fractal momentum
    open_high_efficiency = (data['high'] - data['open']) / (data['high'] - data['low']).replace(0, 1e-10)
    high_close_efficiency = (data['close'] - data['low']) / (data['high'] - data['low']).replace(0, 1e-10)
    
    # 5. Price-Volume Correlation Fractality
    price_vol_corr = data['close'].rolling(5).corr(data['volume'])
    correlation_fractal = 2 - abs(price_vol_corr).fillna(0.5)
    
    # Asymmetric correlation
    up_mask = data['close'] > data['close'].shift(1)
    down_mask = data['close'] < data['close'].shift(1)
    
    upside_corr = data['close'][up_mask].rolling(5).corr(data['volume'][up_mask]).fillna(0)
    downside_corr = data['close'][down_mask].rolling(5).corr(data['volume'][down_mask]).fillna(0)
    
    correlation_asymmetry = upside_corr - downside_corr
    
    # 6. Entropy Regime Classification
    entropy_regime = pd.cut(intraday_entropy, 
                           bins=[0, 0.2, 0.5, 1.0], 
                           labels=[0, 1, 2]).astype(float)
    
    # Fractal efficiency score
    price_efficiency = 1 - (data['high'] - data['low']) / data['close']
    volume_efficiency = data['volume'] / data['volume'].rolling(10).mean()
    fractal_efficiency = (price_efficiency + volume_efficiency) / 2
    
    # 7. Composite Fractal Alpha Generation
    # Multi-fractal signal integration with entropy-adaptive weighting
    
    # Entropy-based weights
    low_entropy_weight = (entropy_regime == 0).astype(float) * 0.6
    med_entropy_weight = (entropy_regime == 1).astype(float) * 0.3
    high_entropy_weight = (entropy_regime == 2).astype(float) * 0.1
    
    entropy_weight = low_entropy_weight + med_entropy_weight + high_entropy_weight
    
    # Fractal regime strength
    fractal_strength = (fractal_consistency.rolling(3).std() + 
                       fractal_divergence.rolling(3).std()) / 2
    
    # Core fractal signals
    price_fractal_signal = fractal_consistency * fractal_efficiency
    volume_fractal_signal = vol_fractal * vol_concentration
    microstructure_signal = (1 - intraday_entropy) * cum_flow_3d / data['volume'].rolling(5).mean()
    gap_fractal_signal = gap_fractal * gap_direction_persistence
    correlation_signal = correlation_fractal * (1 + correlation_asymmetry)
    
    # Dynamic integration with entropy-adaptive weights
    composite_alpha = (
        entropy_weight * price_fractal_signal +
        (1 - entropy_weight) * volume_fractal_signal +
        microstructure_signal * flow_asymmetry.rolling(3).mean() +
        gap_fractal_signal * open_high_efficiency +
        correlation_signal * high_close_efficiency
    ) / 5
    
    # Final signal with regime classification
    final_signal = composite_alpha * fractal_strength
    
    return final_signal.fillna(0)
