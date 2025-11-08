import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Regime-Adaptive Multi-Dimensional Alpha Factor
    """
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Initialize result series
    result = pd.Series(index=df.index, dtype=float)
    
    # Volatility Regime Framework
    # True Range Calculation
    high_low_range = df['high'] - df['low']
    high_prev_close = abs(df['high'] - df['close'].shift(1))
    low_prev_close = abs(df['low'] - df['close'].shift(1))
    daily_range = pd.concat([high_low_range, high_prev_close, low_prev_close], axis=1).max(axis=1)
    
    # Rolling Volatility Classification
    rolling_median = daily_range.rolling(window=15, min_periods=10).median()
    high_vol_regime = daily_range > rolling_median.shift(1)
    low_vol_regime = ~high_vol_regime
    
    # Volume-Price Dynamics Signal
    # Multi-Timeframe Volume-Price Alignment
    def calculate_alignment(price_return, volume_return):
        return np.sign(price_return) * np.sign(volume_return)
    
    # Short-term (3-day)
    price_3d = df['close'].pct_change(3)
    volume_3d = df['volume'].pct_change(3)
    alignment_3d = calculate_alignment(price_3d, volume_3d)
    
    # Medium-term (8-day)
    price_8d = df['close'].pct_change(8)
    volume_8d = df['volume'].pct_change(8)
    alignment_8d = calculate_alignment(price_8d, volume_8d)
    
    # Long-term (15-day)
    price_15d = df['close'].pct_change(15)
    volume_15d = df['volume'].pct_change(15)
    alignment_15d = calculate_alignment(price_15d, volume_15d)
    
    # Volume-Price Consistency Score
    positive_alignments = pd.DataFrame({
        'short': (alignment_3d > 0).astype(int),
        'medium': (alignment_8d > 0).astype(int),
        'long': (alignment_15d > 0).astype(int)
    })
    weights = [3, 2, 1]  # short, medium, long
    consistency_score = (positive_alignments * weights).sum(axis=1) / sum(weights)
    
    # Volume-Price Momentum Divergence
    price_momentum_3d = abs(price_3d)
    volume_momentum_3d = abs(volume_3d)
    momentum_ratio = price_momentum_3d / (volume_momentum_3d + 1e-8)
    momentum_divergence = momentum_ratio.rolling(window=10, min_periods=5).apply(
        lambda x: (x.iloc[-1] - x.mean()) / (x.std() + 1e-8), raw=False
    )
    
    # Pressure Accumulation Framework
    # Multi-dimensional Pressure Calculation
    # Opening Pressure Component
    gap_pressure = (df['open'] - df['close'].shift(1)) * df['volume']
    opening_momentum = (df['high'] - df['open']) - (df['open'] - df['low'])
    
    # Intraday Pressure Component
    buying_pressure = (df['close'] - df['low']) * df['volume']
    selling_pressure = (df['high'] - df['close']) * df['volume']
    
    # Closing Pressure Component
    end_of_day_momentum = (df['close'] - df['open']) * df['volume']
    closing_range = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-8)
    
    # Net Pressure Index
    positive_pressure = buying_pressure + end_of_day_momentum * (end_of_day_momentum > 0)
    negative_pressure = selling_pressure + end_of_day_momentum * (end_of_day_momentum < 0)
    avg_volume_5d = df['volume'].rolling(window=5, min_periods=3).mean()
    net_pressure_index = (positive_pressure - negative_pressure) / (avg_volume_5d + 1e-8)
    
    # Pressure Trend Strength
    pressure_change = net_pressure_index.diff()
    pressure_acceleration = pressure_change.diff()
    pressure_persistence = net_pressure_index.groupby(
        (net_pressure_index.diff() != 0).cumsum()
    ).cumcount() + 1
    
    # Regime-Scaled Pressure
    pressure_high_vol = net_pressure_index * daily_range
    pressure_low_vol = net_pressure_index * ((df['high'] - df['low']) / df['close'].shift(1))
    
    # Momentum Quality Assessment
    # Multi-timeframe Momentum Quality
    momentum_5d = df['close'].pct_change(5)
    momentum_10d = df['close'].pct_change(10)
    momentum_20d = df['close'].pct_change(20)
    
    # Momentum Consistency
    alignment_5_10 = np.sign(momentum_5d) == np.sign(momentum_10d)
    alignment_10_20 = np.sign(momentum_10d) == np.sign(momentum_20d)
    consistency_score_momentum = alignment_5_10.astype(int) + alignment_10_20.astype(int)
    
    # Momentum Sustainability
    momentum_avg_5d = momentum_5d.rolling(window=5, min_periods=3).mean()
    momentum_acceleration = momentum_5d.diff()
    momentum_volatility = momentum_5d.rolling(window=10, min_periods=5).std()
    
    # Momentum Efficiency
    momentum_efficiency = abs(momentum_5d) / (df['volume'].rolling(window=5, min_periods=3).mean() + 1e-8)
    hist_efficiency = momentum_efficiency.rolling(window=20, min_periods=10).mean()
    efficiency_ratio = momentum_efficiency / (hist_efficiency + 1e-8)
    
    # Volume-Confirmed Momentum
    volume_confirmation = (alignment_3d > 0) & (alignment_8d > 0) & (consistency_score > 0.5)
    
    # Regime-Adaptive Signal Integration
    for idx in df.index:
        if high_vol_regime.loc[idx]:
            # High Volatility Regime
            # Primary: Volume-Price Dynamics
            vp_signal = consistency_score.loc[idx] * 0.4 + momentum_divergence.loc[idx] * 0.3
            
            # Secondary: Momentum Quality
            mq_signal = (consistency_score_momentum.loc[idx] * 0.5 + 
                         volume_confirmation.loc[idx] * 0.3 + 
                         efficiency_ratio.loc[idx] * 0.2)
            
            # Tertiary: Pressure Accumulation
            pa_signal = pressure_high_vol.loc[idx] * 0.6 + pressure_persistence.loc[idx] * 0.4
            
            # Combined signal for high volatility
            result.loc[idx] = (vp_signal * 0.5 + mq_signal * 0.3 + pa_signal * 0.2)
            
        else:
            # Low Volatility Regime
            # Primary: Pressure Accumulation
            pa_signal = pressure_low_vol.loc[idx] * 0.5 + pressure_change.loc[idx] * 0.3
            
            # Secondary: Volume-Price Dynamics
            vp_signal = consistency_score.loc[idx] * 0.4 + (alignment_3d.loc[idx] + alignment_8d.loc[idx]) * 0.3
            
            # Tertiary: Momentum Quality
            mq_signal = consistency_score_momentum.loc[idx] * 0.6 + efficiency_ratio.loc[idx] * 0.4
            
            # Combined signal for low volatility
            result.loc[idx] = (pa_signal * 0.6 + vp_signal * 0.3 + mq_signal * 0.1)
    
    # Normalize the final factor
    result = (result - result.rolling(window=20, min_periods=10).mean()) / (result.rolling(window=20, min_periods=10).std() + 1e-8)
    
    return result
