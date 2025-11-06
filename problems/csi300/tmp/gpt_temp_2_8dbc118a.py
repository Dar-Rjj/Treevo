import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Copy dataframe to avoid modifying original
    data = df.copy()
    
    # Helper function for ATR
    def calculate_atr(data, window=5):
        high_low = data['high'] - data['low']
        high_close_prev = abs(data['high'] - data['close'].shift(1))
        low_close_prev = abs(data['low'] - data['close'].shift(1))
        true_range = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
        return true_range.rolling(window=window).mean()
    
    # Calculate basic components
    data['close_1'] = data['close'].shift(1)
    data['close_5'] = data['close'].shift(5)
    data['volume_1'] = data['volume'].shift(1)
    
    # Volatility calculations
    data['vol_2d'] = data['close'].pct_change().rolling(window=2).std()
    data['vol_10d'] = data['close'].pct_change().rolling(window=10).std()
    data['atr_5'] = calculate_atr(data, 5)
    
    # Volume concentration
    data['volume_ma_5'] = data['volume'].rolling(window=5).mean()
    data['volume_concentration'] = data['volume'] / data['volume_ma_5']
    
    # Multi-Scale Volatility Divergence
    volatility_divergence = (data['vol_2d'] / data['vol_10d']) * data['volume_concentration']
    
    # Volatility Efficiency Divergence
    close_diff_5 = abs(data['close'] - data['close_5'])
    high_low_sum_5 = (data['high'] - data['low']).rolling(window=5).sum()
    volume_sum_5 = data['volume'].rolling(window=5).sum()
    
    volatility_efficiency_divergence = (close_diff_5 / high_low_sum_5) - (close_diff_5 / volume_sum_5)
    
    # Fractal Range Compression
    volatility_regime_ratio = data['vol_2d'] / data['vol_10d']
    fractal_range_compression = ((data['high'] - data['low']) / data['atr_5']) * volatility_regime_ratio
    
    # Price-Volume Pressure Association
    price_change_sign = np.sign(data['close'] - data['close_1'])
    volume_change_sign = np.sign(data['volume'] - data['volume_1'])
    net_pressure = (data['close'] - data['open']) / (data['high'] - data['low']).replace(0, 1)
    price_volume_pressure = price_change_sign * volume_change_sign * net_pressure
    
    # Bid-Ask Flow Divergence
    close_low_amount_3 = ((data['close'] - data['low']) * data['amount']).rolling(window=3).sum()
    high_close_amount_3 = ((data['high'] - data['close']) * data['amount']).rolling(window=3).sum()
    bid_ask_flow = np.log(close_low_amount_3 / high_close_amount_3.replace(0, 1)) * data['volume_concentration']
    
    # Pressure Convergence Divergence
    pressure_3d = price_volume_pressure.rolling(window=3).mean()
    pressure_8d = price_volume_pressure.rolling(window=8).mean()
    price_amount_association = np.corrcoef(data['close'].rolling(window=5).apply(lambda x: np.corrcoef(x, data['amount'].loc[x.index])[0,1] if len(x) > 1 else 0), 
                                          data['amount'].rolling(window=5).mean())[0,1]
    pressure_convergence_divergence = (pressure_3d - pressure_8d) * price_amount_association
    
    # Turnover Cluster Divergence
    turnover = data['volume'] * data['close']
    turnover_ma_20 = turnover.rolling(window=20).mean()
    turnover_cluster_divergence = (turnover / turnover_ma_20) * data['volume_concentration']
    
    # Volatility Cluster Intensity
    high_vol_days = (data['vol_2d'] > data['vol_2d'].rolling(window=20).quantile(0.7)).rolling(window=5).sum()
    avg_range_expansion = ((data['high'] - data['low']) / data['atr_5']).rolling(window=5).mean()
    volatility_cluster_intensity = high_vol_days * avg_range_expansion
    
    # Cluster Duration Multiplier
    high_turnover = turnover > turnover_ma_20
    high_volatility = data['vol_2d'] > data['vol_2d'].rolling(window=20).quantile(0.7)
    
    def consecutive_count(series):
        return series * (series.groupby((series != series.shift()).cumsum()).cumcount() + 1)
    
    turnover_duration = consecutive_count(high_turnover)
    volatility_duration = consecutive_count(high_volatility)
    cluster_duration_multiplier = turnover_duration * volatility_duration
    
    # Microstructure Range Pressure (simplified as intraday pressure)
    morning_pressure = (data['open'] - data['low']) / (data['high'] - data['low']).replace(0, 1)
    afternoon_pressure = (data['close'] - data['open']) / (data['high'] - data['low']).replace(0, 1)
    efficiency_expansion = (data['high'] - data['low']) / data['atr_5']
    microstructure_range_pressure = (morning_pressure - afternoon_pressure) * efficiency_expansion
    
    # Fractal Boundary Efficiency
    high_above_prev = data['high'] > data['high'].shift(1)
    low_below_prev = data['low'] < data['low'].shift(1)
    fractal_boundary = (high_above_prev & low_below_prev).rolling(window=5).sum()
    range_compression_divergence = ((data['high'] - data['low']) / data['atr_5']) / ((data['high'] - data['low']).rolling(window=5).mean() / data['atr_5'].rolling(window=5).mean())
    fractal_boundary_efficiency = fractal_boundary * range_compression_divergence
    
    # Pressure Asymmetry Divergence
    volume_confirmation = data['volume'] / data['volume_ma_5']
    pressure_asymmetry_divergence = (morning_pressure / afternoon_pressure.replace(0, 1)) * volume_confirmation
    
    # Multi-Timeframe Regime Switching
    short_term_regime = volatility_efficiency_divergence * price_volume_pressure
    medium_term_regime = pressure_convergence_divergence * turnover_cluster_divergence
    long_term_regime = fractal_range_compression * volatility_cluster_intensity
    
    regime_transition_signal = np.sign(short_term_regime) * np.sign(medium_term_regime) * np.sign(long_term_regime)
    
    # Adaptive Fractal Volatility Integration
    core_divergence_signal = volatility_efficiency_divergence * bid_ask_flow * pressure_convergence_divergence
    
    fractal_regime_filter = np.where(fractal_boundary_efficiency > 0, 1.0, 0.7)
    
    turnover_cluster_active = turnover > turnover_ma_20
    volatility_cluster_active = data['vol_2d'] > data['vol_2d'].rolling(window=20).quantile(0.7)
    cluster_enhancement_multiplier = np.where(turnover_cluster_active & volatility_cluster_active, 1.5, 1.0)
    
    net_pressure_abs = abs(net_pressure)
    pressure_confirmation_weight = np.where(net_pressure_abs > 0.6, 1.4, 0.7)
    
    volatility_regime_weight = np.where(volatility_regime_ratio > 1.2, 1.3, 0.8)
    
    regime_transition_multiplier = abs(regime_transition_signal) * cluster_duration_multiplier
    
    # Final Alpha Generation
    base_alpha = core_divergence_signal * fractal_regime_filter * cluster_enhancement_multiplier
    confirmation_alpha = base_alpha * pressure_confirmation_weight * volatility_regime_weight
    final_alpha = confirmation_alpha * regime_transition_multiplier * microstructure_range_pressure
    
    return final_alpha
