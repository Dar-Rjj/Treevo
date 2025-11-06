import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    data = df.copy()
    
    # Multi-Scale Efficiency-Volatility Alignment
    # Calculate True Range
    data['prev_close'] = data['close'].shift(1)
    data['tr1'] = data['high'] - data['low']
    data['tr2'] = abs(data['high'] - data['prev_close'])
    data['tr3'] = abs(data['low'] - data['prev_close'])
    data['true_range'] = data[['tr1', 'tr2', 'tr3']].max(axis=1)
    
    # Intraday Efficiency Ratio
    data['intraday_efficiency'] = (data['close'] - data['open']) / data['true_range']
    
    # Volatility Persistence
    data['vol_short_trend'] = np.sign(data['true_range'] - data['true_range'].shift(3))
    data['vol_medium_trend'] = np.sign(data['true_range'] - data['true_range'].shift(8))
    
    # Multi-Scale Alignment Signal
    alignment_signal = np.zeros(len(data))
    conditions = [
        (data['intraday_efficiency'] > 0) & (data['vol_short_trend'] > 0) & (data['vol_medium_trend'] > 0),
        (data['intraday_efficiency'] < 0) & (data['vol_short_trend'] < 0) & (data['vol_medium_trend'] < 0),
        (data['intraday_efficiency'] > 0) & (data['vol_short_trend'] < 0) & (data['vol_medium_trend'] < 0),
        (data['intraday_efficiency'] < 0) & (data['vol_short_trend'] > 0) & (data['vol_medium_trend'] > 0)
    ]
    choices = [2, 2, -2, -2]
    alignment_signal = np.select(conditions, choices, default=0)
    
    # Mixed alignment patterns
    mixed_conditions = [
        (data['intraday_efficiency'] > 0) & (data['vol_short_trend'] + data['vol_medium_trend'] > 0),
        (data['intraday_efficiency'] < 0) & (data['vol_short_trend'] + data['vol_medium_trend'] < 0)
    ]
    mixed_choices = [1, -1]
    alignment_signal = np.select(mixed_conditions, mixed_choices, default=alignment_signal)
    
    # Volume-Weighted Multi-Timeframe Momentum Divergence
    data['mid_price'] = (data['high'] + data['low']) / 2
    data['mid_return_5d'] = (data['mid_price'] - data['mid_price'].shift(5)) / data['mid_price'].shift(5)
    data['mid_return_10d'] = (data['mid_price'] - data['mid_price'].shift(10)) / data['mid_price'].shift(10)
    
    data['volume_return_5d'] = (data['volume'] - data['volume'].shift(5)) / data['volume'].shift(5)
    data['volume_return_10d'] = (data['volume'] - data['volume'].shift(10)) / data['volume'].shift(10)
    
    data['short_divergence'] = np.sign(data['mid_return_5d'] - data['volume_return_5d'])
    data['medium_divergence'] = np.sign(data['mid_return_10d'] - data['volume_return_10d'])
    
    momentum_strength = (abs(data['mid_return_5d']) + abs(data['mid_return_10d'])) / 2
    total_divergence = data['short_divergence'] + data['medium_divergence']
    momentum_signal = total_divergence * momentum_strength
    
    # Breakout Efficiency-Pressure Analysis
    data['high_5d_max'] = data['high'].rolling(window=5, min_periods=1).max().shift(1)
    data['low_5d_min'] = data['low'].rolling(window=5, min_periods=1).min().shift(1)
    
    breakout_magnitude = np.maximum(0, data['high'] - data['high_5d_max']) + np.maximum(0, data['low_5d_min'] - data['low'])
    efficiency_ratio = breakout_magnitude / data['true_range']
    
    upside_potential = (data['high'] - data['high_5d_max']) / data['true_range']
    downside_potential = (data['low_5d_min'] - data['low']) / data['true_range']
    pressure_imbalance = upside_potential - downside_potential
    
    # Breakout Signal
    breakout_signal = np.zeros(len(data))
    breakout_conditions = [
        (efficiency_ratio > 0.5) & (pressure_imbalance > 0.2),
        (efficiency_ratio > 0.5) & (pressure_imbalance < -0.2),
        (efficiency_ratio > 0.3) & (pressure_imbalance > 0.1),
        (efficiency_ratio > 0.3) & (pressure_imbalance < -0.1)
    ]
    breakout_choices = [2, -2, 1, -1]
    breakout_signal = np.select(breakout_conditions, breakout_choices, default=0)
    
    # Fractal Efficiency-Volume Confluence
    # Price Fractality (simplified Hurst approximation)
    def hurst_approximation(series, window=10):
        hurst_values = []
        for i in range(len(series)):
            if i < window:
                hurst_values.append(0.5)
                continue
            window_data = series.iloc[i-window+1:i+1]
            if len(window_data) < 2:
                hurst_values.append(0.5)
                continue
            returns = window_data.pct_change().dropna()
            if len(returns) < 2:
                hurst_values.append(0.5)
                continue
            # Simplified Hurst estimation using variance scaling
            var_ratio = returns.var() / (returns.diff().dropna().var() + 1e-8)
            hurst_est = 0.5 + 0.5 * np.log1p(var_ratio) / np.log(2)
            hurst_values.append(np.clip(hurst_est, 0.1, 0.9))
        return pd.Series(hurst_values, index=series.index)
    
    data['price_hurst'] = hurst_approximation(data['close'], window=10)
    
    # Volume Fractality (autocorrelation-based)
    def volume_fractality(volume_series, window=5):
        fractality = []
        for i in range(len(volume_series)):
            if i < window:
                fractality.append(0)
                continue
            window_data = volume_series.iloc[i-window+1:i+1]
            if len(window_data) < 2:
                fractality.append(0)
                continue
            # Simplified volume clustering measure
            autocorr = window_data.autocorr(lag=1)
            if pd.isna(autocorr):
                fractality.append(0)
            else:
                fractality.append(abs(autocorr))
        return pd.Series(fractality, index=volume_series.index)
    
    data['volume_fractality'] = volume_fractality(data['volume'], window=5)
    
    # Confluence Signal
    confluence_signal = np.zeros(len(data))
    confluence_conditions = [
        (data['price_hurst'] > 0.6) & (data['volume_fractality'] < 0.3),
        (data['price_hurst'] < 0.4) & (data['volume_fractality'] > 0.5)
    ]
    confluence_choices = [-1, 1]
    confluence_signal = np.select(confluence_conditions, confluence_choices, default=0)
    
    # Composite Alpha Factor Generation
    # Combine components with weights
    alignment_weighted = alignment_signal * 0.3
    momentum_weighted = momentum_signal * 0.4
    breakout_weighted = breakout_signal * 0.2
    confluence_weighted = confluence_signal * 0.1
    
    # Apply volatility persistence as risk adjustment
    vol_adjustment = 1 / (1 + data['true_range'].rolling(window=5).std())
    
    # Final composite factor
    composite_factor = (alignment_weighted + momentum_weighted + breakout_weighted + confluence_weighted) * vol_adjustment
    
    # Normalize the factor
    composite_factor = (composite_factor - composite_factor.rolling(window=20, min_periods=1).mean()) / (composite_factor.rolling(window=20, min_periods=1).std() + 1e-8)
    
    return pd.Series(composite_factor, index=data.index)
