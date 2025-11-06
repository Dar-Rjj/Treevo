import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Multi-Dimensional Volatility-Adaptive Microstructure Momentum Alpha
    """
    data = df.copy()
    
    # Volatility-Regime Classification
    # Short-term Volatility: 5-day High-Low range mean
    data['high_low_range'] = data['high'] - data['low']
    data['short_term_vol'] = data['high_low_range'].rolling(window=5, min_periods=3).mean()
    
    # Medium-term Volatility: 10-day return standard deviation
    data['returns'] = data['close'].pct_change()
    data['medium_term_vol'] = data['returns'].rolling(window=10, min_periods=5).std()
    
    # Long-term Volatility: 20-day Open-Close spread mean
    data['open_close_spread'] = abs(data['close'] - data['open'])
    data['long_term_vol'] = data['open_close_spread'].rolling(window=20, min_periods=10).mean()
    
    # Bidirectional Order Flow Efficiency
    # Active Buying Efficiency
    high_low_diff = data['high'] - data['low']
    high_low_diff = high_low_diff.replace(0, np.nan)  # Avoid division by zero
    data['buying_efficiency'] = ((data['close'] - data['low']) / high_low_diff) * data['volume']
    
    # Active Selling Efficiency
    data['selling_efficiency'] = ((data['high'] - data['close']) / high_low_diff) * data['volume']
    
    # Net Pressure Efficiency Ratio
    data['net_pressure_ratio'] = (data['buying_efficiency'] - data['selling_efficiency']) / (
        data['buying_efficiency'] + data['selling_efficiency'] + 1e-8)
    
    # Microstructure Noise and Signal Separation
    # Intraday whipsaw detection
    data['prev_high'] = data['high'].shift(1)
    data['prev_low'] = data['low'].shift(1)
    data['whipsaw'] = ((data['high'] > data['prev_high']) & (data['low'] < data['prev_low'])).astype(int)
    data['whipsaw_count'] = data['whipsaw'].rolling(window=5, min_periods=3).sum()
    
    # Micro-reversal magnitude
    data['mid_price'] = (data['high'] + data['low']) / 2
    data['micro_reversal'] = abs(data['close'] - data['mid_price']) / high_low_diff
    
    # Noise-to-efficiency ratio
    data['prev_close'] = data['close'].shift(1)
    data['overnight_gap'] = abs(data['open'] - data['prev_close'])
    data['noise_efficiency_ratio'] = data['high_low_range'] / (data['overnight_gap'] + 1e-8)
    
    # Volume-Microstructure Coordination
    # Volume clustering persistence
    data['volume_20d_avg'] = data['volume'].rolling(window=20, min_periods=10).mean()
    data['high_volume'] = (data['volume'] > data['volume_20d_avg']).astype(int)
    
    # Count consecutive high volume days
    def count_consecutive_high_volume(series):
        count = 0
        result = []
        for val in series:
            if val == 1:
                count += 1
            else:
                count = 0
            result.append(count)
        return result
    
    data['volume_clustering'] = count_consecutive_high_volume(data['high_volume'])
    
    # Volume-pressure efficiency
    data['volume_pressure_efficiency'] = data['net_pressure_ratio'] * data['volume_clustering']
    
    # Microstructure-volume alignment (10-day correlation)
    data['volume_change'] = data['volume'].pct_change()
    data['micro_volume_corr'] = data['net_pressure_ratio'].rolling(window=10, min_periods=5).corr(data['volume_change'])
    
    # Multi-Timeframe Pressure Divergence
    # Short-term Pressure Patterns
    data['prev_open'] = data['open'].shift(1)
    data['gap_resolution'] = abs(data['close'] - data['open']) / (abs(data['open'] - data['prev_close']) + 1e-8)
    
    # Intraday reversal efficiency
    data['intraday_reversal'] = ((data['high'] > data['prev_high']) & (data['close'] < data['open'])).astype(int)
    data['intraday_reversal_count'] = data['intraday_reversal'].rolling(window=5, min_periods=3).sum()
    
    # Medium-term Pressure Momentum
    data['vol_weighted_pressure'] = (data['returns'] / (data['medium_term_vol'] + 1e-8)) * np.sign(data['returns'])
    data['order_flow_momentum'] = data['net_pressure_ratio'].diff(5)
    
    # Microstructure persistence (10-day autocorrelation)
    def rolling_autocorr(series, window):
        return series.rolling(window=window).apply(lambda x: x.autocorr(), raw=False)
    
    data['micro_persistence'] = rolling_autocorr(data['net_pressure_ratio'], 10)
    
    # Long-term Structural Pressure
    data['pressure_trend'] = data['net_pressure_ratio'].rolling(window=20, min_periods=10).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0, raw=False
    )
    
    # High-pressure regime detection
    data['high_volume_days'] = (data['volume'] > 1.5 * data['volume_20d_avg']).astype(int)
    data['high_pressure_regime'] = data['high_volume_days'].rolling(window=20, min_periods=10).sum()
    
    # Volatility-Adaptive Signal Generation
    # Define volatility regimes
    data['volatility_regime'] = 1  # Medium by default
    data.loc[data['short_term_vol'] > data['short_term_vol'].quantile(0.7), 'volatility_regime'] = 2  # High
    data.loc[data['short_term_vol'] < data['short_term_vol'].quantile(0.3), 'volatility_regime'] = 0  # Low
    
    # Cross-Dimensional Pressure Integration
    # Pressure-volume efficiency
    data['pressure_volume_efficiency'] = data['net_pressure_ratio'] * data['volume_clustering']
    
    # Noise-to-pressure ratio
    data['noise_pressure_ratio'] = data['micro_reversal'] / (abs(data['net_pressure_ratio']) + 1e-8)
    
    # Multi-timeframe pressure confirmation
    data['short_term_pressure'] = data['net_pressure_ratio'].rolling(window=5, min_periods=3).mean()
    data['medium_term_pressure'] = data['net_pressure_ratio'].rolling(window=10, min_periods=5).mean()
    data['long_term_pressure'] = data['net_pressure_ratio'].rolling(window=20, min_periods=10).mean()
    
    data['pressure_confirmation'] = (
        (data['short_term_pressure'] > 0).astype(int) + 
        (data['medium_term_pressure'] > 0).astype(int) + 
        (data['long_term_pressure'] > 0).astype(int)
    )
    
    # Composite Microstructure Momentum Alpha
    # Core pressure components
    data['order_flow_composite'] = (
        data['buying_efficiency'].rolling(window=5, min_periods=3).mean() - 
        data['selling_efficiency'].rolling(window=5, min_periods=3).mean()
    )
    
    data['noise_filtered_pressure'] = data['net_pressure_ratio'] * (1 - data['noise_pressure_ratio'])
    
    # Regime-adaptive weighting
    regime_weights = {
        0: 0.3,  # Low volatility - reduced signal strength
        1: 1.0,  # Medium volatility - standard weighting
        2: 1.5   # High volatility - amplified signals
    }
    
    data['regime_weight'] = data['volatility_regime'].map(regime_weights)
    
    # Final alpha generation
    alpha = (
        # Core pressure momentum
        data['order_flow_composite'] * 0.3 +
        data['noise_filtered_pressure'] * 0.25 +
        data['vol_weighted_pressure'] * 0.2 +
        
        # Volume coordination
        data['pressure_volume_efficiency'] * 0.15 +
        
        # Multi-timeframe confirmation
        data['pressure_confirmation'] * data['micro_persistence'] * 0.1
    ) * data['regime_weight']
    
    # Clean up intermediate columns
    result = alpha.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    return result
