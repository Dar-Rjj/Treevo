import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Volatility Regime Classification
    # Range Volatility Ratio
    data['daily_range'] = data['high'] - data['low']
    data['range_vol_ratio'] = data['daily_range'].rolling(5).mean() / data['daily_range'].rolling(20).mean()
    
    # Price volatility (daily returns std)
    data['daily_ret'] = data['close'] / data['close'].shift(1) - 1
    data['price_vol'] = data['daily_ret'].rolling(10).std() * data['close']
    
    # Regime definition
    data['high_vol_regime'] = ((data['range_vol_ratio'] > 1.2) | 
                              (data['daily_range'] > data['price_vol'])).astype(int)
    data['low_vol_regime'] = ((data['range_vol_ratio'] <= 1.2) & 
                             (data['daily_range'] <= data['price_vol'])).astype(int)
    
    # Multi-Scale Price-Volume-Amount Divergence
    # Price Momentum Components
    data['ultra_short_price_mom'] = data['close'] / data['close'].shift(2) - 1
    data['short_price_mom'] = data['close'] / data['close'].shift(5) - 1
    data['medium_price_mom'] = data['close'] / data['close'].shift(15) - 1
    
    # Volume Momentum Components
    data['ultra_short_vol_mom'] = data['volume'] / data['volume'].shift(2) - 1
    data['short_vol_mom'] = data['volume'] / data['volume'].shift(5) - 1
    data['medium_vol_mom'] = data['volume'] / data['volume'].shift(15) - 1
    
    # Amount Momentum Components
    data['short_amount_mom'] = data['amount'] / data['amount'].shift(5) - 1
    data['medium_amount_mom'] = data['amount'] / data['amount'].shift(15) - 1
    
    # Divergence Gap Calculations
    data['price_mom_gap'] = (data['short_price_mom'] - data['medium_price_mom']) * np.sign(data['short_price_mom'])
    data['vol_mom_gap'] = (data['short_vol_mom'] - data['medium_vol_mom']) * np.sign(data['short_vol_mom'])
    data['amount_mom_gap'] = (data['short_amount_mom'] - data['medium_amount_mom']) * np.sign(data['short_amount_mom'])
    data['ultra_short_div'] = data['ultra_short_price_mom'] - data['ultra_short_vol_mom']
    
    # Volume Distribution Efficiency Analysis
    # Volume Concentration Patterns
    def volume_concentration(vol_series):
        if len(vol_series) < 10:
            return np.nan
        sorted_vol = np.sort(vol_series)[-3:]
        return np.sum(sorted_vol) / np.sum(vol_series)
    
    data['vol_concentration'] = data['volume'].rolling(10).apply(volume_concentration, raw=True)
    
    def volume_skew(vol_series):
        if len(vol_series) < 10:
            return np.nan
        p25, p50, p75 = np.percentile(vol_series, [25, 50, 75])
        return ((p75 - p50) - (p50 - p25)) / p50
    
    data['vol_skew'] = data['volume'].rolling(10).apply(volume_skew, raw=True)
    
    # Directional Volume Pressure
    data['upward_pressure_vol'] = data['volume'] * (data['close'] - data['low']) / (data['high'] - data['low']).replace(0, np.nan)
    data['downward_pressure_vol'] = data['volume'] * (data['high'] - data['close']) / (data['high'] - data['low']).replace(0, np.nan)
    data['net_pressure_vol'] = data['upward_pressure_vol'] - data['downward_pressure_vol']
    
    # Volume-Price Efficiency Metrics
    data['price_efficiency'] = np.abs(data['close'] - data['open']) / (data['high'] - data['low']).replace(0, np.nan)
    data['volume_impact_ratio'] = data['net_pressure_vol'] / data['volume'].replace(0, np.nan)
    data['vol_price_alignment'] = data['volume_impact_ratio'] * data['price_efficiency']
    
    # Regime-Adaptive Signal Construction
    # High Volatility Regime Signals
    data['core_div_high_vol'] = (data['price_mom_gap'] * data['vol_mom_gap'] * 
                                (1 + np.abs(data['amount_mom_gap'])))
    
    # Volume Asymmetry Multiplier
    def volume_asymmetry(data_window):
        up_volume = data_window[data_window['close'] > data_window['close'].shift(1)]['volume'].sum()
        down_volume = data_window[data_window['close'] < data_window['close'].shift(1)]['volume'].sum()
        return up_volume / down_volume if down_volume > 0 else 1.0
    
    # Calculate rolling volume asymmetry
    vol_asymmetry_values = []
    for i in range(len(data)):
        if i >= 5:
            window_data = data.iloc[i-4:i+1].copy()
            vol_asymmetry_values.append(volume_asymmetry(window_data))
        else:
            vol_asymmetry_values.append(1.0)
    
    data['vol_asymmetry_mult'] = vol_asymmetry_values
    data['range_adj_signal'] = (data['core_div_high_vol'] * data['vol_asymmetry_mult'] / 
                               data['daily_range'].replace(0, np.nan))
    
    # Low Volatility Regime Signals
    data['core_div_low_vol'] = (data['price_mom_gap'] * data['vol_mom_gap'] * 
                               (1 + data['amount_mom_gap']))
    data['efficiency_mult'] = data['price_efficiency'] * data['vol_price_alignment']
    data['concentration_adj_signal'] = (data['core_div_low_vol'] * data['efficiency_mult'] * 
                                       data['vol_concentration'])
    
    # True Range Integration and Filtering
    # Daily True Range
    data['true_range'] = np.maximum(
        data['high'] - data['low'],
        np.maximum(
            np.abs(data['high'] - data['close'].shift(1)),
            np.abs(data['low'] - data['close'].shift(1))
        )
    )
    
    # 5-day Average True Range
    data['avg_true_range'] = data['true_range'].rolling(5).mean()
    
    # Range-Adjusted Final Signals
    data['high_vol_final'] = data['range_adj_signal'] / data['avg_true_range'].replace(0, np.nan)
    data['low_vol_final'] = data['concentration_adj_signal'] / data['avg_true_range'].replace(0, np.nan)
    
    # Apply filters
    vol_skew_filter = data['vol_skew'] > 0
    concentration_filter = data['vol_concentration'] > 0.3
    pressure_filter = data['net_pressure_vol'] > 0
    
    # Composite Alpha Output
    # Regime-Selected Signal
    data['regime_signal'] = np.where(
        data['high_vol_regime'] == 1,
        data['high_vol_final'],
        data['low_vol_final']
    )
    
    # Apply filters based on regime
    regime_filter = np.where(
        data['high_vol_regime'] == 1,
        vol_skew_filter & concentration_filter,
        concentration_filter & pressure_filter
    )
    
    data['filtered_regime_signal'] = data['regime_signal'] * regime_filter
    
    # Ultra-Short Enhancement
    data['ultra_short_enhance'] = data['ultra_short_div'] * data['vol_concentration']
    
    # Final Factor
    data['final_factor'] = data['filtered_regime_signal'] * (1 + data['ultra_short_enhance'])
    
    return data['final_factor']
