import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

def heuristics_v2(df):
    data = df.copy()
    
    # Multi-Timeframe Efficiency Dynamics
    # Ultra-Short Efficiency (2-day)
    data['true_range_2d'] = data['high'].rolling(window=2).max() - data['low'].rolling(window=2).min()
    data['price_movement_2d'] = abs(data['close'] - data['close'].shift(2))
    data['efficiency_2d'] = data['price_movement_2d'] / data['true_range_2d']
    
    # Short-Term Efficiency (5-day)
    data['true_range_5d'] = data['high'].rolling(window=5).max() - data['low'].rolling(window=5).min()
    data['price_movement_5d'] = abs(data['close'] - data['close'].shift(5))
    data['efficiency_5d'] = data['price_movement_5d'] / data['true_range_5d']
    
    # Medium-Term Efficiency (20-day)
    data['true_range_20d'] = data['high'].rolling(window=20).max() - data['low'].rolling(window=20).min()
    data['price_movement_20d'] = abs(data['close'] - data['close'].shift(20))
    data['efficiency_20d'] = data['price_movement_20d'] / data['true_range_20d']
    
    # Efficiency Acceleration System
    data['accel_ultra_short'] = data['efficiency_5d'] - data['efficiency_2d']
    data['accel_short_medium'] = data['efficiency_20d'] - data['efficiency_5d']
    data['accel_pattern'] = np.sign(data['accel_ultra_short']) * np.sign(data['accel_short_medium'])
    
    # Volatility Regime Classification
    data['vol_short'] = data['close'].rolling(window=5).std()
    data['vol_medium'] = data['close'].rolling(window=20).std()
    data['regime_type'] = np.sign(data['vol_short'] - data['vol_medium'])
    
    # Intraday Volatility Asymmetry
    data['upside_vol_eff'] = (data['high'] - data['open']) / (data['high'] - data['low']).replace(0, np.nan)
    data['downside_vol_eff'] = (data['open'] - data['low']) / (data['high'] - data['low']).replace(0, np.nan)
    data['vol_skew_ratio'] = data['upside_vol_eff'] - data['downside_vol_eff']
    
    # Volatility Compression Patterns
    data['range_compression'] = (data['high'] - data['low']) / (data['high'].shift(4) - data['low'].shift(4)).replace(0, np.nan)
    
    # Calculate compression duration
    compression_mask = data['range_compression'] < 0.8
    data['compression_duration'] = compression_mask.astype(int).groupby((~compression_mask).cumsum()).cumsum()
    data['compression_exhaustion'] = data['compression_duration'] * data['range_compression']
    
    # Microstructure Pressure Dynamics
    data['upward_pressure'] = (data['high'] - data['open']) / (data['close'] - data['low']).replace(0, np.nan)
    data['downward_pressure'] = (data['open'] - data['low']) / (data['high'] - data['close']).replace(0, np.nan)
    data['pressure_ratio'] = data['upward_pressure'] / data['downward_pressure'].replace(0, np.nan)
    
    # Opening Auction Dynamics
    data['opening_gap_momentum'] = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
    data['auction_imbalance'] = (data['open'] - data['low']) - (data['high'] - data['open'])
    data['opening_efficiency'] = abs(data['close'] - data['open']) / abs(data['open'] - data['close'].shift(1)).replace(0, np.nan)
    
    # Volume Concentration Analysis
    data['peak_volume'] = data['volume'].rolling(window=5).max()
    data['avg_volume_5d'] = data['volume'].rolling(window=5).mean()
    data['volume_concentration'] = data['peak_volume'] / data['avg_volume_5d'].replace(0, np.nan)
    
    # Volume persistence calculation
    volume_increase = (data['volume'] > data['volume'].shift(1)).astype(int)
    data['volume_persistence'] = volume_increase.rolling(window=4).sum()
    data['concentration_signal'] = data['volume_concentration'] * data['volume_persistence']
    
    # Price-Volume Cointegration Analysis
    def calculate_cointegration_residuals(window_data):
        if len(window_data) < 2:
            return np.nan
        X = window_data['volume'].values.reshape(-1, 1)
        y = window_data['close'].values
        model = LinearRegression()
        model.fit(X, y)
        return y[-1] - model.predict(X[-1].reshape(1, -1))[0]
    
    # Calculate rolling cointegration residuals
    residuals = []
    for i in range(len(data)):
        if i >= 19:
            window_data = data.iloc[i-19:i+1][['close', 'volume']].copy()
            residual = calculate_cointegration_residuals(window_data)
            residuals.append(residual)
        else:
            residuals.append(np.nan)
    
    data['cointegration_residual'] = residuals
    data['residual_momentum'] = data['cointegration_residual'] - data['cointegration_residual'].shift(1)
    
    # Deviation Amplitude Assessment
    data['historical_residual_range'] = data['cointegration_residual'].rolling(window=20, min_periods=1).apply(
        lambda x: x.max() - x.min() if len(x) > 1 else np.nan, raw=True
    )
    data['current_deviation'] = abs(data['cointegration_residual'] - data['cointegration_residual'].rolling(window=20, min_periods=1).mean())
    data['normalized_deviation'] = data['current_deviation'] / data['historical_residual_range'].replace(0, np.nan)
    
    # Mean Reversion Potential
    data['residual_quantile_20'] = data['cointegration_residual'].rolling(window=20, min_periods=1).quantile(0.2)
    data['residual_quantile_80'] = data['cointegration_residual'].rolling(window=20, min_periods=1).quantile(0.8)
    
    oversold_condition = (data['cointegration_residual'] < data['residual_quantile_20']) & (data['residual_momentum'] > 0)
    overbought_condition = (data['cointegration_residual'] > data['residual_quantile_80']) & (data['residual_momentum'] < 0)
    
    data['cointegration_signal'] = np.sign(data['residual_momentum']) * (1 - data['normalized_deviation'])
    
    # Cross-Timeframe Momentum Alignment
    data['ultra_short_return'] = data['close'] / data['close'].shift(2) - 1
    data['short_term_return'] = data['close'] / data['close'].shift(5) - 1
    data['medium_term_return'] = data['close'] / data['close'].shift(20) - 1
    data['return_alignment'] = np.sign(data['ultra_short_return']) + np.sign(data['short_term_return']) + np.sign(data['medium_term_return'])
    
    # Momentum Quality Assessment
    return_signs = np.sign(data['close'].pct_change())
    return_consistency = (return_signs == return_signs.shift(1)).rolling(window=5).sum()
    data['return_consistency'] = return_consistency
    data['vol_adjusted_momentum'] = data['ultra_short_return'] / data['vol_short'].replace(0, np.nan)
    data['quality_score'] = data['return_consistency'] * data['vol_adjusted_momentum']
    
    # Breakout Validation System
    data['range_expansion'] = (data['high'] - data['low']) / (data['high'].rolling(window=4).mean() - data['low'].rolling(window=4).mean()).replace(0, np.nan)
    data['close_strength'] = (data['close'] - data['open']) / (data['high'] - data['low']).replace(0, np.nan)
    data['validated_breakout'] = data['range_expansion'] * data['close_strength'] * np.sign(data['ultra_short_return'])
    
    # Volume-Amount Flow Dynamics
    up_volume = np.where(data['close'] > data['open'], data['volume'], 0)
    down_volume = np.where(data['close'] < data['open'], data['volume'], 0)
    
    data['up_volume_5d'] = pd.Series(up_volume).rolling(window=5).sum()
    data['down_volume_5d'] = pd.Series(down_volume).rolling(window=5).sum()
    data['volume_asymmetry_ratio'] = data['up_volume_5d'] / data['down_volume_5d'].replace(0, np.nan)
    
    # Amount Efficiency Gradient
    data['daily_efficiency'] = abs(data['close'] - data['close'].shift(1)) / data['amount'].replace(0, np.nan)
    data['efficiency_trend'] = data['daily_efficiency'] - data['daily_efficiency'].rolling(window=5).mean().shift(1)
    data['efficiency_momentum'] = np.sign(data['efficiency_trend']) * abs(data['efficiency_trend'])
    
    # Order Flow Microstructure
    data['amount_efficiency'] = data['amount'] / (data['volume'] * data['close']).replace(0, np.nan)
    data['flow_concentration'] = data['amount'] / (data['amount'].shift(2) + data['amount'].shift(1)).replace(0, np.nan)
    data['microstructure_liquidity'] = data['amount_efficiency'] * data['flow_concentration']
    
    # Regime-Adaptive Signal Construction
    # Core Efficiency-Cointegration Signal
    data['base_signal'] = data['accel_pattern'] * data['cointegration_signal']
    data['volume_enhanced'] = data['base_signal'] * data['concentration_signal']
    data['microstructure_boosted'] = data['volume_enhanced'] * data['pressure_ratio']
    
    # Volatility Regime Integration
    high_vol_regime = data['microstructure_boosted'] * data['vol_skew_ratio']
    low_vol_regime = data['microstructure_boosted'] * data['compression_exhaustion']
    regime_transition = data['microstructure_boosted'] * np.sign(data['regime_type'] - data['regime_type'].shift(1))
    
    # Combine regime signals
    data['regime_weighted_signal'] = np.where(
        data['vol_short'] > data['vol_medium'], 
        high_vol_regime, 
        np.where(
            data['vol_short'] < data['vol_medium'] * 0.7, 
            low_vol_regime, 
            regime_transition
        )
    )
    
    # Volume-Efficiency Enhancement
    data['asymmetric_volume'] = data['regime_weighted_signal'] * data['volume_asymmetry_ratio']
    data['amount_flow'] = data['asymmetric_volume'] * data['efficiency_momentum']
    data['liquidity_adjusted'] = data['amount_flow'] * data['microstructure_liquidity']
    
    # Momentum Confirmation Layer
    data['quality_filtered'] = data['liquidity_adjusted'] * data['quality_score']
    data['breakout_verified'] = data['quality_filtered'] * data['validated_breakout']
    data['cross_timeframe_aligned'] = data['breakout_verified'] * (1 + data['return_alignment'] / 3)
    
    # Final Alpha Factor
    alpha_factor = data['cross_timeframe_aligned']
    
    return alpha_factor
