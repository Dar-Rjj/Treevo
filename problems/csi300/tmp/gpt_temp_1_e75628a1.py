import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Copy dataframe to avoid modifying original
    data = df.copy()
    
    # Calculate returns
    data['returns'] = data['close'].pct_change()
    
    # Multi-Timeframe Momentum
    data['momentum_5d'] = data['close'].pct_change(5)
    data['momentum_10d'] = data['close'].pct_change(10)
    data['momentum_divergence'] = data['momentum_5d'] - data['momentum_10d']
    
    # Volatility Regime
    data['volatility_20d'] = data['returns'].rolling(window=20).std()
    volatility_threshold = data['volatility_20d'].rolling(window=50).median()
    data['high_vol_regime'] = (data['volatility_20d'] > volatility_threshold).astype(int)
    
    # Volume-Validated Signal
    data['volume_ma_5'] = data['volume'].rolling(window=5).mean()
    data['volume_ma_10'] = data['volume'].rolling(window=10).mean()
    data['volume_trend'] = (data['volume_ma_5'] - data['volume_ma_10']) / data['volume_ma_10']
    data['volume_confirmation'] = np.sign(data['momentum_divergence']) * np.sign(data['volume_trend'])
    
    # Volatility-Regime Adjusted Momentum Signal
    data['regime_momentum_signal'] = np.where(
        data['high_vol_regime'] == 1,
        -data['momentum_divergence'],  # Reversal in high volatility
        data['momentum_divergence']    # Continuation in low volatility
    )
    
    # Apply volume confirmation
    data['final_momentum_signal'] = data['regime_momentum_signal'] * (1 + 0.5 * data['volume_confirmation'])
    
    # Range Breakout Efficiency
    data['prev_close'] = data['close'].shift(1)
    data['true_range'] = np.maximum(
        data['high'] - data['low'],
        np.maximum(
            abs(data['high'] - data['prev_close']),
            abs(data['low'] - data['prev_close'])
        )
    )
    
    # Breakout detection
    data['new_high'] = (data['high'] > data['high'].rolling(window=20).max()).astype(int)
    data['new_low'] = (data['low'] < data['low'].rolling(window=20).min()).astype(int)
    data['breakout_magnitude'] = np.where(
        data['new_high'] == 1, 
        data['high'] - data['high'].rolling(window=20).max(),
        np.where(
            data['new_low'] == 1,
            data['low'].rolling(window=20).min() - data['low'],
            0
        )
    )
    data['efficiency_ratio'] = data['breakout_magnitude'] / data['true_range']
    data['efficiency_ratio'] = data['efficiency_ratio'].replace([np.inf, -np.inf], 0).fillna(0)
    
    # Intraday Pressure
    data['buying_pressure'] = (data['close'] - data['low']) / (data['high'] - data['low'] + 1e-8)
    data['selling_pressure'] = (data['high'] - data['close']) / (data['high'] - data['low'] + 1e-8)
    data['net_pressure'] = data['buying_pressure'] - data['selling_pressure']
    
    # Accumulated pressure with reset logic
    data['pressure_cumsum'] = 0.0
    pressure_reset_threshold = 0.8
    
    for i in range(1, len(data)):
        if abs(data['net_pressure'].iloc[i]) > pressure_reset_threshold:
            data.loc[data.index[i], 'pressure_cumsum'] = data['net_pressure'].iloc[i]
        else:
            data.loc[data.index[i], 'pressure_cumsum'] = (
                data['pressure_cumsum'].iloc[i-1] + data['net_pressure'].iloc[i]
            )
    
    # Combine Breakout and Pressure
    data['breakout_pressure_signal'] = data['efficiency_ratio'] * data['pressure_cumsum']
    
    # Scale by volatility regime
    data['scaled_breakout_signal'] = data['breakout_pressure_signal'] / (data['volatility_20d'] + 1e-8)
    
    # Mean Reversion Setup
    data['ma_20'] = data['close'].rolling(window=20).mean()
    data['price_deviation'] = (data['close'] - data['ma_20']) / data['ma_20']
    data['opening_gap'] = (data['open'] - data['prev_close']) / data['prev_close']
    
    # Extreme deviations (z-score like)
    data['deviation_zscore'] = (
        data['price_deviation'] - data['price_deviation'].rolling(window=50).mean()
    ) / (data['price_deviation'].rolling(window=50).std() + 1e-8)
    
    # Liquidity Conditions
    data['volume_volatility'] = data['volume'].rolling(window=10).std()
    data['volume_mean'] = data['volume'].rolling(window=20).mean()
    data['turnover_rate'] = data['volume'] / data['volume_mean']
    
    # Liquidity classification
    liquidity_threshold = data['turnover_rate'].rolling(window=50).median()
    data['high_liquidity'] = (data['turnover_rate'] > liquidity_threshold).astype(int)
    
    # Gap behavior analysis
    data['gap_persistence'] = (np.sign(data['opening_gap']) == np.sign(data['opening_gap'].shift(1))).astype(int)
    data['gap_persistence_count'] = data['gap_persistence'].rolling(window=5).sum()
    
    # Historical gap mean reversion tendency
    data['gap_reversion'] = np.sign(data['opening_gap']) != np.sign(data['returns'])
    data['gap_reversion_rate'] = data['gap_reversion'].rolling(window=20).mean()
    
    # Liquidity-weighted mean reversion signal
    data['mean_reversion_signal'] = -data['deviation_zscore'] * (1 + data['high_liquidity'])
    data['gap_fade_signal'] = -data['opening_gap'] * data['gap_reversion_rate'] * (1 + data['high_liquidity'])
    data['combined_reversion_signal'] = data['mean_reversion_signal'] + data['gap_fade_signal']
    
    # Trend Persistence
    data['direction'] = np.where(data['close'] > data['prev_close'], 1, -1)
    data['consecutive_direction'] = 0
    
    for i in range(1, len(data)):
        if data['direction'].iloc[i] == data['direction'].iloc[i-1]:
            data.loc[data.index[i], 'consecutive_direction'] = data['consecutive_direction'].iloc[i-1] + data['direction'].iloc[i]
        else:
            data.loc[data.index[i], 'consecutive_direction'] = data['direction'].iloc[i]
    
    # Price movement amplitude
    data['daily_range'] = (data['high'] - data['low']) / data['prev_close']
    data['avg_range'] = data['daily_range'].rolling(window=20).mean()
    data['range_amplitude'] = data['daily_range'] / data['avg_range']
    
    # Trend acceleration
    data['ma_10'] = data['close'].rolling(window=10).mean()
    data['ma_10_prev'] = data['ma_10'].shift(1)
    data['ma_10_prev2'] = data['ma_10'].shift(2)
    data['trend_acceleration'] = (data['ma_10'] - data['ma_10_prev']) - (data['ma_10_prev'] - data['ma_10_prev2'])
    
    # Volume patterns
    data['volume_clustering'] = data['volume'].rolling(window=5).std() / data['volume'].rolling(window=20).std()
    data['volume_breakout'] = (data['volume'] > data['volume'].rolling(window=20).mean() * 1.5).astype(int)
    data['volume_price_correlation'] = data['volume'].rolling(window=10).corr(data['close'])
    
    # Combine trend and volume signals
    data['trend_confirmation'] = data['consecutive_direction'] * data['volume_clustering']
    data['momentum_signal'] = data['trend_acceleration'] * data['volume_breakout']
    data['weak_trend_warning'] = -data['range_amplitude'] * (1 - data['volume_clustering'])
    
    data['combined_trend_signal'] = (
        data['trend_confirmation'] + 
        data['momentum_signal'] + 
        data['weak_trend_warning']
    )
    
    # Price Efficiency
    data['price_efficiency'] = (data['close'] - data['open']) / (data['true_range'] + 1e-8)
    data['slippage_measure'] = data['true_range'] / (abs(data['close'] - data['prev_close']) + 1e-8)
    
    # Relative Strength (using rolling sector proxy - here using market as proxy)
    data['sector_momentum'] = data['close'].pct_change(10).rolling(window=50).mean()  # Proxy for sector
    data['relative_performance'] = data['momentum_10d'] - data['sector_momentum']
    
    # Volume-weighted validation
    data['volume_intensity'] = data['volume'] / data['volume'].rolling(window=20).mean()
    data['efficiency_signal'] = data['price_efficiency'] * data['relative_performance'] * data['volume_intensity']
    
    # Final composite factor
    weights = {
        'momentum': 0.25,
        'breakout': 0.20,
        'reversion': 0.25,
        'trend': 0.15,
        'efficiency': 0.15
    }
    
    data['composite_factor'] = (
        weights['momentum'] * data['final_momentum_signal'] +
        weights['breakout'] * data['scaled_breakout_signal'] +
        weights['reversion'] * data['combined_reversion_signal'] +
        weights['trend'] * data['combined_trend_signal'] +
        weights['efficiency'] * data['efficiency_signal']
    )
    
    # Normalize final factor
    factor_mean = data['composite_factor'].rolling(window=50).mean()
    factor_std = data['composite_factor'].rolling(window=50).std()
    data['final_factor'] = (data['composite_factor'] - factor_mean) / (factor_std + 1e-8)
    
    return data['final_factor']
