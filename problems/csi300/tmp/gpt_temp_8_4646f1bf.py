import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Volatility-Regime Adaptive Momentum-Volume Convergence factor
    """
    # Make a copy to avoid modifying original dataframe
    data = df.copy()
    
    # Volatility Regime Classification
    # Calculate Average True Range (ATR)
    data['prev_close'] = data['close'].shift(1)
    data['tr1'] = data['high'] - data['low']
    data['tr2'] = abs(data['high'] - data['prev_close'])
    data['tr3'] = abs(data['low'] - data['prev_close'])
    data['true_range'] = data[['tr1', 'tr2', 'tr3']].max(axis=1)
    data['atr_20'] = data['true_range'].rolling(window=20, min_periods=10).mean()
    
    # Classify volatility regime
    data['atr_percentile_20'] = data['atr_20'].rolling(window=20, min_periods=10).apply(
        lambda x: (x.iloc[-1] > np.percentile(x.dropna(), 80)) if len(x.dropna()) >= 10 else np.nan
    )
    data['atr_percentile_80'] = data['atr_20'].rolling(window=20, min_periods=10).apply(
        lambda x: (x.iloc[-1] < np.percentile(x.dropna(), 20)) if len(x.dropna()) >= 10 else np.nan
    )
    
    data['high_vol_regime'] = data['atr_percentile_20'].fillna(0)
    data['low_vol_regime'] = data['atr_percentile_80'].fillna(0)
    
    # Multi-Timeframe Momentum Analysis
    data['momentum_5'] = data['close'] / data['close'].shift(5) - 1
    data['momentum_20'] = data['close'] / data['close'].shift(20) - 1
    
    # Momentum Divergence Detection
    data['momentum_sign_5'] = np.sign(data['momentum_5'])
    data['momentum_sign_20'] = np.sign(data['momentum_20'])
    data['momentum_consistency'] = (data['momentum_sign_5'] == data['momentum_sign_20']).astype(float)
    
    # Volume Confirmation Framework
    data['volume_5_avg'] = data['volume'].rolling(window=5, min_periods=3).mean()
    data['volume_20_avg'] = data['volume'].rolling(window=20, min_periods=10).mean()
    data['volume_ratio'] = data['volume_5_avg'] / data['volume_20_avg']
    
    # Volume-Momentum Alignment
    data['volume_trend'] = np.sign(data['volume_ratio'] - 1)
    data['momentum_alignment'] = (data['volume_trend'] == data['momentum_sign_5']).astype(float)
    
    # Volume Breakout Signal
    data['volume_intensity'] = data['volume'] / data['volume_20_avg']
    data['volume_breakout'] = (data['volume_intensity'] > 1.5).astype(float)
    
    # Adaptive Signal Combination
    # High Volatility Regime signals
    high_vol_signal = (
        0.5 * data['momentum_5'].fillna(0) +
        0.3 * data['volume_breakout'].fillna(0) +
        0.2 * (1 - data['momentum_consistency']).fillna(0)  # Emphasize divergence in high vol
    )
    
    # Low Volatility Regime signals
    low_vol_signal = (
        0.4 * data['momentum_20'].fillna(0) +
        0.4 * data['momentum_alignment'].fillna(0) +
        0.2 * data['momentum_consistency'].fillna(0)
    )
    
    # Combine regimes
    data['adaptive_signal'] = (
        data['high_vol_regime'] * high_vol_signal +
        data['low_vol_regime'] * low_vol_signal +
        (1 - data['high_vol_regime'] - data['low_vol_regime']) * 0.5 * (high_vol_signal + low_vol_signal)
    )
    
    # Price-Level Breakout Efficiency
    data['resistance_20'] = data['high'].rolling(window=20, min_periods=10).max()
    data['support_20'] = data['low'].rolling(window=20, min_periods=10).min()
    data['range_utilization'] = (data['close'] - data['support_20']) / (data['resistance_20'] - data['support_20']).replace(0, np.nan)
    
    # Breakout Quality Assessment
    data['breakout_magnitude'] = np.where(
        data['close'] > data['resistance_20'],
        (data['close'] - data['resistance_20']) / data['resistance_20'],
        np.where(
            data['close'] < data['support_20'],
            (data['support_20'] - data['close']) / data['support_20'],
            0
        )
    )
    
    # Liquidity-Enhanced Validation
    data['volume_10_avg'] = data['volume'].rolling(window=10, min_periods=5).mean()
    data['liquidity_ratio'] = data['volume'] / data['volume_10_avg']
    data['price_efficiency'] = data['amount'] / (data['volume'] * data['close']).replace(0, np.nan)
    
    # Adaptive breakout weighting
    liquidity_weight = np.where(
        data['liquidity_ratio'] > 1.2, 1.5,
        np.where(data['liquidity_ratio'] < 0.8, 0.5, 1.0)
    )
    
    data['breakout_signal'] = data['breakout_magnitude'] * liquidity_weight * data['price_efficiency'].fillna(1)
    
    # Intraday Pressure Accumulation
    data['overnight_gap'] = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
    
    # Buying/Selling Pressure Index
    data['pressure_index'] = (2 * data['close'] - data['low'] - data['high']) / (data['high'] - data['low']).replace(0, np.nan)
    
    # Pressure Accumulation with exponential decay
    decay_weights = np.array([0.5, 0.3, 0.15, 0.04, 0.01])  # 5-day exponential decay
    data['pressure_accumulation'] = data['pressure_index'].rolling(window=5, min_periods=3).apply(
        lambda x: np.dot(x.values, decay_weights[:len(x)]) if len(x) >= 3 else np.nan
    )
    
    # Volume-Weighted Execution
    data['intraday_pressure'] = data['pressure_accumulation'] * data['volume_intensity'].fillna(1)
    
    # Momentum-Volume-Liquidity Triple Convergence
    # Multi-timeframe momentum
    data['momentum_60'] = data['close'] / data['close'].shift(60) - 1
    
    # Momentum consistency across periods
    momentum_signs = np.sign(data[['momentum_5', 'momentum_20', 'momentum_60']])
    data['momentum_unanimity'] = (momentum_signs.nunique(axis=1) == 1).astype(float)
    
    # Volatility-adjusted momentum
    data['volatility_20'] = data['close'].pct_change().rolling(window=20, min_periods=10).std()
    data['risk_adj_momentum'] = data['momentum_20'] / data['volatility_20'].replace(0, np.nan)
    
    # Volume confirmation
    volume_trends = np.sign(data[['volume_5_avg', 'volume_20_avg']].pct_change(periods=5).fillna(0))
    data['volume_trend_alignment'] = (volume_trends['volume_5_avg'] == volume_trends['volume_20_avg']).astype(float)
    
    # Volume-momentum divergence
    data['volume_momentum_divergence'] = (np.sign(data['momentum_5']) != np.sign(data['volume_ratio'] - 1)).astype(float)
    
    # Liquidity measures
    data['dollar_volume_efficiency'] = data['amount'] / (data['volume'] * data['close']).replace(0, np.nan)
    data['price_impact'] = (data['high'] - data['low']) / data['close'] / data['volume'].replace(0, np.nan)
    
    # Triple Convergence Signal
    momentum_component = (
        0.4 * data['risk_adj_momentum'].fillna(0) +
        0.3 * data['momentum_unanimity'].fillna(0) +
        0.3 * (1 - data['volume_momentum_divergence']).fillna(0)
    )
    
    volume_component = (
        0.5 * data['volume_trend_alignment'].fillna(0) +
        0.5 * data['volume_breakout'].fillna(0)
    )
    
    liquidity_component = (
        0.6 * data['dollar_volume_efficiency'].fillna(0) +
        0.4 * (1 / data['price_impact'].replace(0, np.nan)).fillna(0)
    )
    
    data['triple_convergence'] = (
        0.5 * momentum_component +
        0.3 * volume_component +
        0.2 * liquidity_component
    )
    
    # Final composite alpha factor
    data['alpha_factor'] = (
        0.4 * data['adaptive_signal'].fillna(0) +
        0.3 * data['breakout_signal'].fillna(0) +
        0.2 * data['intraday_pressure'].fillna(0) +
        0.1 * data['triple_convergence'].fillna(0)
    )
    
    # Normalize the final factor
    data['alpha_factor_normalized'] = (
        data['alpha_factor'] - data['alpha_factor'].rolling(window=60, min_periods=20).mean()
    ) / data['alpha_factor'].rolling(window=60, min_periods=20).std()
    
    return data['alpha_factor_normalized']
