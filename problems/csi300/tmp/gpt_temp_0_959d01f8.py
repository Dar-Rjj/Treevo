import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Calculate basic price and volume features
    df['hl_range'] = (df['high'] - df['low']) / df['close']
    df['returns'] = df['close'].pct_change()
    df['volume_change'] = df['volume'].pct_change()
    
    # Multi-Timeframe Order Flow Momentum
    # Short-term Reversal with Volume Divergence
    df['high_low_change_2d'] = (df['high'] - df['low']).rolling(window=2).mean()
    df['volume_trend_3d'] = df['volume'].rolling(window=3).apply(lambda x: (x[-1] - x[0]) / x[0] if x[0] != 0 else 0)
    df['short_term_reversal'] = -df['high_low_change_2d'] * df['volume_trend_3d']
    
    # Medium-term Momentum Alignment
    df['momentum_5d'] = df['close'].pct_change(periods=5)
    df['momentum_8d'] = df['close'].pct_change(periods=8)
    df['volume_trend_5d'] = df['volume'].rolling(window=5).apply(lambda x: (x[-1] - x[0]) / x[0] if x[0] != 0 else 0)
    df['medium_term_momentum'] = (df['momentum_5d'] + df['momentum_8d']) * df['volume_trend_5d']
    
    # Order Flow Consistency
    df['volume_acceleration'] = df['volume_change'].rolling(window=3).mean()
    df['order_flow_consistency'] = np.sign(df['short_term_reversal']) * np.sign(df['medium_term_momentum']) * df['volume_acceleration']
    
    # Liquidity Stress with Microstructure Efficiency
    # Volume Acceleration under Stress
    df['liquidity_stress_ratio'] = (df['hl_range'] * df['volume']).rolling(window=3).mean()
    df['volume_acceleration_stress'] = df['volume_change'] * df['liquidity_stress_ratio']
    
    # Volatility-Weighted Efficiency
    df['true_range'] = pd.concat([
        df['high'] - df['low'],
        abs(df['high'] - df['close'].shift(1)),
        abs(df['low'] - df['close'].shift(1))
    ], axis=1).max(axis=1)
    df['atr_5d'] = df['true_range'].rolling(window=5).mean()
    df['intraday_efficiency'] = (df['close'] - df['open']) / (df['high'] - df['low']).replace(0, np.nan)
    df['volatility_efficiency'] = df['atr_5d'] * df['intraday_efficiency'].fillna(0)
    
    # Trade Size Dynamics in Stress
    df['avg_trade_size'] = df['amount'] / df['volume'].replace(0, np.nan)
    df['trade_size_momentum'] = df['avg_trade_size'].pct_change(periods=3)
    df['trade_size_stress'] = df['trade_size_momentum'] * df['liquidity_stress_ratio']
    
    # Price-Volume Divergence Analysis
    # Microstructure Divergence Signals
    df['price_trend_3d'] = df['close'].pct_change(periods=3)
    df['volume_trend_3d_div'] = df['volume'].rolling(window=3).apply(lambda x: (x[-1] - x[0]) / x[0] if x[0] != 0 else 0)
    df['microstructure_divergence'] = df['volume_trend_3d_div'] - df['price_trend_3d']
    
    # Large Trade Activity Context
    median_volume = df['volume'].rolling(window=20).median()
    df['days_above_median'] = (df['volume'] > median_volume).astype(int).rolling(window=5).sum()
    df['large_trade_context'] = df['days_above_median'] * df['trade_size_momentum']
    
    # Efficiency-Weighted Divergence
    df['price_range_efficiency'] = abs(df['close'] - df['open']) / (df['high'] - df['low']).replace(0, np.nan)
    df['efficiency_weighted_div'] = df['price_range_efficiency'].fillna(0) * abs(df['microstructure_divergence'])
    
    # Composite Alpha Construction
    # Regime-Scaled Order Flow
    volatility_regime = df['atr_5d'].rolling(window=10).rank(pct=True)
    df['regime_scaled_order_flow'] = volatility_regime * df['order_flow_consistency'] * df['microstructure_divergence']
    
    # Stress-Filtered Microstructure
    df['stress_filtered_micro'] = df['liquidity_stress_ratio'] * df['price_range_efficiency'].fillna(0) * df['volatility_efficiency']
    
    # Multi-level Convergence
    df['multi_level_convergence'] = df['order_flow_consistency'] * df['efficiency_weighted_div'] * df['stress_filtered_micro']
    
    # Final composite alpha factor
    alpha = (df['regime_scaled_order_flow'] + 
             df['stress_filtered_micro'] + 
             df['multi_level_convergence']) / 3
    
    # Clean up and return
    alpha = alpha.replace([np.inf, -np.inf], np.nan).fillna(0)
    return alpha
