import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    data = df.copy()
    
    # Calculate True Range
    data['prev_close'] = data['close'].shift(1)
    data['tr1'] = data['high'] - data['low']
    data['tr2'] = abs(data['high'] - data['prev_close'])
    data['tr3'] = abs(data['low'] - data['prev_close'])
    data['true_range'] = data[['tr1', 'tr2', 'tr3']].max(axis=1)
    
    # 1. Dual-Window Efficiency Framework
    # Efficiency Regime Identification
    data['range_efficiency'] = data['true_range'] / data['true_range'].rolling(window=5).mean()
    
    # Price efficiency calculations
    data['price_change_3d'] = data['close'] - data['close'].shift(3)
    data['abs_daily_returns'] = abs(data['close'] - data['close'].shift(1))
    data['sum_abs_returns_3d'] = data['abs_daily_returns'].rolling(window=3).sum()
    data['price_efficiency'] = data['price_change_3d'] / data['sum_abs_returns_3d']
    
    # Efficiency momentum divergence
    data['price_change_8d'] = data['close'] - data['close'].shift(8)
    data['sum_abs_returns_8d'] = data['abs_daily_returns'].rolling(window=8).sum()
    data['efficiency_3d'] = data['price_change_3d'] / data['sum_abs_returns_3d']
    data['efficiency_8d'] = data['price_change_8d'] / data['sum_abs_returns_8d']
    data['efficiency_divergence'] = data['efficiency_3d'] - data['efficiency_8d']
    
    # 2. Volume Acceleration & Elasticity
    # Volume Dynamics
    data['volume_acceleration'] = (data['volume'] / data['volume'].shift(3)) - 1
    data['dollar_volume_velocity'] = data['amount'] / data['amount'].shift(3)
    data['volume_trend_ratio'] = data['volume'].rolling(window=10).mean() / data['volume'].rolling(window=20).mean()
    
    # Price Elasticity Analysis
    data['daily_range'] = data['high'] - data['low']
    data['avg_3d_range'] = data['daily_range'].rolling(window=3).mean()
    data['elasticity'] = (data['daily_range'] / data['avg_3d_range']) - 1
    
    # 3. Microstructure Resilience & Confirmation
    # Large Trade Analysis
    data['avg_trade_size'] = data['amount'] / data['volume']
    data['avg_amount_5d'] = data['amount'].rolling(window=5).mean()
    
    # Calculate large trade ratio
    large_trade_count = pd.Series(index=data.index, dtype=float)
    for i in range(len(data)):
        if i >= 3:
            window_data = data.iloc[i-3:i+1]
            count = (window_data['amount'] > window_data['avg_amount_5d']).sum()
            large_trade_count.iloc[i] = count / 3
        else:
            large_trade_count.iloc[i] = 0
    data['large_trade_ratio'] = large_trade_count
    
    data['price_impact_proxy'] = data['true_range'] / data['volume']
    
    # Market Microstructure Resilience
    data['intraday_return'] = data['close'] - data['open']
    data['total_range'] = data['high'] - data['low']
    
    # Price recovery (when close > open)
    recovery_mask = data['intraday_return'] > 0
    data['price_recovery'] = 0
    data.loc[recovery_mask, 'price_recovery'] = (data.loc[recovery_mask, 'close'] - data.loc[recovery_mask, 'low']) / data.loc[recovery_mask, 'total_range']
    
    # Downside absorption (when close < open)
    absorption_mask = data['intraday_return'] < 0
    data['downside_absorption'] = 0
    data.loc[absorption_mask, 'downside_absorption'] = (data.loc[absorption_mask, 'close'] - data.loc[absorption_mask, 'low']) / abs(data.loc[absorption_mask, 'intraday_return'])
    
    data['resilience_score'] = (data['price_recovery'] + data['downside_absorption']) / 2
    
    # 4. Efficiency-Adaptive Signal Construction
    # Elasticity-adjusted momentum
    data['elasticity_adj_momentum'] = data['efficiency_divergence'] * (1 + data['elasticity'])
    
    # Volume-accelerated signal
    data['volume_accel_signal'] = data['elasticity_adj_momentum'] * data['volume_acceleration']
    
    # Resilience-adjusted momentum
    data['resilience_adj_momentum'] = data['volume_accel_signal'] * data['resilience_score']
    
    # Large trade impact factor
    data['large_trade_adj_signal'] = data['resilience_adj_momentum'] * (1 - data['large_trade_ratio'])
    
    # 5. Dynamic Confirmation & Enhancement
    # Spread-Efficiency Divergence
    data['mid_price'] = (data['high'] + data['low']) / 2
    data['effective_spread'] = 2 * abs(data['close'] - data['mid_price']) / data['mid_price']
    data['avg_spread_5d'] = data['effective_spread'].rolling(window=5).mean()
    data['avg_spread_10d'] = data['effective_spread'].rolling(window=10).mean()
    data['liquidity_efficiency_div'] = data['efficiency_divergence'] * ((data['avg_spread_5d'] / data['avg_spread_10d']) - 1)
    
    # Volume Concentration Analysis
    data['volume_5d_avg'] = data['volume'].rolling(window=5).mean()
    data['amount_5d_avg'] = data['amount'].rolling(window=5).mean()
    data['net_concentration'] = (data['volume'] / data['volume_5d_avg']) - 1
    data['dollar_volume_concentration'] = (data['amount'] / data['amount_5d_avg']) - 1
    
    # Combined signal
    data['combined_signal'] = data['large_trade_adj_signal'] + data['liquidity_efficiency_div']
    
    # 6. Final Alpha Generation
    # Stability-Weighted Signal
    data['price_stability'] = 1 - (abs(data['close'] - data['close'].shift(1)) / data['close'].shift(1))
    
    # Signal consistency
    data['signal_consistency'] = np.sign(data['combined_signal']) * np.sign(data['combined_signal'].shift(1))
    
    # Stability-weighted output
    data['stability_weighted_output'] = data['combined_signal'] * data['price_stability'] * (1 + data['signal_consistency'])
    
    # Volatility context
    data['range_volatility'] = np.sqrt((data['daily_range'] ** 2).rolling(window=20).mean())
    data['return_volatility'] = np.sqrt(((data['close'] / data['close'].shift(1) - 1) ** 2).rolling(window=20).mean())
    
    # Final factor
    data['final_factor'] = (data['stability_weighted_output'] * data['volume_trend_ratio'] * data['dollar_volume_concentration']) / (data['range_volatility'] * data['return_volatility']) * np.sign(data['efficiency_divergence'])
    
    # Clean up intermediate columns
    result = data['final_factor'].copy()
    
    return result
