import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    data = df.copy()
    
    # Calculate True Range
    method1 = data['high'] - data['low']
    method2 = abs(data['high'] - data['close'].shift(1))
    method3 = abs(data['low'] - data['close'].shift(1))
    data['true_range'] = np.maximum(method1, np.maximum(method2, method3))
    
    # Multi-Scale Volatility Efficiency
    # Short-Term Efficiency (5-day)
    net_move_5d = abs(data['close'] - data['close'].shift(5))
    total_move_5d = abs(data['close'] - data['close'].shift(1)).rolling(window=5).sum()
    data['efficiency_5d'] = net_move_5d / (total_move_5d + 1e-8)
    
    # Medium-Term Efficiency (10-day)
    net_move_10d = abs(data['close'] - data['close'].shift(10))
    total_move_10d = abs(data['close'] - data['close'].shift(1)).rolling(window=10).sum()
    data['efficiency_10d'] = net_move_10d / (total_move_10d + 1e-8)
    
    # Liquidity Momentum
    data['avg_amount_3d'] = (data['amount'] + data['amount'].shift(1) + data['amount'].shift(2)) / 3
    data['avg_amount_8d'] = data['amount'].rolling(window=8).mean()
    data['liquidity_momentum'] = data['avg_amount_3d'] / (data['avg_amount_8d'] + 1e-8) - 1
    
    # Volume Efficiency
    data['intraday_eff'] = abs(data['close'] - data['open']) / (data['high'] - data['low'] + 1e-8)
    
    # Calculate average trade size (amount/volume)
    data['trade_size'] = data['amount'] / (data['volume'] + 1e-8)
    data['avg_trade_5d'] = data['trade_size'].rolling(window=5).mean()
    data['avg_trade_20d'] = data['trade_size'].rolling(window=20).mean()
    data['volume_concentration'] = data['avg_trade_5d'] / (data['avg_trade_20d'] + 1e-8)
    
    # Volatility Regime Classification
    data['avg_TR_5d'] = data['true_range'].rolling(window=5).mean()
    data['avg_TR_20d'] = data['true_range'].rolling(window=20).mean()
    data['volatility_ratio'] = data['avg_TR_5d'] / (data['avg_TR_20d'] + 1e-8)
    
    # Momentum Quality Assessment
    # 5-day return
    data['return_5d'] = data['close'] / data['close'].shift(5) - 1
    
    # Directional Consistency
    daily_returns = data['close'].pct_change()
    sign_5d = np.sign(data['return_5d'])
    same_sign_count = pd.Series(index=data.index, dtype=float)
    
    for i in range(len(data)):
        if i >= 5:
            window_returns = daily_returns.iloc[i-4:i+1]
            window_signs = np.sign(window_returns)
            count = (window_signs == sign_5d.iloc[i]).sum()
            same_sign_count.iloc[i] = count / 5
        else:
            same_sign_count.iloc[i] = 0
    
    data['directional_consistency'] = same_sign_count
    
    # Momentum Stability
    data['momentum_stability'] = pd.Series(index=data.index, dtype=float)
    for i in range(len(data)):
        if i >= 5:
            window_returns = daily_returns.iloc[i-4:i+1]
            variance = window_returns.var()
            data['momentum_stability'].iloc[i] = abs(data['return_5d'].iloc[i]) / (variance + 0.0001)
        else:
            data['momentum_stability'].iloc[i] = 0
    
    # Generate Composite Alpha Factor
    # Base Efficiency
    data['base_efficiency'] = (data['efficiency_5d'] + data['efficiency_10d']) * np.sign(data['return_5d'])
    
    # Apply Liquidity Adjustment
    liquidity_condition = data['liquidity_momentum']
    data['liquidity_adjusted'] = data['base_efficiency'] * np.where(
        liquidity_condition > 0.1, 1.5,
        np.where(liquidity_condition > -0.1, 1.0, 0.7)
    )
    
    # Apply Volume Enhancement
    data['volume_enhanced'] = data['liquidity_adjusted'] * data['intraday_eff'] * data['volume_concentration']
    
    # Apply Volatility Multiplier
    volatility_condition = data['volatility_ratio']
    data['volatility_adjusted'] = data['volume_enhanced'] * np.where(
        volatility_condition > 1.2, 0.8,
        np.where(volatility_condition >= 0.8, 1.2, 1.0)
    )
    
    # Final Alpha
    data['alpha_factor'] = data['volatility_adjusted'] * data['directional_consistency'] * data['momentum_stability']
    
    return data['alpha_factor']
