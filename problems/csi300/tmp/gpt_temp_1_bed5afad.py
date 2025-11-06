import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Novel alpha factor combining microstructure momentum, liquidity patterns, 
    behavioral asymmetry, and multi-timeframe regime analysis
    """
    # Copy data to avoid modifying original
    data = df.copy()
    
    # 1. Asymmetric Microstructure Momentum
    # Order Flow Imbalance using volume and amount
    data['dollar_volume'] = data['volume'] * data['close']
    data['avg_trade_size'] = data['amount'] / data['volume'].replace(0, np.nan)
    data['trade_size_ratio'] = data['avg_trade_size'] / data['avg_trade_size'].rolling(5).mean()
    
    # Persistent buy-sell pressure asymmetry
    data['price_pressure'] = (data['close'] - data['open']) / (data['high'] - data['low']).replace(0, np.nan)
    data['pressure_persistence'] = data['price_pressure'].rolling(3).apply(lambda x: np.corrcoef(range(len(x)), x)[0,1] if len(x) > 1 else 0)
    
    # 2. Liquidity-Driven Reversal Patterns
    # Range efficiency with liquidity cycles
    data['true_range'] = np.maximum(data['high'] - data['low'], 
                                   np.maximum(abs(data['high'] - data['close'].shift(1)), 
                                             abs(data['low'] - data['close'].shift(1))))
    data['range_efficiency'] = (data['close'] - data['open']) / data['true_range'].replace(0, np.nan)
    
    # Volume acceleration regime detection
    data['volume_ma_5'] = data['volume'].rolling(5).mean()
    data['volume_ma_20'] = data['volume'].rolling(20).mean()
    data['volume_acceleration'] = (data['volume_ma_5'] - data['volume_ma_20']) / data['volume_ma_20']
    data['volume_spike_ratio'] = data['volume'] / data['volume_ma_20']
    
    # 3. Behavioral Momentum Asymmetry
    # Herding-induced price acceleration
    data['returns_3d'] = data['close'].pct_change(3)
    data['returns_8d'] = data['close'].pct_change(8)
    data['momentum_divergence'] = data['returns_3d'] - data['returns_8d']
    
    # Attention-driven overreaction measurement
    data['volume_return_corr'] = data['volume'].rolling(5).corr(data['close'].pct_change())
    data['price_volume_response'] = data['close'].pct_change() * data['volume_spike_ratio']
    
    # 4. Multi-Timeframe Regime Synthesis
    # Short-long term momentum divergence
    data['price_accel_3d'] = data['close'].pct_change(3) - data['close'].pct_change(1)
    data['price_accel_8d'] = data['close'].pct_change(8) - data['close'].pct_change(3)
    data['acceleration_gap'] = data['price_accel_3d'] - data['price_accel_8d']
    
    # Volume-price efficiency convergence
    data['vwap'] = (data['amount'] / data['volume']).replace([np.inf, -np.inf], np.nan)
    data['vwap_efficiency'] = (data['close'] - data['vwap']) / data['true_range'].replace(0, np.nan)
    
    # Final factor synthesis
    # Combine components with regime-dependent weights
    data['microstructure_momentum'] = (
        data['pressure_persistence'] * 0.3 +
        data['trade_size_ratio'] * 0.2 +
        np.sign(data['momentum_divergence']) * abs(data['acceleration_gap']) * 0.25
    )
    
    data['liquidity_reversal'] = (
        -data['range_efficiency'] * data['volume_acceleration'] * 0.3 +
        -data['vwap_efficiency'] * 0.2 +
        -np.sign(data['volume_return_corr']) * data['price_volume_response'] * 0.25
    )
    
    # Final alpha factor
    alpha_factor = (
        data['microstructure_momentum'] * 0.6 +
        data['liquidity_reversal'] * 0.4
    )
    
    # Normalize and clean
    alpha_factor = alpha_factor.replace([np.inf, -np.inf], np.nan)
    alpha_factor = (alpha_factor - alpha_factor.rolling(20).mean()) / alpha_factor.rolling(20).std()
    
    return alpha_factor
