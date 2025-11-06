import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # 1. Microstructure Momentum Components
    # Price Change Efficiency
    data['price_change'] = data['close'] - data['open']
    data['daily_range'] = data['high'] - data['low']
    data['price_efficiency'] = data['price_change'] / (data['daily_range'] + 1e-8)
    
    # Price Slippage
    data['upside_slippage'] = data['high'] - data['open']
    data['downside_slippage'] = data['open'] - data['low']
    data['net_slippage'] = data['upside_slippage'] - data['downside_slippage']
    
    # Momentum Persistence
    data['direction'] = np.sign(data['price_change'])
    data['dir_consistency'] = 0
    for i in range(1, 4):
        data[f'dir_lag_{i}'] = data['direction'].shift(i)
        data['dir_consistency'] += (data['direction'] == data[f'dir_lag_{i}']).astype(int)
    
    # Momentum Strength
    data['abs_change'] = abs(data['price_change'])
    data['typical_range'] = data['daily_range'].rolling(window=10, min_periods=5).mean()
    data['momentum_strength'] = data['abs_change'] / (data['typical_range'] + 1e-8)
    
    # 2. Volume-Volatility Relationship
    # Normalized Volume
    data['volume_rank'] = data['volume'].rolling(window=10, min_periods=5).apply(
        lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min() + 1e-8), raw=False
    )
    
    # Volatility Percentile (True Range based)
    data['true_range'] = np.maximum(
        data['high'] - data['low'],
        np.maximum(
            abs(data['high'] - data['close'].shift(1)),
            abs(data['low'] - data['close'].shift(1))
        )
    )
    data['volatility_rank'] = data['true_range'].rolling(window=10, min_periods=5).apply(
        lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min() + 1e-8), raw=False
    )
    
    # Anomalous Trading Patterns
    data['volume_spike_low_vol'] = (data['volume_rank'] > 0.7) & (data['volatility_rank'] < 0.3)
    data['low_volume_high_vol'] = (data['volume_rank'] < 0.3) & (data['volatility_rank'] > 0.7)
    
    # Volume Divergence Score
    data['volume_divergence'] = 0
    data.loc[data['volume_spike_low_vol'], 'volume_divergence'] = 1
    data.loc[data['low_volume_high_vol'], 'volume_divergence'] = -1
    
    # 3. Signal Integration
    # Market Regime Classification
    data['market_volatility'] = data['true_range'].rolling(window=20, min_periods=10).std()
    data['vol_regime'] = (data['market_volatility'] > data['market_volatility'].rolling(window=50, min_periods=25).median()).astype(int)
    
    # Trading Efficiency
    data['amount_volume_ratio'] = data['amount'] / (data['volume'] + 1e-8)
    data['efficiency_rank'] = data['amount_volume_ratio'].rolling(window=10, min_periods=5).apply(
        lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min() + 1e-8), raw=False
    )
    
    # Momentum-Volume Divergence Score
    data['momentum_volume_score'] = data['momentum_strength'] * data['volume_divergence']
    
    # Regime-dependent weighting
    data['regime_weight'] = np.where(data['vol_regime'] == 1, 0.7, 1.3)
    
    # Price Efficiency Adjustment
    data['slippage_component'] = data['net_slippage'] / (data['daily_range'] + 1e-8)
    data['efficiency_weight'] = 1 + data['efficiency_rank']
    
    # Final Alpha Factor
    data['alpha_factor'] = (
        data['momentum_volume_score'] * data['regime_weight'] +
        data['dir_consistency'] * data['slippage_component'] * data['efficiency_weight']
    ) / (abs(data['regime_weight']) + abs(data['efficiency_weight']) + 1e-8)
    
    # Clean up intermediate columns
    result = data['alpha_factor'].copy()
    
    return result
