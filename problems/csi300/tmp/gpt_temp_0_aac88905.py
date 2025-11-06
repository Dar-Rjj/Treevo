import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Multi-Scale Order Flow Imbalance with Liquidity-Regime Adaptive Weighting
    """
    data = df.copy()
    
    # Calculate Order Flow Imbalance Metrics
    # Intraday Pressure Indicators
    data['buy_pressure'] = (data['close'] - data['low']) / (data['high'] - data['low'] + 1e-8)
    data['sell_pressure'] = (data['high'] - data['close']) / (data['high'] - data['low'] + 1e-8)
    data['opening_gap_pressure'] = abs(data['open'] - data['close'].shift(1)) / (data['high'] - data['low'] + 1e-8)
    
    # Daily imbalance
    data['daily_imbalance'] = data['volume'] * (data['close'] - data['open']) / (data['high'] - data['low'] + 1e-8)
    
    # Volume-Weighted Imbalance Trends
    data['vw_imbalance_3d'] = data['daily_imbalance'].rolling(window=3, min_periods=1).sum()
    data['imbalance_8d_avg'] = data['daily_imbalance'].rolling(window=8, min_periods=1).mean()
    data['imbalance_momentum'] = data['daily_imbalance'] / (data['imbalance_8d_avg'] + 1e-8) - 1
    
    # 15-day imbalance consistency
    imbalance_sign = np.sign(data['daily_imbalance'])
    data['imbalance_consistency'] = imbalance_sign.rolling(window=15, min_periods=1).apply(
        lambda x: np.sum(x == x.iloc[-1]) / len(x) if len(x) > 0 else 0, raw=False
    )
    
    # Price-Volume Divergence Patterns
    price_change = data['close'] - data['close'].shift(1)
    volume_change = data['volume'] - data['volume'].shift(1)
    data['volume_lead_divergence'] = ((np.sign(volume_change) != np.sign(price_change)) & 
                                    (np.sign(volume_change) != 0)).astype(int)
    
    # Divergence persistence
    data['divergence_persistence'] = data['volume_lead_divergence'].rolling(window=5, min_periods=1).sum()
    
    # Identify Liquidity Regime Context
    # Market Depth Conditions
    data['liquidity_concentration'] = data['volume'] / (data['volume'].rolling(window=10, min_periods=1).sum() + 1e-8)
    data['price_impact'] = (data['high'] - data['low']) / (data['volume'] + 1e-8)
    data['absorption_capacity'] = ((data['high'] - data['low']) / (data['volume'] + 1e-8)).rolling(window=5, min_periods=1).mean()
    
    # Classify Order Flow Characteristics
    data['aggressive_flow'] = ((data['buy_pressure'] > 0.7) | (data['sell_pressure'] > 0.7)) & \
                             (data['volume'] > data['volume'].rolling(window=20, min_periods=1).mean())
    data['passive_flow'] = ((data['high'] - data['low']) < data['high'].rolling(window=20, min_periods=1).mean() * 0.01) & \
                          (data['volume'] > data['volume'].rolling(window=20, min_periods=1).mean() * 0.5)
    
    # Microstructure Stress Levels
    data['range_expansion'] = (data['high'] - data['low']) / (data['high'].rolling(window=5, min_periods=1).mean() - 
                                                            data['low'].rolling(window=5, min_periods=1).mean() + 1e-8)
    
    # Closing momentum persistence
    close_position = (data['close'] - data['open']) / (data['high'] - data['low'] + 1e-8)
    data['closing_bias_persistence'] = (np.sign(close_position) == np.sign(close_position.shift(1))).rolling(window=3, min_periods=1).sum()
    
    # Generate Imbalance-Based Alpha Components
    # Multi-Scale Imbalance Score
    short_term_pressure = (data['buy_pressure'] - data['sell_pressure']) * 0.35
    medium_term_momentum = data['imbalance_momentum'] * 0.4
    long_term_consistency = data['imbalance_consistency'] * np.sign(data['daily_imbalance']) * 0.25
    data['multi_scale_imbalance'] = short_term_pressure + medium_term_momentum + long_term_consistency
    
    # Regime-Adaptive Signal Modulation
    high_liquidity = data['liquidity_concentration'] > data['liquidity_concentration'].rolling(window=20, min_periods=1).quantile(0.7)
    low_liquidity = data['liquidity_concentration'] < data['liquidity_concentration'].rolling(window=20, min_periods=1).quantile(0.3)
    
    regime_weight = np.where(high_liquidity, 1.2, 
                           np.where(low_liquidity, 0.8, 1.0))
    
    # Apply Divergence Confirmation Rules
    divergence_enhancement = np.where(data['volume_lead_divergence'] == 1, 1.3, 1.0)
    
    # Generate Composite Alpha Signal
    # Combine Imbalance Score with Liquidity Context
    base_signal = data['multi_scale_imbalance'] * regime_weight * divergence_enhancement
    
    # Apply Microstructure Stress Adjustments
    range_adjustment = np.where(data['range_expansion'] > 1.2, 1.2, 
                              np.where(data['range_expansion'] < 0.8, 0.8, 1.0))
    
    closing_bias_adjustment = 1 + (data['closing_bias_persistence'] * 0.1)
    
    # Final signal with regime confidence
    final_signal = base_signal * range_adjustment * closing_bias_adjustment
    
    # Normalize the signal
    signal_std = final_signal.rolling(window=20, min_periods=1).std()
    normalized_signal = final_signal / (signal_std + 1e-8)
    
    return pd.Series(normalized_signal, index=data.index)
