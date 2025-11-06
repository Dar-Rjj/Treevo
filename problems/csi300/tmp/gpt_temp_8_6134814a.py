import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    data = df.copy()
    
    # Multi-Scale Gap Efficiency Analysis
    data['short_term_gap_eff'] = np.abs(data['close'] - data['open']) / (data['high'] - data['low'])
    
    # Medium-term gap efficiency
    data['medium_term_range_sum'] = data['high'].rolling(window=6).apply(lambda x: (x[1:] - data.loc[x.index[1:], 'low']).sum(), raw=False)
    data['medium_term_gap_eff'] = np.abs(data['close'] - data['open'].shift(5)) / data['medium_term_range_sum']
    data['gap_eff_divergence'] = data['short_term_gap_eff'] - data['medium_term_gap_eff']
    
    # Gap volatility compression
    recent_gap_sum = np.abs(data['close'] - data['open']).rolling(window=5).sum()
    longer_gap_sum = np.abs(data['close'] - data['open']).rolling(window=10).sum()
    data['gap_vol_compression'] = recent_gap_sum / longer_gap_sum - 1
    
    # Volatility-Adjusted Momentum System
    data['true_range'] = np.maximum(
        data['high'] - data['low'],
        np.maximum(
            np.abs(data['high'] - data['close'].shift(1)),
            np.abs(data['low'] - data['close'].shift(1))
        )
    )
    data['true_range_component'] = data['true_range'] / data['close'].shift(1)
    data['gap_vol_component'] = np.abs(data['open'] - data['close'].shift(1)) / data['close'].shift(1)
    
    # Volatility ratio
    price_change_5d = np.abs(data['close'] - data['close'].shift(5)) / 5
    data['volatility_ratio'] = data['true_range_component'] / (price_change_5d / data['close'].shift(5))
    
    # Volatility regime classification
    data['high_vol'] = (data['true_range_component'] > 0.03).astype(int)
    data['low_vol'] = (data['true_range_component'] < 0.01).astype(int)
    data['normal_vol'] = ((data['true_range_component'] >= 0.01) & (data['true_range_component'] <= 0.03)).astype(int)
    
    # Regime-adaptive momentum
    data['efficiency_ratio'] = (data['close'] - data['open']) / data['true_range_component']
    data['vol_scaled_gap'] = (data['open'] - data['close'].shift(1)) / data['true_range_component']
    data['momentum_diff'] = (data['close'] / data['close'].shift(4) - 1) - (data['close'] / data['close'].shift(9) - 1)
    
    # Microstructure Anchoring & Confirmation
    data['opening_anchor_dev'] = (data['open'] - (data['high'].shift(1) + data['low'].shift(1)) / 2) / ((data['high'].shift(1) - data['low'].shift(1)) / 2)
    data['closing_anchor_dev'] = (data['close'] - (data['high'] + data['low']) / 2) / ((data['high'] - data['low']) / 2)
    data['midpoint_persistence'] = np.sign(data['opening_anchor_dev']) * np.sign(data['closing_anchor_dev'])
    
    # Volume-pressure confirmation
    data['morning_gap_pressure'] = (data['high'] - data['open']) / np.abs(data['open'] - data['close'].shift(1))
    data['gap_fill_pressure'] = (data['close'] - data['open']) / np.abs(data['open'] - data['close'].shift(1))
    data['pressure_asymmetry'] = data['morning_gap_pressure'] - data['gap_fill_pressure']
    
    # Anchor breakout detection
    data['strong_anchor_signal'] = data['midpoint_persistence'] * np.abs(data['closing_anchor_dev'])
    data['weak_anchor_signal'] = -data['midpoint_persistence'] * (1 - np.abs(data['closing_anchor_dev']))
    data['anchor_breakout_signal'] = (np.abs(data['opening_anchor_dev']) > 1.5 * np.abs(data['closing_anchor_dev'])).astype(int)
    
    # Amount Flow Integration
    data['amount_per_trade_eff'] = data['amount'] / data['volume']
    data['flow_concentration_ratio'] = data['amount'] / (data['amount'].shift(1) + data['amount'].shift(2) + data['amount'].shift(3)) * 3
    data['flow_persistence'] = np.sign(data['amount'] - data['amount'].shift(1)) * np.sign(data['close'] - data['close'].shift(1))
    
    # Flow-efficiency alignment
    data['concentrated_efficiency'] = data['efficiency_ratio'] * data['flow_concentration_ratio']
    data['distributed_efficiency'] = (np.abs(data['close'] - data['open']) / (data['high'] - data['low'])) * (1 / data['flow_concentration_ratio'])
    data['flow_momentum_alignment'] = data['flow_persistence'] * (data['close'] - data['open'])
    
    # Volume asymmetry metrics
    up_days = data['close'] > data['close'].shift(1)
    data['upside_volume_ratio'] = data['volume'].rolling(window=10).apply(
        lambda x: x[up_days.loc[x.index]].mean() / x.mean() if len(x[up_days.loc[x.index]]) > 0 else 0, raw=False
    )
    
    returns = data['close'].pct_change().fillna(0)
    positive_returns_sum = returns.rolling(window=10).apply(lambda x: x[x > 0].sum(), raw=False)
    negative_returns_sum = returns.rolling(window=10).apply(lambda x: x[x < 0].sum(), raw=False)
    data['price_asymmetry'] = np.log(1 + positive_returns_sum) - np.log(1 + np.abs(negative_returns_sum))
    data['volume_asymmetry'] = data['upside_volume_ratio'] * data['price_asymmetry']
    
    # Cross-Regime Signal Synthesis
    data['high_vol_alpha'] = data['vol_scaled_gap'] * data['anchor_breakout_signal']
    data['low_vol_alpha'] = (np.abs(data['close'] - data['open']) / (data['high'] - data['low'])) * data['strong_anchor_signal']
    data['normal_vol_alpha'] = data['efficiency_ratio'] * data['flow_momentum_alignment']
    
    # Gap-momentum integration
    data['base_gap_momentum'] = np.sqrt(data['gap_eff_divergence'] * data['momentum_diff'])
    data['volume_confirmed_gap_momentum'] = data['base_gap_momentum'] * np.cbrt(data['pressure_asymmetry'] * data['volume_asymmetry'])
    data['flow_enhanced_gap_momentum'] = data['volume_confirmed_gap_momentum'] * data['flow_concentration_ratio']
    
    # Regime-adaptive combination
    data['high_vol_signal'] = data['high_vol_alpha'] * data['gap_vol_compression']
    data['low_vol_signal'] = data['low_vol_alpha'] * (1 / data['true_range_component'])
    data['normal_vol_signal'] = data['normal_vol_alpha'] * data['volatility_ratio']
    data['regime_weighted_signal'] = data['high_vol_signal'] + data['low_vol_signal'] + data['normal_vol_signal']
    
    # Final Alpha Assembly
    data['core_alpha_component'] = data['regime_weighted_signal'] * data['flow_enhanced_gap_momentum']
    data['microstructure_confirmation'] = data['core_alpha_component'] * data['strong_anchor_signal']
    data['final_alpha'] = data['microstructure_confirmation'] * (data['close'] - (data['high'] + data['low']) / 2) / (data['high'] - data['low'])
    
    return data['final_alpha']
