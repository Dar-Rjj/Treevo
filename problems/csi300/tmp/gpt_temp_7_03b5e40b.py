import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate basic price differences and ratios
    data['close_5d_diff'] = data['close'] - data['close'].shift(5)
    data['close_10d_diff'] = data['close'].shift(5) - data['close'].shift(10)
    data['close_3d_diff'] = data['close'] - data['close'].shift(3)
    data['close_6d_diff'] = data['close'].shift(3) - data['close'].shift(6)
    data['close_20d_diff'] = data['close'].shift(5) - data['close'].shift(20)
    
    # Calculate momentum decay base component
    data['momentum_decay_base'] = data['close_5d_diff'] / (data['close_10d_diff'] + 1e-10)
    
    # Volatility components
    data['daily_range'] = data['high'] - data['low']
    data['prev_daily_range'] = data['daily_range'].shift(1)
    data['range_5d'] = data['daily_range'] / (data['daily_range'].shift(5) + 1e-10)
    data['volatility_expansion'] = data['daily_range'] / (data['prev_daily_range'] + 1e-10)
    
    # Volume components
    data['volume_ratio'] = data['volume'] / (data['volume'].shift(5) + 1e-10)
    data['volume_range_coherence'] = (data['volume'] * data['daily_range']) / (
        data['volume'].shift(1) * data['prev_daily_range'] + 1e-10)
    data['trade_size'] = data['amount'] / (data['volume'] + 1e-10)
    
    # Trend components
    data['trend_aligned'] = np.sign(data['close_5d_diff']) * data['momentum_decay_base']
    data['trend_consistency'] = (np.sign(data['close_20d_diff']) * 
                                np.sign(data['close_5d_diff']))
    
    # Clean momentum
    data['clean_momentum'] = (data['close'] / data['close'].shift(1) - 1) * data['momentum_decay_base']
    
    # Breakout asymmetry
    data['high_diff'] = data['high'] - data['high'].shift(1)
    data['low_diff'] = data['low'] - data['low'].shift(1)
    data['breakout_asymmetry'] = data['high_diff'] - data['low_diff']
    
    # Microstructure rejection components
    data['max_open_close'] = data[['open', 'close']].max(axis=1)
    data['min_open_close'] = data[['open', 'close']].min(axis=1)
    data['upside_rejection'] = data['high'] - data['max_open_close']
    data['downside_rejection'] = data['min_open_close'] - data['low']
    data['net_rejection'] = data['upside_rejection'] - data['downside_rejection']
    data['rejection_bias'] = data['upside_rejection'] / (data['downside_rejection'] + 1e-10)
    
    # Efficiency components
    data['intraday_efficiency'] = np.abs(data['close'] - data['open']) / (data['daily_range'] + 1e-10)
    data['efficiency_momentum'] = (data['intraday_efficiency'] / data['intraday_efficiency'].shift(1) - 1)
    
    # Order flow components
    data['implicit_bid'] = ((data['close'] - data['low']) / (data['daily_range'] + 1e-10)) * data['volume']
    data['implicit_ask'] = ((data['high'] - data['close']) / (data['daily_range'] + 1e-10)) * data['volume']
    data['order_flow_imbalance'] = (data['implicit_bid'] - data['implicit_ask']) / (data['volume'] + 1e-10)
    data['order_flow_momentum'] = data['order_flow_imbalance'] * data['momentum_decay_base']
    
    # Momentum decay acceleration
    data['short_term_acceleration'] = (
        (data['close_3d_diff'] / (data['close'].shift(3) + 1e-10)) - 
        (data['close_6d_diff'] / (data['close'].shift(6) + 1e-10))
    )
    
    # Define momentum decay acceleration for use in other components
    data['momentum_decay_acceleration'] = data['short_term_acceleration'] * data['momentum_decay_base']
    
    # Volatility regime classification
    data['volatility_regime'] = np.where(data['daily_range'] > data['daily_range'].rolling(20).mean() * 1.2, 
                                       1.3, 1.0)
    
    # Volume concentration
    data['volume_ma_5'] = data['volume'].rolling(5).mean()
    data['volume_concentration'] = np.where(data['volume'] > data['volume_ma_5'] * 1.5, 1.2, 0.8)
    
    # Calculate persistence measures
    def calculate_persistence(data, signal_col, price_col, window=5):
        persistence = []
        for i in range(len(data)):
            if i < window:
                persistence.append(0)
                continue
            count = 0
            for j in range(window):
                idx = i - j
                if (np.sign(data[signal_col].iloc[idx]) == 
                    np.sign(data[price_col].iloc[idx] - data[price_col].iloc[idx-1])):
                    count += 1
            persistence.append(count / window)
        return persistence
    
    data['momentum_persistence'] = calculate_persistence(data, 'clean_momentum', 'close')
    data['volume_persistence'] = calculate_persistence(data, 'volume_ratio', 'close')
    data['efficiency_persistence'] = calculate_persistence(data, 'intraday_efficiency', 'close')
    data['order_flow_persistence'] = calculate_persistence(data, 'order_flow_imbalance', 'close')
    
    # Range dynamics
    data['range_expansion'] = (data['daily_range'] / data['prev_daily_range'] > 1.2).astype(float)
    data['range_contraction'] = (data['daily_range'] / data['prev_daily_range'] < 0.8).astype(float)
    data['mean_reversion_strength'] = 1 - (np.abs(data['close'] - data['close'].shift(1)) / (data['daily_range'] + 1e-10))
    
    # Core decay velocity factors
    data['microstructure_confirmed_decay'] = (
        data['net_rejection'] * 
        np.sign(data['net_rejection']) * np.sign(data['order_flow_momentum']) * 
        data['volatility_regime']
    )
    
    data['volume_efficiency_decay'] = (
        data['intraday_efficiency'] * (data['volume_ratio'] - 1) * data['momentum_decay_base'] *
        data['volume_persistence'] * data['volume_concentration']
    )
    
    data['breakout_momentum_decay'] = (
        data['breakout_asymmetry'] * data['intraday_efficiency'] * data['momentum_decay_acceleration'] *
        np.sign(data['breakout_asymmetry']) * np.sign(data['intraday_efficiency']) *
        data['trend_consistency']
    )
    
    data['range_enhanced_decay'] = (
        data['mean_reversion_strength'] * data['range_expansion'] * data['volatility_regime']
    )
    
    # Divergence-enhanced decay components
    data['aligned_momentum_decay'] = (
        data['momentum_decay_base'] * data['volume_ratio'] *
        np.sign(data['clean_momentum']) * np.sign(data['volume_ratio'] - 1) * 
        np.sign(data['momentum_decay_acceleration']) * data['momentum_persistence']
    )
    
    data['efficiency_confirmed_decay'] = (
        data['efficiency_momentum'] * data['momentum_decay_acceleration'] *
        np.sign(data['efficiency_momentum']) * np.sign(data['volume_ratio'] - 1) *
        data['efficiency_persistence']
    )
    
    data['order_flow_decay_velocity'] = (
        data['order_flow_momentum'] *
        np.sign(data['order_flow_imbalance']) * np.sign(data['net_rejection']) *
        data['momentum_decay_base'] * data['volume_ratio']
    )
    
    data['volatility_adaptive_decay'] = (
        data['volatility_expansion'] * data['momentum_decay_acceleration'] *
        np.sign(data['volatility_expansion']) * np.sign(data['momentum_decay_acceleration']) *
        data['volatility_regime']
    )
    
    # Final composite alpha construction with regime-specific weights
    primary_decay = data['microstructure_confirmed_decay'] * data['volatility_regime']
    secondary_decay = data['aligned_momentum_decay'] * data['volume_concentration']
    tertiary_decay = data['efficiency_confirmed_decay'] * np.abs(data['trend_consistency'])
    quaternary_decay = data['order_flow_decay_velocity'] * data['range_expansion']
    
    # Composite regime-adaptive alpha
    composite_alpha = (
        0.4 * primary_decay +
        0.3 * secondary_decay +
        0.2 * tertiary_decay +
        0.1 * quaternary_decay
    )
    
    # Apply persistence validation
    persistence_filter = (
        data['momentum_persistence'] * data['volume_persistence'] * 
        data['efficiency_persistence'] * data['order_flow_persistence']
    )
    
    final_alpha = composite_alpha * persistence_filter
    
    # Clean up and return
    final_alpha = final_alpha.replace([np.inf, -np.inf], np.nan)
    final_alpha = final_alpha.fillna(0)
    
    return final_alpha
