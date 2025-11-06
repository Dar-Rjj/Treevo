import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    data = df.copy()
    
    # Calculate basic price changes and ranges
    data['prev_close'] = data['close'].shift(1)
    data['prev_high'] = data['high'].shift(1)
    data['prev_low'] = data['low'].shift(1)
    data['price_change'] = data['close'] / data['prev_close'] - 1
    
    # Opening Auction Dynamics
    data['opening_imbalance'] = (data['open'] - (data['prev_high'] + data['prev_low'])/2) / ((data['prev_high'] - data['prev_low'])/2 + 1e-8)
    data['opening_volume_intensity'] = data['volume'] / (data['volume'].shift(1) + 1e-8)  # Simplified as full day volume
    data['opening_momentum'] = data['opening_imbalance'] * data['opening_volume_intensity']
    
    # Closing Auction Pressure
    data['closing_imbalance'] = (data['close'] - (data['high'] + data['low'])/2) / ((data['high'] - data['low'])/2 + 1e-8)
    data['closing_volume_concentration'] = data['volume'] / (data['volume'] + 1e-8)  # Simplified as full day volume
    data['closing_pressure'] = data['closing_imbalance'] * data['closing_volume_concentration']
    
    # Auction Cycle Momentum
    data['auction_momentum_gap'] = data['opening_momentum'] - data['closing_pressure']
    
    # Calculate auction persistence
    auction_sign = np.sign(data['auction_momentum_gap'])
    auction_persistence = pd.Series(index=data.index, dtype=float)
    for i in range(9, len(data)):
        window = auction_sign.iloc[i-9:i+1]
        if len(window) == 10:
            current_sign = window.iloc[-1]
            prev_signs = window.iloc[:-1]
            persistence = (prev_signs == current_sign).sum()
            auction_persistence.iloc[i] = persistence
    data['auction_persistence'] = auction_persistence
    data['auction_cycle_strength'] = data['auction_momentum_gap'] * data['auction_persistence']
    
    # Volatility-Regime Detection
    data['abs_return'] = abs(data['price_change'])
    
    # Multi-Scale Volatility Clustering
    short_term_vol = data['abs_return'].rolling(window=4, min_periods=1).mean().shift(1)
    medium_term_vol = data['abs_return'].rolling(window=9, min_periods=1).mean().shift(1)
    
    short_term_cluster = pd.Series(index=data.index, dtype=float)
    medium_term_cluster = pd.Series(index=data.index, dtype=float)
    
    for i in range(9, len(data)):
        # Short-term cluster
        short_window = data['abs_return'].iloc[i-9:i+1]
        short_threshold = short_term_vol.iloc[i]
        short_term_cluster.iloc[i] = (short_window > short_threshold).sum()
        
        # Medium-term cluster
        medium_window = data['abs_return'].iloc[i-19:i+1] if i >= 19 else data['abs_return'].iloc[:i+1]
        medium_threshold = medium_term_vol.iloc[i]
        medium_term_cluster.iloc[i] = (medium_window > medium_threshold).sum()
    
    data['volatility_clustering_ratio'] = short_term_cluster / (1 + medium_term_cluster)
    
    # Asymmetric Volatility Response
    up_days = data['close'] > data['prev_close']
    down_days = data['close'] < data['prev_close']
    
    data['up_day_volatility'] = data['abs_return'].where(up_days).rolling(window=10, min_periods=1).mean()
    data['down_day_volatility'] = data['abs_return'].where(down_days).rolling(window=10, min_periods=1).mean()
    data['volatility_asymmetry'] = data['up_day_volatility'] - data['down_day_volatility']
    
    # Volatility Regime Classification
    data['current_volatility_level'] = data['abs_return'].rolling(window=5, min_periods=1).mean()
    data['historical_volatility_baseline'] = data['abs_return'].rolling(window=15, min_periods=1).mean().shift(5)
    data['volatility_regime'] = data['current_volatility_level'] / (data['historical_volatility_baseline'] + 1e-8)
    
    # Volume-Amount Coherence Patterns
    data['volume_skewness'] = (data['volume'] - data['volume'].shift(1)) / (data['volume'] + data['volume'].shift(1) + 1e-8)
    data['volatility_persistence'] = (data['high'] - data['low']) / (data['prev_high'] - data['prev_low'] + 1e-8)
    
    # Coherence Score (simplified correlation)
    coherence_score = pd.Series(index=data.index, dtype=float)
    for i in range(9, len(data)):
        if i >= 10:
            volume_window = data['volume'].iloc[i-9:i+1]
            range_window = (data['high'] - data['low']).iloc[i-9:i+1]
            if len(volume_window) == 10 and len(range_window) == 10:
                coherence_score.iloc[i] = volume_window.corr(range_window)
    data['coherence_score'] = coherence_score.fillna(0)
    data['coherence_momentum'] = (data['coherence_score'] - data['coherence_score'].shift(5)) * (1 - abs(data['volume_skewness']))
    
    # Amount Flow Dynamics
    data['amount_flow_persistence'] = (data['amount'] > data['amount'].shift(1)).rolling(window=10, min_periods=1).sum()
    data['amount_volatility'] = data['amount'].rolling(window=5, min_periods=1).std() / (data['amount'].rolling(window=5, min_periods=1).mean() + 1e-8)
    data['price_volatility'] = data['close'].rolling(window=5, min_periods=1).std() / (data['close'].rolling(window=5, min_periods=1).mean() + 1e-8)
    data['amount_price_volatility_ratio'] = data['amount_volatility'] / (data['price_volatility'] + 1e-8)
    
    # Volume-Amount Divergence
    volume_amount_corr = pd.Series(index=data.index, dtype=float)
    for i in range(4, len(data)):
        if i >= 5:
            volume_window = data['volume'].iloc[i-4:i+1]
            amount_window = data['amount'].iloc[i-4:i+1]
            if len(volume_window) == 5 and len(amount_window) == 5:
                volume_amount_corr.iloc[i] = volume_window.corr(amount_window)
    data['volume_amount_correlation'] = volume_amount_corr.fillna(0)
    
    data['volume_amount_momentum'] = (data['volume']/data['volume'].shift(1) - 1) * (data['amount']/data['amount'].shift(1) - 1)
    data['divergence_signal'] = data['volume_amount_correlation'] * data['volume_amount_momentum']
    
    # Microstructure Efficiency Signals
    data['range_utilization'] = abs(data['close'] - data['open']) / (data['high'] - data['low'] + 1e-8)
    data['range_efficiency_persistence'] = (data['range_utilization'] > 0.7).rolling(window=10, min_periods=1).sum()
    data['range_momentum'] = data['range_utilization'] * data['range_efficiency_persistence']
    
    # Price Efficiency Under Stress
    data['volume_stress'] = (data['volume'] / data['volume'].rolling(window=4, min_periods=1).mean().shift(1)) * abs((data['volume'] - data['volume'].shift(5)) / (data['volume'].shift(5) + 1e-8))
    data['price_efficiency_ratio'] = abs(data['close'] - data['open']) / (data['high'] - data['low'] + 1e-8)
    data['efficiency_under_stress'] = data['price_efficiency_ratio'].where(data['volume_stress'] > 1, 0)
    
    # Microstructure Momentum
    data['gap_momentum'] = (abs(data['open'] - data['prev_close']) / (data['prev_close'] + 1e-8)) * (abs(data['open'] - data['prev_close']) / data['prev_close'] > data['abs_return'].rolling(window=10, min_periods=1).mean()).rolling(window=20, min_periods=1).sum()
    
    data['close_position'] = abs(data['close'] - (data['high'] + data['low'])/2) / ((data['high'] - data['low'])/2 + 1e-8)
    data['close_position_persistence'] = (data['close'] > (data['high'] + data['low'])/2).rolling(window=10, min_periods=1).sum()
    data['close_position_momentum'] = data['close_position'] * data['close_position_persistence']
    
    data['micro_momentum_composite'] = data['gap_momentum'] * data['close_position_momentum']
    
    # Regime-Adaptive Factor Construction
    data['high_vol_auction'] = data['auction_cycle_strength'] / (data['current_volatility_level'] + 1e-8)
    data['high_vol_coherence'] = data['coherence_momentum'] * data['volatility_asymmetry']
    data['high_vol_micro'] = data['micro_momentum_composite'] / (data['current_volatility_level'] + 1e-8)
    
    data['low_vol_auction'] = data['auction_cycle_strength'] * data['range_momentum']
    data['low_vol_coherence'] = data['coherence_momentum'] * data['amount_flow_persistence']
    data['low_vol_micro'] = data['micro_momentum_composite'] * data['efficiency_under_stress']
    
    # Regime-Adaptive Selection
    high_vol_condition = data['volatility_regime'] > 1
    data['auction_factor'] = np.where(high_vol_condition, data['high_vol_auction'], data['low_vol_auction'])
    data['coherence_factor'] = np.where(high_vol_condition, data['high_vol_coherence'], data['low_vol_coherence'])
    data['micro_factor'] = np.where(high_vol_condition, data['high_vol_micro'], data['low_vol_micro'])
    
    # Compression-Breakout Dynamics
    data['price_range_compression'] = (data['high'] - data['low']) / (data['high'] - data['low']).rolling(window=4, min_periods=1).mean().shift(1)
    data['volume_compression'] = data['volume'] / data['volume'].rolling(window=4, min_periods=1).mean().shift(1)
    data['compression_intensity'] = data['price_range_compression'] * data['volume_compression']
    
    data['price_breakout'] = (data['high'] - data['low']) / (data['high'] - data['low']).rolling(window=9, min_periods=1).mean().shift(1)
    data['volume_breakout'] = data['volume'] / data['volume'].rolling(window=9, min_periods=1).mean().shift(1)
    data['breakout_momentum'] = data['price_breakout'] * data['volume_breakout']
    
    # Compression-Breakout Cycle
    compression_condition = (data['price_range_compression'] < 0.8) & (data['volume_compression'] < 0.8)
    breakout_condition = (data['price_breakout'] > 1.2) & (data['volume_breakout'] > 1.2)
    
    data['compression_phase'] = compression_condition.rolling(window=10, min_periods=1).sum()
    data['breakout_phase'] = breakout_condition.rolling(window=10, min_periods=1).sum()
    data['cycle_strength'] = data['breakout_phase'] / (1 + data['compression_phase'])
    
    # Final Alpha Synthesis
    abs_auction = abs(data['auction_factor'])
    abs_coherence = abs(data['coherence_factor'])
    abs_micro = abs(data['micro_factor'])
    total_abs = abs_auction + abs_coherence + abs_micro + 1e-8
    
    data['auction_weight'] = abs_auction / total_abs
    data['coherence_weight'] = abs_coherence / total_abs
    data['micro_weight'] = abs_micro / total_abs
    
    data['breakout_component'] = data['breakout_momentum'] * data['cycle_strength']
    data['enhanced_core'] = (data['auction_factor'] * data['auction_weight'] + 
                           data['coherence_factor'] * data['coherence_weight'] + 
                           data['micro_factor'] * data['micro_weight'])
    
    data['final_alpha'] = data['enhanced_core'] + data['breakout_component'] * data['divergence_signal']
    
    return data['final_alpha']
