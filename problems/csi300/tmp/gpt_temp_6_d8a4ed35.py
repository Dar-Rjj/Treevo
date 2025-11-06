import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    data = df.copy()
    
    # Momentum Acceleration Framework
    # Multi-Timeframe Momentum Structure
    data['short_term_momentum'] = data['close'] / data['close'].shift(3) - 1
    data['medium_term_momentum'] = data['close'] / data['close'].shift(20) - 1
    data['momentum_decay_rate'] = data['short_term_momentum'] / data['medium_term_momentum'] - 1
    
    # Range-Based Momentum Enhancement
    data['upward_range_momentum'] = data['high'] - data['close'].shift(1)
    data['downward_range_momentum'] = data['close'].shift(1) - data['low']
    data['net_range_momentum'] = data['upward_range_momentum'] - data['downward_range_momentum']
    data['range_momentum_acceleration'] = data['net_range_momentum'] / data['net_range_momentum'].shift(5) - 1
    
    # Momentum Quality Assessment
    data['momentum_alignment'] = np.sign(data['short_term_momentum']) * np.sign(data['range_momentum_acceleration'])
    data['momentum_divergence'] = np.abs(data['momentum_decay_rate'] - data['range_momentum_acceleration'])
    data['momentum_quality_score'] = data['momentum_alignment'] * (1 - data['momentum_divergence'])
    
    # Liquidity-Convergence Analysis
    # Volume Liquidity Dynamics
    data['volume_liquidity'] = data['volume'] / (data['high'] - data['low']).replace(0, np.nan)
    data['liquidity_momentum'] = data['volume_liquidity'] / data['volume_liquidity'].shift(5) - 1
    data['volume_efficiency'] = np.abs(data['close'] - data['open']) * data['amount']
    
    # Amount-Price Convergence
    data['amount_momentum'] = data['amount'] / data['amount'].rolling(window=5).mean().shift(1) - 1
    data['price_amount_alignment'] = np.sign(data['short_term_momentum']) * np.sign(data['amount_momentum'])
    data['convergence_strength'] = data['price_amount_alignment'] * np.abs(data['amount_momentum'])
    
    # Microstructure Confirmation
    data['opening_gap_strength'] = data['open'] / data['close'].shift(1) - 1
    data['intraday_reversal_signal'] = data['close'] / data['high'] - 1
    data['closing_momentum'] = data['close'] / data['low'] - 1
    
    # Volatility-Regime Framework
    # Volatility Classification
    data['true_range'] = np.maximum(
        data['high'] - data['low'],
        np.maximum(
            np.abs(data['high'] - data['close'].shift(1)),
            np.abs(data['low'] - data['close'].shift(1))
        )
    )
    data['rolling_volatility'] = data['true_range'].rolling(window=10).mean()
    data['volatility_regime'] = 'normal'
    high_vol_threshold = 1.5 * data['true_range'].rolling(window=20).mean()
    low_vol_threshold = 0.5 * data['true_range'].rolling(window=20).mean()
    data.loc[data['true_range'] > high_vol_threshold, 'volatility_regime'] = 'high'
    data.loc[data['true_range'] < low_vol_threshold, 'volatility_regime'] = 'low'
    
    # Liquidity Regime Detection
    data['volume_concentration'] = data['volume'].rolling(window=5).max() / data['volume'].rolling(window=5).sum()
    data['spread_proxy'] = (data['high'] - data['low']) / data['close']
    spread_median = data['spread_proxy'].rolling(window=20).median()
    data['liquidity_regime'] = 'medium'
    data.loc[(data['volume_concentration'] < 0.4) & (data['spread_proxy'] < spread_median), 'liquidity_regime'] = 'high'
    data.loc[data['spread_proxy'] > 1.5 * spread_median, 'liquidity_regime'] = 'low'
    
    # Adaptive Signal Integration
    # Core Signal Components
    data['momentum_core'] = data['momentum_quality_score'] * data['convergence_strength']
    data['liquidity_core'] = data['volume_efficiency'] * data['liquidity_momentum']
    data['microstructure_core'] = data['opening_gap_strength'] * data['closing_momentum']
    
    # Volatility-Adaptive Weighting
    data['volatility_weighted_signal'] = 0.0
    high_vol_mask = data['volatility_regime'] == 'high'
    normal_vol_mask = data['volatility_regime'] == 'normal'
    low_vol_mask = data['volatility_regime'] == 'low'
    
    data.loc[high_vol_mask, 'volatility_weighted_signal'] = (
        data['momentum_core'] * 0.6 + 
        data['liquidity_core'] * 0.3 + 
        data['microstructure_core'] * 0.1
    )
    data.loc[normal_vol_mask, 'volatility_weighted_signal'] = (
        data['momentum_core'] * 0.4 + 
        data['liquidity_core'] * 0.4 + 
        data['microstructure_core'] * 0.2
    )
    data.loc[low_vol_mask, 'volatility_weighted_signal'] = (
        data['momentum_core'] * 0.3 + 
        data['liquidity_core'] * 0.3 + 
        data['microstructure_core'] * 0.4
    )
    
    # Liquidity Regime Adjustment
    data['liquidity_adjusted_signal'] = data['volatility_weighted_signal']
    data.loc[data['liquidity_regime'] == 'high', 'liquidity_adjusted_signal'] *= 1.2
    data.loc[data['liquidity_regime'] == 'low', 'liquidity_adjusted_signal'] *= 0.7
    
    # Final Alpha Synthesis
    # Signal Quality Validation
    data['short_term_roc'] = data['close'] / data['close'].shift(5) - 1
    data['medium_term_roc'] = data['close'] / data['close'].shift(20) - 1
    
    # Calculate rolling correlations
    short_term_corr = []
    medium_term_corr = []
    
    for i in range(len(data)):
        if i >= 20:
            short_window = data.iloc[i-9:i+1] if i >= 9 else data.iloc[:i+1]
            medium_window = data.iloc[i-19:i+1] if i >= 19 else data.iloc[:i+1]
            
            st_corr = short_window['liquidity_adjusted_signal'].corr(short_window['short_term_roc'])
            mt_corr = medium_window['liquidity_adjusted_signal'].corr(medium_window['medium_term_roc'])
            
            short_term_corr.append(st_corr if not np.isnan(st_corr) else 0)
            medium_term_corr.append(mt_corr if not np.isnan(mt_corr) else 0)
        else:
            short_term_corr.append(0)
            medium_term_corr.append(0)
    
    data['short_term_predictive_power'] = short_term_corr
    data['medium_term_consistency'] = medium_term_corr
    data['quality_score'] = (data['short_term_predictive_power'] + data['medium_term_consistency']) / 2
    
    # Dynamic Scaling
    data['return_volatility'] = (data['close'] / data['close'].shift(1) - 1).rolling(window=10).std()
    data['volatility_scaled_signal'] = data['liquidity_adjusted_signal'] / data['return_volatility'].replace(0, np.nan)
    data['volume_scaling'] = data['volume'] / data['volume'].rolling(window=10).mean()
    data['volume_scaled_signal'] = data['volatility_scaled_signal'] * data['volume_scaling']
    
    # Final Alpha Factor
    data['raw_alpha'] = data['volume_scaled_signal'] * data['quality_score']
    
    # Trend Regime Filter
    data['trend_regime'] = 'bearish'
    data.loc[data['close'] > data['close'].rolling(window=20).mean(), 'trend_regime'] = 'bullish'
    
    # Enhanced Alpha
    data['enhanced_alpha'] = data['raw_alpha']
    bullish_mask = data['trend_regime'] == 'bullish'
    data.loc[bullish_mask, 'enhanced_alpha'] = data['raw_alpha'] * (1 + data['momentum_decay_rate'])
    data.loc[~bullish_mask, 'enhanced_alpha'] = data['raw_alpha'] * (1 - data['intraday_reversal_signal'])
    
    return data['enhanced_alpha']
