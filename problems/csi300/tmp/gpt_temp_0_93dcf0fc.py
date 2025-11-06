import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Volatility Regime Identification
    data['short_term_vol'] = (data['high'] - data['low']) / data['close'].shift(1)
    data['medium_term_vol'] = data['close'].rolling(window=5).std() / data['close'].shift(5)
    data['vol_regime'] = np.where(data['short_term_vol'] > data['medium_term_vol'], 'high', 'low')
    
    # Microstructure Anchoring Framework
    data['price_anchoring'] = np.abs(data['close'] - (data['high'] + data['low']) / 2) / (data['high'] - data['low'])
    data['mid_range'] = (data['high'] + data['low']) / 2
    data['range_50pct_low'] = data['mid_range'] - 0.25 * (data['high'] - data['low'])
    data['range_50pct_high'] = data['mid_range'] + 0.25 * (data['high'] - data['low'])
    data['in_middle_range'] = ((data['close'] >= data['range_50pct_low']) & 
                              (data['close'] <= data['range_50pct_high'])).astype(int)
    data['volume_middle_pct'] = data['in_middle_range'] * data['volume'] / data['volume']
    
    # Anchoring persistence (3-day correlation of anchoring patterns)
    anchoring_series = data['price_anchoring'].rolling(window=3)
    data['anchoring_persistence'] = anchoring_series.apply(
        lambda x: np.corrcoef(range(len(x)), x)[0,1] if len(x) == 3 and not np.isnan(x).any() else 0, 
        raw=False
    )
    
    # Regime-Adaptive Momentum Signals
    data['high_vol_momentum'] = (data['close'] - data['close'].shift(2)) / (data['high'] - data['low'])
    data['low_vol_momentum'] = (data['close'] - data['close'].shift(5)) / data['close'].rolling(window=5).std()
    
    # Microstructure Quality Assessment
    data['trade_size'] = data['amount'] / data['volume']
    data['trade_size_efficiency'] = data['trade_size'] / data['trade_size'].rolling(window=5).mean()
    data['price_impact'] = np.abs(data['close'] - data['open']) / data['trade_size']
    data['market_depth'] = data['volume_middle_pct'].rolling(window=3).mean()
    
    # Cross-Regime Signal Integration
    data['regime_weighted_momentum'] = np.where(
        data['vol_regime'] == 'high', 
        data['high_vol_momentum'] * 0.7 + data['low_vol_momentum'] * 0.3,
        data['high_vol_momentum'] * 0.3 + data['low_vol_momentum'] * 0.7
    )
    
    # Anchoring confirmation (when momentum and anchoring align)
    momentum_direction = np.sign(data['regime_weighted_momentum'])
    anchoring_direction = np.where(data['close'] > data['mid_range'], 1, -1)
    data['anchoring_aligned'] = (momentum_direction == anchoring_direction).astype(float)
    data['anchoring_confirmation'] = 1 + (data['anchoring_aligned'] * data['price_anchoring'])
    
    # Microstructure quality filtering
    data['quality_filter'] = np.where(
        (data['trade_size_efficiency'] > 0.8) & 
        (data['price_impact'] < data['price_impact'].rolling(window=10).quantile(0.8)) &
        (data['market_depth'] > 0.3),
        1.0, 0.5
    )
    
    # Cross-timeframe validation
    data['momentum_2d'] = data['close'] - data['close'].shift(2)
    data['momentum_5d'] = data['close'] - data['close'].shift(5)
    data['momentum_aligned'] = (np.sign(data['momentum_2d']) == np.sign(data['momentum_5d'])).astype(float)
    data['timeframe_validation'] = 1 + (data['momentum_aligned'] * 0.2)
    
    # Adaptive Signal Enhancement
    # Regime-dependent scaling
    vol_ratio = data['short_term_vol'] / (data['medium_term_vol'] + 1e-8)
    data['regime_scaling'] = np.where(
        data['vol_regime'] == 'high',
        1 / (1 + vol_ratio),  # Reduce in high volatility
        1 + (1 / (vol_ratio + 1e-8))  # Amplify in low volatility
    )
    
    # Anchoring persistence validation
    data['persistence_multiplier'] = np.where(
        data['anchoring_persistence'] > 0.5, 1.2,  # Strong anchoring
        np.where(data['anchoring_persistence'] < -0.5, 0.8,  # Weak anchoring
                np.where(np.abs(data['anchoring_persistence']) < 0.2, 0.5, 1.0)  # Contradictory
        )
    )
    
    # Dynamic Composite Generation
    data['base_momentum'] = data['regime_weighted_momentum'] * data['anchoring_confirmation']
    data['quality_adjusted'] = data['base_momentum'] * data['quality_filter'] * data['timeframe_validation']
    data['persistence_enhanced'] = data['quality_adjusted'] * data['persistence_multiplier']
    data['final_factor'] = data['persistence_enhanced'] * data['regime_scaling']
    
    # Clean up intermediate columns and return final factor
    result = data['final_factor'].copy()
    return result
