import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    data = df.copy()
    
    # Volatility-Momentum Regime Classification
    # Short-term momentum regime identification
    data['momentum_3d'] = data['close'] / data['close'].shift(3) - 1
    data['momentum_5d'] = data['close'] / data['close'].shift(5) - 1
    
    def classify_momentum_regime(row):
        if pd.isna(row['momentum_3d']) or pd.isna(row['momentum_5d']):
            return 'Transition'
        if row['momentum_3d'] > 0 and row['momentum_5d'] > 0:
            return 'Bullish'
        elif row['momentum_3d'] < 0 and row['momentum_5d'] < 0:
            return 'Bearish'
        else:
            return 'Transition'
    
    data['momentum_regime'] = data.apply(classify_momentum_regime, axis=1)
    
    # Volatility regime detection
    data['volatility_3d'] = (data['high'] - data['low']) / data['close']
    data['rolling_high_8d'] = data['high'].rolling(window=8, min_periods=1).max()
    data['rolling_low_8d'] = data['low'].rolling(window=8, min_periods=1).min()
    data['volatility_8d'] = (data['rolling_high_8d'] - data['rolling_low_8d']) / data['close']
    
    def classify_volatility_regime(row):
        if pd.isna(row['volatility_3d']) or pd.isna(row['volatility_8d']):
            return 'Stable'
        if row['volatility_3d'] > row['volatility_8d'] * 1.1:
            return 'High'
        elif row['volatility_3d'] < row['volatility_8d'] * 0.9:
            return 'Low'
        else:
            return 'Stable'
    
    data['volatility_regime'] = data.apply(classify_volatility_regime, axis=1)
    
    # Regime combination matrix
    data['regime_combo'] = data['momentum_regime'] + '_' + data['volatility_regime']
    
    # Volume-Price Efficiency Analysis
    # Price movement efficiency calculation
    data['net_price_movement'] = abs(data['close'] - data['close'].shift(5))
    
    def calculate_total_oscillation(row, idx):
        if idx < 4:
            return np.nan
        total = 0
        for i in range(5):
            total += abs(data.loc[data.index[idx-i], 'high'] - data.loc[data.index[idx-i], 'low'])
        return total
    
    data['total_oscillation'] = [calculate_total_oscillation(row, i) for i, (idx, row) in enumerate(data.iterrows())]
    data['price_efficiency'] = data['net_price_movement'] / data['total_oscillation']
    
    # Volume confirmation strength
    data['volume_trend'] = data['volume'] / data['volume'].shift(5) - 1
    data['volume_price_alignment'] = np.sign(data['momentum_5d']) * np.sign(data['volume_trend'])
    data['volume_efficiency_score'] = data['volume_price_alignment'] * data['price_efficiency']
    
    # Amount-based liquidity assessment
    data['amount_momentum'] = data['amount'] / data['amount'].shift(3) - 1
    data['amount_volume_consistency'] = np.sign(data['amount_momentum']) * np.sign(data['volume_trend'])
    data['liquidity_efficiency'] = data['amount_volume_consistency'] * data['volume_efficiency_score']
    
    # Breakout-Retracement Pattern Recognition
    # Fractal breakout strength measurement
    data['rolling_high_5d'] = data['high'].rolling(window=5, min_periods=1).max()
    data['breakout_5d'] = data['high'] / data['rolling_high_5d'].shift(1) - 1
    
    data['rolling_low_3d'] = data['low'].rolling(window=3, min_periods=1).min()
    data['retracement_3d'] = data['low'] / data['rolling_low_3d'].shift(1) - 1
    
    data['breakout_retracement_ratio'] = data['breakout_5d'] / abs(data['retracement_3d'].replace(0, 0.0001))
    
    # Volume-confirmed pattern strength
    data['volume_breakout'] = data['volume'] / ((data['volume'].shift(1) + data['volume'].shift(2) + data['volume'].shift(3)) / 3)
    data['volume_retracement'] = data['volume'].shift(1) / ((data['volume'].shift(2) + data['volume'].shift(3) + data['volume'].shift(4)) / 3)
    data['volume_pattern_ratio'] = data['volume_breakout'] / data['volume_retracement'].replace(0, 0.0001)
    
    # Multi-Timeframe Momentum Convergence
    data['momentum_8d'] = data['close'] / data['close'].shift(8) - 1
    data['momentum_convergence'] = np.sign(data['momentum_3d']) * np.sign(data['momentum_8d']) * np.minimum(abs(data['momentum_3d']), abs(data['momentum_8d']))
    data['momentum_acceleration'] = (data['momentum_3d'] - data['momentum_8d']) / data['momentum_8d'].replace(0, 0.0001)
    
    # Regime-adaptive momentum weighting
    def calculate_momentum_weighting(row):
        if row['volatility_regime'] == 'High':
            return 0.7, 0.3
        elif row['volatility_regime'] == 'Low':
            return 0.3, 0.7
        else:
            return 0.5, 0.5
    
    weight_results = data.apply(calculate_momentum_weighting, axis=1, result_type='expand')
    data[['short_weight', 'medium_weight']] = weight_results
    
    # Price-Range Efficiency Synthesis
    data['range_efficiency'] = (data['close'] - data['open']) / (data['high'] - data['low']).replace(0, 0.0001)
    data['range_consistency_3d'] = (data['high'] - data['low']) / ((data['high'].shift(1) - data['low'].shift(1) + data['high'].shift(2) - data['low'].shift(2)) / 2)
    
    # Intraday momentum persistence
    data['opening_momentum'] = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
    data['intraday_momentum'] = (data['close'] - data['open']) / data['open']
    data['momentum_continuation'] = np.sign(data['opening_momentum']) * np.sign(data['intraday_momentum']) * np.minimum(abs(data['opening_momentum']), abs(data['intraday_momentum']))
    
    # Volume-Price Regime Consistency
    data['avg_volume_5d'] = data['volume'].rolling(window=5, min_periods=1).mean()
    
    def classify_volume_regime(row):
        if pd.isna(row['volume']) or pd.isna(row['avg_volume_5d']):
            return 'Normal'
        if row['volume'] > 1.5 * row['avg_volume_5d']:
            return 'High'
        elif row['volume'] < 0.7 * row['avg_volume_5d']:
            return 'Low'
        else:
            return 'Normal'
    
    data['volume_regime'] = data.apply(classify_volume_regime, axis=1)
    
    # Volume regime persistence
    data['volume_regime_persistence'] = 0
    current_regime = None
    persistence_count = 0
    
    for i, (idx, row) in enumerate(data.iterrows()):
        if i == 0 or row['volume_regime'] != current_regime:
            persistence_count = 1
            current_regime = row['volume_regime']
        else:
            persistence_count += 1
        data.loc[idx, 'volume_regime_persistence'] = persistence_count
    
    data['volume_regime_stability'] = data['volume_regime_persistence'] / 5
    
    # Multi-Dimensional Signal Integration
    # Primary momentum-volatility component
    def get_volatility_multiplier(regime):
        multipliers = {'High': 1.2, 'Low': 0.8, 'Stable': 1.0}
        return multipliers.get(regime, 1.0)
    
    data['volatility_multiplier'] = data['volatility_regime'].apply(get_volatility_multiplier)
    data['base_signal'] = data['momentum_5d'] * data['volatility_multiplier']
    
    # Volume-efficiency enhancement
    data['volume_confirmed_momentum'] = data['base_signal'] * data['volume_efficiency_score']
    data['efficiency_adjusted'] = data['volume_confirmed_momentum'] * data['price_efficiency']
    data['liquidity_enhanced'] = data['efficiency_adjusted'] * data['liquidity_efficiency']
    
    # Breakout-pattern confirmation
    data['pattern_strength'] = data['breakout_retracement_ratio'] * data['volume_pattern_ratio']
    data['pattern_aligned_momentum'] = data['efficiency_adjusted'] * data['pattern_strength']
    
    # Final Factor Construction
    # Regime-Adaptive Signal Blending
    def regime_signal_blending(row):
        regime = row['regime_combo']
        
        if regime == 'Bullish_High':
            return 0.4 * row['pattern_aligned_momentum'] + 0.6 * row['momentum_3d']
        elif regime == 'Bullish_Low':
            return 0.6 * row['pattern_aligned_momentum'] + 0.4 * row['momentum_8d']
        elif regime == 'Bearish_High':
            return 0.5 * row['pattern_aligned_momentum'] + 0.5 * row['momentum_3d']
        elif regime == 'Bearish_Low':
            return 0.7 * row['pattern_aligned_momentum'] + 0.3 * row['momentum_8d']
        else:  # Transition regimes
            return 0.5 * row['pattern_aligned_momentum'] + 0.5 * (row['momentum_3d'] * row['short_weight'] + row['momentum_8d'] * row['medium_weight'])
    
    data['final_signal'] = data.apply(regime_signal_blending, axis=1)
    
    # Volume-Regime Adjustment
    def volume_adjustment(row):
        if row['volume_regime'] == 'High':
            return row['final_signal'] * 1.3
        elif row['volume_regime'] == 'Low':
            return row['final_signal'] * 0.7
        else:
            return row['final_signal'] * 1.0
    
    data['volume_adjusted_signal'] = data.apply(volume_adjustment, axis=1)
    
    # Final Multi-Dimensional Momentum-Volatility Regime Factor
    data['range_adjusted_signal'] = data['volume_adjusted_signal'] * (1 + data['range_efficiency'])
    data['final_factor'] = data['range_adjusted_signal'] * (1 + data['volume_regime_stability'])
    
    return data['final_factor']
