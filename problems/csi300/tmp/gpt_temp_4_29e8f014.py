import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    data = df.copy()
    
    # Multi-Scale Gap-Pressure Analysis
    # Short-term Gap-Pressure (3-day)
    data['overnight_gap'] = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
    data['mid_close_pressure'] = (data['high'] + data['low']) / 2 - data['close']
    data['gap_pressure_momentum'] = data['overnight_gap'] * data['mid_close_pressure'] * np.sign(data['close'] - data['close'].shift(1))
    
    # Medium-term Gap-Pressure (8-day)
    data['gap_compression_ratio'] = (abs(data['open'] / data['close'].shift(1) - 1) / 
                                   abs(data['open'].shift(8) / data['close'].shift(9) - 1)).replace([np.inf, -np.inf], 0)
    
    # Calculate 8-day pressure sum and price range
    data['pressure_sum_8d'] = data['mid_close_pressure'].rolling(window=8, min_periods=1).sum()
    data['price_range_8d'] = data['high'].rolling(window=8, min_periods=1).max() - data['low'].rolling(window=8, min_periods=1).min()
    data['pressure_persistence'] = data['pressure_sum_8d'] / data['price_range_8d'].replace(0, 1)
    
    data['abs_pressure_sum_8d'] = data['mid_close_pressure'].abs().rolling(window=8, min_periods=1).sum()
    data['gap_pressure_efficiency'] = ((data['close'] / data['close'].shift(8) - 1) / 
                                     data['abs_pressure_sum_8d']).replace([np.inf, -np.inf], 0)
    
    # Long-term Gap-Pressure (20-day)
    data['gap_volatility_20d'] = data['overnight_gap'].rolling(window=20, min_periods=1).std()
    data['structural_gap_break'] = data['gap_pressure_momentum'] / data['gap_volatility_20d'].replace(0, 1)
    
    # Fractal Volume-Amount Dynamics
    data['liquidity'] = data['amount'] / data['volume'].replace(0, 1)
    data['transaction_size_momentum'] = (data['liquidity'] / data['liquidity'].shift(3)) - 1
    data['volume_pressure_alignment'] = data['volume'] * data['mid_close_pressure']
    
    # Medium-term intensity (8-day)
    def rolling_correlation(x, y, window):
        return pd.Series([x.iloc[i-window+1:i+1].corr(y.iloc[i-window+1:i+1]) 
                         if i >= window-1 else np.nan for i in range(len(x))], index=x.index)
    
    data['amount_pressure_corr'] = rolling_correlation(data['amount'], data['mid_close_pressure'], 8)
    data['volume_fractal'] = data['volume'] / data['volume'].rolling(window=7, min_periods=1).mean().shift(1)
    
    # Liquidity Quality Assessment
    data['flow_efficiency'] = (data['close'] - data['open']) / data['amount'].replace(0, 1)
    data['liquidity_momentum'] = data['liquidity'] - data['liquidity'].shift(5)
    data['weighted_pressure'] = data['mid_close_pressure'] * data['liquidity_momentum']
    
    # Volatility-Regime Adjusted Analysis
    data['micro_volatility'] = ((data['high'] - data['low']) / data['close']) * abs(data['open'] - data['close'].shift(1)) / data['close'].shift(1).replace(0, 1)
    data['meso_volatility'] = data['close'].rolling(window=3, min_periods=1).std() / data['volume'].rolling(window=3, min_periods=1).std().replace(0, 1)
    data['macro_volatility'] = data['close'].rolling(window=10, min_periods=1).std() / data['close'].shift(10).rolling(window=10, min_periods=1).std().replace(0, 1)
    
    data['volatility_slope'] = data['meso_volatility'] / data['macro_volatility'].replace(0, 1)
    data['volume_volatility_regime'] = data['volume_fractal'] * data['micro_volatility']
    data['regime_confidence'] = abs(data['volatility_slope'] * data['volume_volatility_regime'])
    
    # Fractal Momentum Convergence
    data['ultra_short_momentum'] = (data['close'] / data['close'].shift(1) - 1) - (data['volume'] / data['volume'].shift(1) - 1)
    data['short_term_momentum'] = (data['close'] / data['close'].shift(3) - 1) - (data['volume'] / data['volume'].shift(3) - 1)
    data['momentum_acceleration'] = data['ultra_short_momentum'] - data['short_term_momentum']
    
    # Multi-Scale Signal Integration
    data['opening_pressure'] = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
    data['closing_momentum'] = (data['close'] - (data['high'] + data['low'])/2) / ((data['high'] + data['low'])/2).replace(0, 1)
    data['micro_score'] = data['opening_pressure'] * data['closing_momentum']
    
    # Meso-Structure Alignment
    data['close_volume_pressure'] = data['close'].rolling(window=3, min_periods=1).corr((data['volume'] * data['mid_close_pressure']).rolling(window=3, min_periods=1).mean())
    data['gap_amount_corr'] = rolling_correlation(data['overnight_gap'], data['amount'], 3)
    data['meso_score'] = data['close_volume_pressure'] * data['gap_amount_corr']
    
    # Signal Hierarchy Construction
    data['fractal_momentum_score'] = (data['ultra_short_momentum'] + data['short_term_momentum'] + data['momentum_acceleration']) / 3
    data['base_signal'] = data['gap_pressure_momentum'] * data['fractal_momentum_score']
    data['structure_enhancement'] = data['base_signal'] * data['meso_score']
    data['multi_scale_integration'] = data['structure_enhancement'] * data['micro_score']
    
    # Composite Fractal Alpha Generation
    data['volatility_weighted_signal'] = data['base_signal'] / (1 + abs(data['meso_volatility']))
    data['regime_validated_signal'] = data['volatility_weighted_signal'] * data['volume_volatility_regime']
    
    # Final Alpha Factor
    data['alpha_factor'] = data['regime_validated_signal'] * data['multi_scale_integration']
    
    return data['alpha_factor']
