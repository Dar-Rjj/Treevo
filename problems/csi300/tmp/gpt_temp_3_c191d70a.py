import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Initialize result series
    result = pd.Series(index=df.index, dtype=float)
    
    # Calculate basic components
    df['close_ret_1'] = df['close'].pct_change(1)
    df['close_ret_2'] = df['close'].pct_change(2)
    df['volume_ret_1'] = df['volume'].pct_change(1)
    df['volume_ret_5'] = df['volume'].pct_change(5)
    df['volume_ret_13'] = df['volume'].pct_change(13)
    
    # Calculate intraday efficiency
    df['intraday_efficiency'] = np.abs(df['close'] - df['open']) / (df['high'] - df['low'])
    df['intraday_efficiency_prev'] = df['intraday_efficiency'].shift(1)
    
    # Calculate position ratios
    df['close_to_low_ratio'] = (df['close'] - df['low']) / (df['high'] - df['low'])
    df['high_to_close_ratio'] = (df['high'] - df['close']) / (df['high'] - df['low'])
    
    # Calculate ranges for different timeframes
    df['range_5'] = df['high'].rolling(5).max() - df['low'].rolling(5).min()
    df['range_13'] = df['high'].rolling(13).max() - df['low'].rolling(13).min()
    
    # Calculate gap efficiency
    df['gap_efficiency'] = np.abs(df['open'] - df['close'].shift(1)) / (df['high'] - df['low'])
    
    # Calculate volume acceleration
    df['volume_acceleration'] = (df['volume'] - df['volume'].shift(1)) - (df['volume'].shift(1) - df['volume'].shift(2))
    df['range_ratio'] = (df['high'] - df['low']) / (df['high'].shift(1) - df['low'].shift(1))
    
    # Calculate price acceleration
    df['price_acceleration'] = (df['close'] - df['close'].shift(1)) - (df['close'].shift(1) - df['close'].shift(2))
    
    # Calculate morning strength and afternoon weakness proxies
    df['morning_strength'] = (df['open'] - df['low']) / (df['high'] - df['low'])
    df['afternoon_weakness'] = (df['high'] - df['close']) / (df['high'] - df['low'])
    
    # Calculate range expansion/contraction persistence
    df['range_expansion'] = ((df['high'] - df['low']) > (df['high'].shift(1) - df['low'].shift(1))).astype(float)
    df['range_expansion_persistence'] = df['range_expansion'].rolling(5).mean()
    df['range_contraction_persistence'] = 1 - df['range_expansion_persistence']
    
    # Calculate volume concentration proxies
    df['high_volume_concentration'] = (df['volume'] > df['volume'].rolling(5).mean()).astype(float)
    df['low_volume_concentration'] = (df['volume'] < df['volume'].rolling(5).mean()).astype(float)
    
    # Calculate momentum reversal asymmetry
    df['momentum_reversal'] = ((df['close_ret_1'] > 0) & (df['close_ret_1'].shift(1) < 0)).astype(float) - ((df['close_ret_1'] < 0) & (df['close_ret_1'].shift(1) > 0)).astype(float)
    
    # Calculate volatility regime asymmetry
    df['volatility_regime'] = ((df['high'] - df['low']) > (df['high'] - df['low']).rolling(10).mean()).astype(float)
    
    for i in range(len(df)):
        if i < 13:  # Need enough data for calculations
            result.iloc[i] = 0
            continue
            
        try:
            # Multi-Scale Temporal Asymmetry
            # Micro-Scale Momentum Asymmetry
            if df['close'].iloc[i] > df['close'].iloc[i-1] and df['close'].iloc[i-1] > df['close'].iloc[i-2]:
                micro_asymmetry = ((df['close'].iloc[i] - df['close'].iloc[i-1]) / (df['close'].iloc[i-1] - df['close'].iloc[i-2]) * 
                                 np.sign(df['close'].iloc[i] - df['close'].iloc[i-1]) * 
                                 (df['volume'].iloc[i] / df['volume'].iloc[i-1]))
            else:
                micro_asymmetry = 0
            
            # Meso-Scale Range Asymmetry
            meso_asymmetry = ((df['close'].iloc[i] - df['close'].iloc[i-5]) / df['range_5'].iloc[i] * 
                            (df['volume'].iloc[i] / df['volume'].iloc[i-5]) * 
                            (df['morning_strength'].iloc[i] - df['afternoon_weakness'].iloc[i]))
            
            # Macro-Scale Volatility Asymmetry
            macro_asymmetry = ((df['close'].iloc[i] - df['close'].iloc[i-13]) / df['range_13'].iloc[i] * 
                             (df['range_expansion_persistence'].iloc[i] - df['range_contraction_persistence'].iloc[i]) * 
                             df['volume_ret_13'].iloc[i])
            
            multi_scale_asymmetry = micro_asymmetry + meso_asymmetry + macro_asymmetry
            
            # Quantum Position-Temporal Dynamics
            # Opening Position Efficiency
            opening_efficiency = (df['gap_efficiency'].iloc[i] * 
                                df['volume_ret_1'].iloc[i] * 
                                df['morning_strength'].iloc[i])
            
            # Intraday Position Momentum
            intraday_momentum = (df['close_to_low_ratio'].iloc[i] * 
                               np.sign(df['close_ret_1'].iloc[i]) * 
                               (df['intraday_efficiency'].iloc[i] - df['intraday_efficiency_prev'].iloc[i]))
            
            # Closing Position Elasticity
            price_stretch = df['close_to_low_ratio'].iloc[i]
            prev_stretch = df['close_to_low_ratio'].iloc[i-1]
            closing_elasticity = (df['close_to_low_ratio'].iloc[i] * 
                                (df['volume'].iloc[i] / df['volume'].iloc[i-1]) * 
                                (price_stretch - prev_stretch))
            
            position_dynamics = opening_efficiency + intraday_momentum + closing_elasticity
            
            # Volume-Volatility Quantum Integration
            # Quantum Volume Acceleration
            volume_acceleration = (df['volume_acceleration'].iloc[i] * df['range_ratio'].iloc[i])
            
            # Quantum Price-Volume Divergence
            price_volume_divergence = (df['price_acceleration'].iloc[i] * 
                                     (df['volume'].iloc[i] / df['volume'].iloc[i-1]) * 
                                     (df['high_volume_concentration'].iloc[i] - df['low_volume_concentration'].iloc[i]))
            
            # Quantum Volatility Regime
            volatility_regime = (df['range_expansion_persistence'].iloc[i] * 
                               (df['volume'].iloc[i] / df['volume'].iloc[i-1]))
            
            volume_volatility_integration = volume_acceleration + price_volume_divergence + volatility_regime
            
            # Core Temporal Quantum
            core_temporal_quantum = multi_scale_asymmetry * position_dynamics * volume_volatility_integration
            
            # Temporal Efficiency Quantum Patterns
            # Gap Efficiency Quantum
            gap_efficiency_quantum = (df['gap_efficiency'].iloc[i] * 
                                    (df['volume'].iloc[i] / df['volume'].iloc[i-1]) * 
                                    np.sign(df['close_ret_1'].iloc[i]))
            
            # Intraday Efficiency Momentum
            intraday_efficiency_momentum = (df['intraday_efficiency'].iloc[i] * 
                                          (df['intraday_efficiency'].iloc[i] - df['intraday_efficiency_prev'].iloc[i]) * 
                                          (df['volume'].iloc[i] / df['volume'].iloc[i-1]))
            
            # Efficiency-Volume Quantum
            efficiency_volume_quantum = ((df['intraday_efficiency'].iloc[i] - df['intraday_efficiency_prev'].iloc[i]) * 
                                       (df['high_volume_concentration'].iloc[i] - df['low_volume_concentration'].iloc[i]) * 
                                       (df['volume'].iloc[i] / df['volume'].iloc[i-1]))
            
            efficiency_patterns = gap_efficiency_quantum + intraday_efficiency_momentum + efficiency_volume_quantum
            
            # Enhanced Quantum Efficiency
            enhanced_quantum_efficiency = core_temporal_quantum * efficiency_patterns
            
            # Quantum Asymmetry Regimes
            # Position Quantum Regime
            pos_ratio = df['close_to_low_ratio'].iloc[i]
            if pos_ratio > 0:
                position_regime = (-pos_ratio * np.log(pos_ratio) * 
                                 df['high_to_close_ratio'].iloc[i] * 
                                 (df['morning_strength'].iloc[i] - df['afternoon_weakness'].iloc[i]))
            else:
                position_regime = 0
            
            # Volume Quantum Regime
            vol_diff = df['volume'].iloc[i] - df['volume'].iloc[i-1]
            vol_sum = df['volume'].iloc[i] + df['volume'].iloc[i-1]
            if vol_sum > 0:
                volume_regime = (-(vol_diff/vol_sum) * np.log(np.abs(vol_diff/vol_sum)) * 
                               df['volume_acceleration'].iloc[i] * df['price_acceleration'].iloc[i])
            else:
                volume_regime = 0
            
            # Temporal Quantum Regime
            temporal_regime = (np.abs(df['intraday_efficiency'].iloc[i] - df['intraday_efficiency_prev'].iloc[i]) * 
                             df['momentum_reversal'].iloc[i] * df['volatility_regime'].iloc[i])
            
            asymmetry_regimes = position_regime + volume_regime + temporal_regime
            
            # Regime-Weighted Quantum
            regime_weighted_quantum = enhanced_quantum_efficiency * asymmetry_regimes
            
            # Final Temporal Quantum Alpha
            final_alpha = (regime_weighted_quantum * volume_acceleration * closing_elasticity)
            
            result.iloc[i] = final_alpha
            
        except (ZeroDivisionError, ValueError, KeyError):
            result.iloc[i] = 0
    
    # Replace infinite values and NaN with 0
    result = result.replace([np.inf, -np.inf], 0).fillna(0)
    
    return result
