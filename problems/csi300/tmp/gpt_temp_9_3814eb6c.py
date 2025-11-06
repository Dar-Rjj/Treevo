import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Multi-Scale Asymmetric Momentum-Pressure Framework
    Generates regime-adaptive alpha factor using asymmetric momentum dynamics and pressure validation
    """
    data = df.copy()
    
    # Ensure data is sorted by date
    data = data.sort_index()
    
    # Asymmetric Momentum Dynamics
    # Micro-scale (3-day) Asymmetry
    data['micro_upside'] = (data['high'] - data['open']) / (data['open'] - data['low']).replace(0, np.nan)
    data['micro_downside'] = (data['close'] - data['low']) / (data['high'] - data['close']).replace(0, np.nan)
    data['micro_acceleration'] = (data['close'] / data['close'].shift(1) - 1) - (data['close'].shift(1) / data['close'].shift(2) - 1)
    data['micro_asymmetry'] = data['micro_upside'] - data['micro_downside'] + data['micro_acceleration']
    
    # Meso-scale (8-day) Persistence
    data['close_ret'] = data['close'] / data['close'].shift(1) - 1
    data['up_days'] = data['close_ret'].rolling(window=8, min_periods=1).apply(lambda x: (x > 0).sum(), raw=True)
    data['down_days'] = data['close_ret'].rolling(window=8, min_periods=1).apply(lambda x: (x < 0).sum(), raw=True)
    data['meso_directional'] = data['up_days'] - data['down_days']
    
    data['meso_range'] = (data['close'] - data['low'].rolling(window=8, min_periods=1).min()) / \
                        (data['high'].rolling(window=8, min_periods=1).max() - data['low'].rolling(window=8, min_periods=1).min()).replace(0, np.nan)
    
    data['meso_efficiency'] = (data['close'] - data['close'].shift(8)) / \
                             data['close_ret'].abs().rolling(window=8, min_periods=1).sum().replace(0, np.nan)
    data['meso_persistence'] = data['meso_directional'] + data['meso_range'] + data['meso_efficiency']
    
    # Macro-scale (21-day) Regime
    data['macro_trend'] = (data['close'] - data['close'].shift(21)) / \
                         data['close_ret'].abs().rolling(window=21, min_periods=1).sum().replace(0, np.nan)
    
    data['macro_vol_cluster'] = (data['high'] - data['low']).rolling(window=21, min_periods=1).std() / \
                               (data['high'] - data['low']).rolling(window=21, min_periods=1).mean().replace(0, np.nan)
    
    def rolling_corr(x):
        if len(x) < 21:
            return np.nan
        return pd.Series(x).corr(pd.Series(x).shift(1))
    
    data['macro_memory'] = data['close_ret'].rolling(window=21, min_periods=21).apply(rolling_corr, raw=False)
    data['macro_regime'] = data['macro_trend'] - data['macro_vol_cluster'] + data['macro_memory'].fillna(0)
    
    # Volume Momentum Integration
    data['ultra_short_vol_mom'] = data['volume'] / data['volume'].shift(1).replace(0, np.nan) - 1
    data['short_vol_mom'] = data['volume'] / data['volume'].shift(3).replace(0, np.nan) - 1
    data['medium_vol_mom'] = data['volume'] / data['volume'].shift(8).replace(0, np.nan) - 1
    data['long_vol_mom'] = data['volume'] / data['volume'].shift(21).replace(0, np.nan) - 1
    
    # Pressure Dynamics Framework
    # Opening Pressure Analysis
    data['gap_pressure'] = (data['open'] - data['close'].shift(1)) / data['close'].shift(1).replace(0, np.nan)
    data['gap_absorption'] = (data['close'] - data['open']) / (data['open'] - data['close'].shift(1)).abs().replace(0, np.nan)
    data['opening_efficiency'] = np.sign(data['close'] - data['open']) * (data['close'] - data['open']).abs() / \
                                (data['high'] - data['low']).replace(0, np.nan)
    
    # Intraday Pressure Evolution
    data['morning_pressure'] = (data['high'] - data['open']) / (data['high'] - data['low']).replace(0, np.nan)
    data['afternoon_pressure'] = (data['close'] - data['low']) / (data['high'] - data['low']).replace(0, np.nan)
    data['pressure_consistency'] = np.sign(data['morning_pressure'] - 0.5) * np.sign(data['afternoon_pressure'] - 0.5)
    
    # Volume-Pressure Validation
    data['micro_pressure'] = data['volume'] / data['volume'].rolling(window=3, min_periods=1).mean().replace(0, np.nan)
    data['meso_pressure'] = data['volume'] / data['volume'].rolling(window=8, min_periods=1).mean().replace(0, np.nan)
    data['macro_pressure'] = data['volume'] / data['volume'].rolling(window=21, min_periods=1).mean().replace(0, np.nan)
    
    data['volume_intensity'] = data['volume'] / (data['high'] - data['low']).replace(0, np.nan)
    data['price_compression'] = (data['high'] - data['low']) / data['close'].shift(1).replace(0, np.nan)
    data['pressure_index'] = data['volume_intensity'] * data['price_compression']
    
    # Regime Classification System
    # Price Efficiency Regime
    data['price_efficiency'] = (data['close'] - data['open']) / (data['high'] - data['low']).replace(0, np.nan)
    data['eff_volatility'] = data['price_efficiency'].rolling(window=5, min_periods=1).std()
    data['eff_vol_benchmark'] = data['price_efficiency'].rolling(window=20, min_periods=1).std()
    
    conditions = [
        data['eff_volatility'] > 2 * data['eff_vol_benchmark'],
        data['eff_volatility'] < 0.5 * data['eff_vol_benchmark']
    ]
    choices = [2, 0]  # 2: High Volatility, 0: Low Volatility, 1: Normal Volatility
    data['volatility_regime'] = np.select(conditions, choices, default=1)
    
    # Volume Regime Classification
    data['volume_percentile'] = data['volume'].rolling(window=21, min_periods=1).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )
    conditions_vol = [
        data['volume_percentile'] > 0.75,
        data['volume_percentile'] < 0.25
    ]
    choices_vol = [2, 0]  # 2: High Pressure, 0: Low Pressure, 1: Normal Pressure
    data['volume_regime'] = np.select(conditions_vol, choices_vol, default=1)
    
    # Asymmetric Alignment Detection
    data['micro_meso_align'] = np.sign(data['micro_asymmetry']) * np.sign(data['meso_persistence'])
    data['meso_macro_align'] = np.sign(data['meso_persistence']) * np.sign(data['macro_regime'])
    data['full_align'] = (data['micro_meso_align'] > 0) & (data['meso_macro_align'] > 0)
    
    # Convergence-Divergence Analysis
    # Momentum-Pressure Convergence
    data['ultra_short_conv'] = data['micro_asymmetry'] * data['opening_efficiency']
    data['short_term_conv'] = data['meso_persistence'] * data['pressure_consistency']
    data['medium_term_conv'] = data['macro_regime'] * data['gap_absorption']
    data['long_term_conv'] = data['macro_trend'] * (data['morning_pressure'] + data['afternoon_pressure'] - 1)
    
    # Volume-Price Convergence
    data['ultra_short_vol_price'] = data['micro_asymmetry'] * data['ultra_short_vol_mom']
    data['short_vol_price'] = data['meso_persistence'] * data['short_vol_mom']
    data['medium_vol_price'] = data['macro_regime'] * data['medium_vol_mom']
    data['long_vol_price'] = data['macro_trend'] * data['long_vol_mom']
    
    # Asymmetric Convergence Patterns
    data['price_convergence'] = (data['close'] - data['close'].shift(1)) / (data['close'].shift(1) - data['close'].shift(2)).replace(0, np.nan)
    data['volume_convergence'] = (data['volume'] - data['volume'].shift(1)) / (data['volume'].shift(1) - data['volume'].shift(2)).replace(0, np.nan)
    data['convergence_ratio'] = data['price_convergence'] * data['volume_convergence']
    
    # Composite Alpha Generation with Regime-Adaptive Weighting
    def regime_adaptive_weighting(row):
        if row['volatility_regime'] == 2:  # High Volatility
            micro_weight, meso_weight, macro_weight = 0.5, 0.3, 0.2
            conv_weights = [0.4, 0.3, 0.2, 0.1]  # Emphasize ultra-short
            vol_confirmation = 1.5 if row['volume_regime'] == 2 else 0.8
        elif row['volatility_regime'] == 0:  # Low Volatility
            micro_weight, meso_weight, macro_weight = 0.2, 0.3, 0.5
            conv_weights = [0.1, 0.2, 0.3, 0.4]  # Emphasize long-term
            vol_confirmation = 0.8 if row['volume_regime'] == 0 else 1.2
        else:  # Normal Volatility
            micro_weight, meso_weight, macro_weight = 0.33, 0.33, 0.34
            conv_weights = [0.25, 0.25, 0.25, 0.25]
            vol_confirmation = 1.0
        
        # Base asymmetric momentum
        base_momentum = (micro_weight * row['micro_asymmetry'] + 
                        meso_weight * row['meso_persistence'] + 
                        macro_weight * row['macro_regime'])
        
        # Convergence enhancement
        convergence = (conv_weights[0] * row['ultra_short_conv'] +
                     conv_weights[1] * row['short_term_conv'] +
                     conv_weights[2] * row['medium_term_conv'] +
                     conv_weights[3] * row['long_term_conv'])
        
        # Volume confirmation
        volume_align = (np.sign(row['ultra_short_vol_mom']) * row['pressure_consistency'] * 
                       np.sign(row['micro_asymmetry']) * row['ultra_short_vol_mom'])
        
        # Final alpha with regime adaptation
        alpha = base_momentum * (1 + convergence) * vol_confirmation * (1 + 0.2 * volume_align)
        
        # Apply alignment multiplier
        if row['full_align']:
            alpha *= 1.2
        
        return alpha
    
    # Calculate final alpha factor
    alpha_values = data.apply(regime_adaptive_weighting, axis=1)
    
    # Clean and return the alpha series
    alpha_series = alpha_values.replace([np.inf, -np.inf], np.nan).fillna(0)
    alpha_series = alpha_series.clip(lower=-3, upper=3)  # Reasonable bounds
    
    return alpha_series
