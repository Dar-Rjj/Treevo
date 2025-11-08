import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Asymmetric Volatility-Efficiency Framework
    # Volatility Asymmetry Components
    data['upside_vol'] = np.maximum(0, data['high'] - data['close'])
    data['downside_vol'] = np.maximum(0, data['close'] - data['low'])
    
    data['upside_vol_avg'] = data['upside_vol'].rolling(window=10, min_periods=1).mean()
    data['downside_vol_avg'] = data['downside_vol'].rolling(window=10, min_periods=1).mean()
    data['vol_asymmetry_ratio'] = data['upside_vol_avg'] / (data['downside_vol_avg'] + 1e-8)
    
    # Price-Volume Efficiency Analysis
    data['intraday_efficiency'] = (data['close'] - data['low']) / (data['high'] - data['low'] + 1e-8)
    
    # Volume Percentile calculation
    data['volume_rank'] = data['volume'].rolling(window=20, min_periods=1).apply(
        lambda x: (x[-1] - x.min()) / (x.max() - x.min() + 1e-8) if len(x) > 0 else 0.5
    )
    
    data['efficiency_signal'] = data['intraday_efficiency'] * data['volume_rank']
    
    # Asymmetric Efficiency Integration
    data['vol_efficiency_alignment'] = data['vol_asymmetry_ratio'] * data['efficiency_signal']
    data['efficiency_momentum'] = data['efficiency_signal'] - data['efficiency_signal'].shift(3)
    data['core_asymmetric_signal'] = data['vol_efficiency_alignment'] * np.sign(data['efficiency_momentum'])
    
    # Momentum-Pressure Confirmation System
    # Quality Momentum Assessment
    data['daily_return'] = data['close'] / data['close'].shift(1) - 1
    data['return_sign'] = np.sign(data['daily_return'])
    
    # Directional Persistence
    def count_consecutive_signs(series):
        if len(series) < 5:
            return 1
        current_sign = series.iloc[-1]
        count = 1
        for i in range(len(series)-2, -1, -1):
            if series.iloc[i] == current_sign:
                count += 1
            else:
                break
        return count
    
    data['directional_persistence'] = data['return_sign'].rolling(window=5, min_periods=1).apply(
        count_consecutive_signs, raw=False
    )
    
    # Return-to-Volatility
    data['five_day_return'] = data['close'] / data['close'].shift(5) - 1
    data['avg_range_5d'] = (data['high'] - data['low']).rolling(window=5, min_periods=1).mean()
    data['return_to_volatility'] = data['five_day_return'] / (data['avg_range_5d'] + 1e-8)
    
    data['quality_signal'] = data['directional_persistence'] * data['return_to_volatility']
    
    # Microstructure Pressure Analysis
    data['price_rejection'] = (data['high'] - data['close']) / (data['close'] - data['low'] + 1e-8)
    data['volume_concentration'] = (data['high'] - data['low']) / (data['volume'] + 1e-8)
    data['pressure_score'] = data['price_rejection'] * data['volume_concentration']
    
    # Momentum-Pressure Integration
    data['quality_pressure_alignment'] = data['quality_signal'] * data['pressure_score']
    data['confirmation_strength'] = data['quality_pressure_alignment'] * np.sign(data['pressure_score'])
    data['enhanced_core_signal'] = data['core_asymmetric_signal'] * data['confirmation_strength']
    
    # Regime-Adaptive Signal Processing
    # Volatility Regime Classification
    data['recent_range_avg'] = (data['high'] - data['low']).rolling(window=5, min_periods=1).mean()
    data['historical_range_avg'] = (data['high'] - data['low']).rolling(window=20, min_periods=1).mean()
    data['volatility_regime'] = np.where(data['recent_range_avg'] > data['historical_range_avg'], 'High', 'Low')
    
    # Dynamic Component Weighting
    data['efficiency_correlation'] = data['efficiency_signal'].rolling(window=5, min_periods=1).corr(data['quality_signal'])
    data['asymmetry_correlation'] = data['vol_asymmetry_ratio'].rolling(window=5, min_periods=1).corr(data['pressure_score'])
    data['component_weight'] = np.abs(data['efficiency_correlation']) * np.abs(data['asymmetry_correlation'])
    
    # Regime-Specific Processing
    high_vol_mask = data['volatility_regime'] == 'High'
    low_vol_mask = data['volatility_regime'] == 'Low'
    
    data['regime_adapted_signal'] = np.nan
    data.loc[high_vol_mask, 'regime_adapted_signal'] = (
        data.loc[high_vol_mask, 'enhanced_core_signal'] * data.loc[high_vol_mask, 'component_weight']
    )
    
    # Low volatility: smoothed signal
    data['enhanced_core_smoothed'] = data['enhanced_core_signal'].rolling(window=3, min_periods=1).mean()
    data.loc[low_vol_mask, 'regime_adapted_signal'] = (
        data.loc[low_vol_mask, 'enhanced_core_smoothed'] * data.loc[low_vol_mask, 'component_weight']
    )
    
    # Composite Alpha Factor Generation
    # Primary Signal Construction
    data['avg_range_10d'] = (data['high'] - data['low']).rolling(window=10, min_periods=1).mean()
    data['range_normalized_core'] = data['regime_adapted_signal'] / (data['avg_range_10d'] + 1e-8)
    
    # Dynamic Adjustment Components
    data['efficiency_persistence'] = data['efficiency_signal'] - data['efficiency_signal'].shift(3)
    data['asymmetry_trend'] = data['vol_asymmetry_ratio'] - data['vol_asymmetry_ratio'].shift(5)
    data['adjustment_factor'] = np.sign(data['efficiency_persistence']) * np.abs(data['asymmetry_trend'])
    
    # Final Alpha Factor
    data['base_factor'] = data['range_normalized_core'] * data['adjustment_factor']
    
    # Return the final factor series
    return data['base_factor']
