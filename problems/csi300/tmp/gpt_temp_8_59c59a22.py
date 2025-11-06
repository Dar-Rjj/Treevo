import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate True Range
    df['prev_close'] = df['close'].shift(1)
    df['tr1'] = df['high'] - df['low']
    df['tr2'] = abs(df['high'] - df['prev_close'])
    df['tr3'] = abs(df['low'] - df['prev_close'])
    df['true_range'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
    
    # Multi-Timeframe Fractal Efficiency Analysis
    # Short-term (3-day) Fractal Efficiency
    df['fe_3d'] = abs(df['close'] - df['close'].shift(3)) / df['true_range'].rolling(window=3, min_periods=1).sum()
    
    # Medium-term (10-day) Fractal Efficiency
    df['fe_10d'] = abs(df['close'] - df['close'].shift(10)) / df['true_range'].rolling(window=10, min_periods=1).sum()
    
    # Long-term (20-day) Fractal Efficiency
    df['fe_20d'] = abs(df['close'] - df['close'].shift(20)) / df['true_range'].rolling(window=20, min_periods=1).sum()
    
    # Efficiency Momentum Divergence
    df['short_med_momentum'] = (df['fe_10d'] - df['fe_3d']) * np.sign(df['close'] - df['close'].shift(3))
    df['med_long_momentum'] = (df['fe_20d'] - df['fe_10d']) * np.sign(df['close'] - df['close'].shift(10))
    df['efficiency_divergence'] = df['short_med_momentum'] - df['med_long_momentum']
    
    # Microstructure Pressure Convergence
    df['vw_anchor'] = (df['high'] * df['volume'] + df['low'] * df['volume']) / (2 * df['volume'])
    df['upside_pressure'] = (df['high'] - df['vw_anchor']) * df['volume']
    df['downside_pressure'] = (df['vw_anchor'] - df['low']) * df['volume']
    
    df['sum_upside_5d'] = df['upside_pressure'].rolling(window=5, min_periods=1).sum()
    df['sum_downside_5d'] = df['downside_pressure'].rolling(window=5, min_periods=1).sum()
    df['pressure_asymmetry_5d'] = np.log(1 + df['sum_upside_5d']) - np.log(1 + df['sum_downside_5d'])
    df['pressure_convergence_signal'] = df['pressure_asymmetry_5d'] * np.sign((df['close'] - df['open']) * df['volume'])
    
    # Volume Flow and Concentration
    df['upward_pressure_vol'] = np.where(df['close'] > df['vw_anchor'], df['volume'], 0)
    df['downward_pressure_vol'] = np.where(df['close'] < df['vw_anchor'], df['volume'], 0)
    
    df['sum_upward_vol_3d'] = df['upward_pressure_vol'].rolling(window=3, min_periods=1).sum()
    df['sum_downward_vol_3d'] = df['downward_pressure_vol'].rolling(window=3, min_periods=1).sum()
    df['volume_flow_ratio'] = np.log(1 + df['sum_upward_vol_3d']) - np.log(1 + df['sum_downward_vol_3d'])
    
    df['max_vol_5d'] = df['volume'].rolling(window=5, min_periods=1).max()
    df['sum_vol_5d'] = df['volume'].rolling(window=5, min_periods=1).sum()
    df['ma_vol_20d'] = df['volume'].rolling(window=20, min_periods=1).mean()
    df['volume_quality_score'] = (df['max_vol_5d'] / df['sum_vol_5d']) * (df['volume'] / df['ma_vol_20d'])
    
    # Adaptive Signal Synthesis
    df['efficiency_momentum_signal'] = df['efficiency_divergence'] * df['volume_quality_score']
    df['pressure_convergence_signal_2'] = df['pressure_asymmetry_5d'] * df['volume_flow_ratio']
    
    # Volatility regime detection
    df['volatility'] = df['true_range'].rolling(window=20, min_periods=1).std()
    df['volatility_quantile'] = df['volatility'].rolling(window=60, min_periods=1).apply(
        lambda x: pd.qcut(x, q=3, labels=False, duplicates='drop').iloc[-1] if len(x) >= 3 else 1, raw=False
    )
    
    # Regime-Adaptive Weighting
    df['sum_upside_3d'] = df['upside_pressure'].rolling(window=3, min_periods=1).sum()
    df['sum_downside_3d'] = df['downside_pressure'].rolling(window=3, min_periods=1).sum()
    
    df['regime_signal'] = np.where(
        df['volatility_quantile'] == 2,  # High volatility
        df['efficiency_momentum_signal'] * df['pressure_convergence_signal_2'],
        np.where(
            df['volatility_quantile'] == 0,  # Low volatility
            ((df['close'] - df['open']) / (df['high'] - df['low'] + 1e-8)) * 
            (np.log(1 + df['sum_upside_3d']) - np.log(1 + df['sum_downside_3d'])),
            (df['efficiency_momentum_signal'] + df['pressure_convergence_signal_2']) / 2 * df['volume_quality_score']  # Normal volatility
        )
    )
    
    # Quality-Enhanced Signal
    df['pressure_asymmetry_pos_count'] = (df['pressure_asymmetry_5d'] > 0).rolling(window=5, min_periods=1).sum()
    df['quality_enhanced_signal'] = (
        df['regime_signal'] * 
        abs(df['pressure_asymmetry_5d']) * 
        df['pressure_asymmetry_pos_count'] * 
        np.sign((df['close'] - df['open']) * df['volume'])
    )
    
    # Final Alpha Output
    df['alpha'] = (
        df['quality_enhanced_signal'] * 
        df['efficiency_divergence'] * 
        df['volume_quality_score'] * 
        df['pressure_convergence_signal']
    )
    
    # Clean up intermediate columns
    result = df['alpha'].copy()
    
    return result
