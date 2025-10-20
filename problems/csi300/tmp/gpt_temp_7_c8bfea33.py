import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Multi-Timeframe Momentum Efficiency with Volume-Amount-Price Convergence
    """
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Initialize result series
    result = pd.Series(index=df.index, dtype=float)
    
    # Calculate returns
    df['returns'] = df['close'].pct_change()
    
    # Multi-Timeframe Efficiency-Momentum Framework
    for window in [5, 20, 60]:
        df[f'return_{window}d'] = df['close'].pct_change(window)
        df[f'abs_return_sum_{window}d'] = df['returns'].abs().rolling(window).sum()
        df[f'efficiency_{window}d'] = df[f'return_{window}d'] / (df[f'abs_return_sum_{window}d'] + 1e-8)
    
    # Momentum acceleration hierarchy
    df['return_3d'] = df['close'].pct_change(3)
    df['return_8d'] = df['close'].pct_change(8)
    df['return_21d'] = df['close'].pct_change(21)
    
    df['primary_acceleration'] = df['return_3d'] - df['return_8d']
    df['secondary_acceleration'] = df['return_8d'] - df['return_21d']
    
    # Efficiency momentum
    df['efficiency_momentum_5d'] = df['efficiency_5d'] - df['efficiency_5d'].shift(1)
    df['efficiency_momentum_20d'] = df['efficiency_20d'] - df['efficiency_20d'].shift(1)
    
    # Cross-timeframe alignment strength
    df['efficiency_momentum_convergence'] = (
        np.sign(df['efficiency_5d']) * np.sign(df['efficiency_momentum_5d']) +
        np.sign(df['efficiency_20d']) * np.sign(df['efficiency_momentum_20d']) +
        np.sign(df['efficiency_60d']) * np.sign(df['efficiency_momentum_5d'])
    )
    df['acceleration_consistency'] = np.sign(df['primary_acceleration']) * np.sign(df['secondary_acceleration'])
    
    # Volume-Amount-Price Divergence Analysis
    df['volume_momentum_3d'] = (df['volume'] - df['volume'].shift(3)) / (df['volume'].shift(3) + 1e-8)
    df['volume_acceleration'] = df['volume'].pct_change(3) - df['volume'].pct_change(8)
    df['volume_efficiency'] = df['returns'] / (df['volume'] + 1e-8)
    
    # Amount-based order flow dynamics
    df['amount_ma_5d'] = df['amount'].rolling(5).mean()
    df['amount_concentration'] = df['amount'] / (df['amount_ma_5d'] + 1e-8)
    df['order_flow_efficiency'] = df['returns'] / (df['amount'] + 1e-8)
    df['amount_volume_divergence'] = df['amount'].pct_change(3) - df['volume'].pct_change(3)
    
    # Price efficiency divergence detection
    df['price_per_volume'] = (df['close'] - df['close'].shift(1)) / (df['volume'] + 1e-8)
    df['efficiency_trend'] = df['efficiency_5d'] - df['efficiency_20d']
    df['volume_price_divergence'] = np.abs(df['primary_acceleration']) / (np.abs(df['volume_acceleration']) + 0.001)
    
    # Range Dynamics & Volatility Regime Integration
    df['daily_range'] = df['high'] - df['low']
    df['range_utilization'] = (df['close'] - df['open']) / (df['daily_range'] + 1e-8)
    df['range_ma_5d'] = df['daily_range'].rolling(5).mean()
    df['range_ma_20d'] = df['daily_range'].rolling(20).mean()
    df['range_compression'] = df['range_ma_5d'] / (df['range_ma_20d'] + 1e-8)
    
    # True Range calculation
    df['tr1'] = df['high'] - df['low']
    df['tr2'] = np.abs(df['high'] - df['close'].shift(1))
    df['tr3'] = np.abs(df['low'] - df['close'].shift(1))
    df['true_range'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
    df['atr_14d'] = df['true_range'].rolling(14).mean()
    df['atr_20d'] = df['true_range'].rolling(20).mean()
    df['atr_60d'] = df['true_range'].rolling(60).mean()
    
    # Return volatility
    df['return_vol_20d'] = df['returns'].rolling(20).std()
    
    # Bollinger Bands
    df['bb_middle'] = df['close'].rolling(20).mean()
    df['bb_std'] = df['close'].rolling(20).std()
    df['bb_upper'] = df['bb_middle'] + 2 * df['bb_std']
    df['bb_lower'] = df['bb_middle'] - 2 * df['bb_std']
    df['bb_width'] = df['bb_upper'] - df['bb_lower']
    df['squeeze_intensity'] = 1 / (df['bb_width'] + 0.001)
    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-8)
    
    # Volatility regime classification
    df['volatility_regime_ratio'] = df['atr_20d'] / (df['atr_60d'] + 1e-8)
    
    # Define regime thresholds
    high_vol_threshold = df['volatility_regime_ratio'].quantile(0.7)
    low_vol_threshold = df['volatility_regime_ratio'].quantile(0.3)
    
    # Regime-Adaptive Convergence Framework
    for idx in df.index:
        if pd.isna(df.loc[idx, 'volatility_regime_ratio']):
            continue
            
        if df.loc[idx, 'volatility_regime_ratio'] > high_vol_threshold:
            # High Volatility Regime
            regime_signal = (
                df.loc[idx, 'efficiency_trend'] * 0.4 +
                (np.sign(df.loc[idx, 'efficiency_5d']) * df.loc[idx, 'amount_concentration'] * df.loc[idx, 'acceleration_consistency']) * 0.3 +
                df.loc[idx, 'range_utilization'] * 0.2 +
                df.loc[idx, 'primary_acceleration'] * 0.1
            )
            
        elif df.loc[idx, 'volatility_regime_ratio'] < low_vol_threshold:
            # Low Volatility Regime
            regime_signal = (
                df.loc[idx, 'primary_acceleration'] * 0.4 +
                df.loc[idx, 'efficiency_trend'] * 0.3 +
                df.loc[idx, 'acceleration_consistency'] * 0.2 +
                (1 / (df.loc[idx, 'range_compression'] + 0.001)) * 0.1
            )
            
        else:
            # Transition Regime
            regime_signal = (
                df.loc[idx, 'primary_acceleration'] * df.loc[idx, 'efficiency_trend'] * 0.3 +
                df.loc[idx, 'volume_acceleration'] * df.loc[idx, 'amount_concentration'] * 0.3 +
                df.loc[idx, 'range_utilization'] * df.loc[idx, 'range_compression'] * 0.2 +
                df.loc[idx, 'efficiency_momentum_convergence'] * 0.2
            )
        
        # Volatility-Adaptive Enhancement
        if df.loc[idx, 'volatility_regime_ratio'] > high_vol_threshold:
            enhanced_signal = regime_signal / (df.loc[idx, 'atr_14d'] + 0.001)
        elif df.loc[idx, 'volatility_regime_ratio'] < low_vol_threshold:
            enhanced_signal = regime_signal * (1 + 1/(df.loc[idx, 'range_compression'] + 0.001))
        else:
            enhanced_signal = regime_signal * df.loc[idx, 'volatility_regime_ratio']
        
        # Bollinger Band integration
        amplified_signal = enhanced_signal * df.loc[idx, 'squeeze_intensity']
        position_adjusted = amplified_signal * (1 - np.abs(df.loc[idx, 'bb_position'] - 0.5))
        final_signal = position_adjusted * (1 + df.loc[idx, 'range_utilization'])
        
        # Multi-Dimensional Pattern Recognition
        cross_factor_alignment = (
            np.sign(df.loc[idx, 'efficiency_5d']) * np.sign(df.loc[idx, 'primary_acceleration']) +
            np.sign(df.loc[idx, 'volume_acceleration']) * np.sign(df.loc[idx, 'amount_concentration']) * np.sign(df.loc[idx, 'efficiency_trend']) +
            df.loc[idx, 'range_utilization'] * np.sign(df.loc[idx, 'efficiency_5d'])
        )
        
        convergence_strength = (
            df.loc[idx, 'primary_acceleration'] * df.loc[idx, 'secondary_acceleration'] +
            cross_factor_alignment * df.loc[idx, 'acceleration_consistency'] +
            cross_factor_alignment * regime_signal
        )
        
        # Composite Alpha Synthesis
        alpha_value = final_signal * (1 + convergence_strength * 0.1)
        
        result.loc[idx] = alpha_value
    
    # Fill NaN values with 0
    result = result.fillna(0)
    
    return result
