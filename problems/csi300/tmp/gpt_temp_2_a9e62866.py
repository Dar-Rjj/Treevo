import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    data = df.copy()
    
    # Gap Analysis Framework
    data['gap_size'] = data['open'] / data['close'].shift(1) - 1
    data['gap_direction'] = np.sign(data['gap_size'])
    
    # Historical gap distribution for magnitude classification
    data['gap_abs'] = data['gap_size'].abs()
    data['gap_magnitude_rank'] = data['gap_abs'].rolling(window=50, min_periods=20).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) >= 20 else np.nan, raw=False
    )
    
    # Intraday Gap Reaction Analysis
    data['daily_range'] = data['high'] - data['low']
    data['reaction_strength'] = (data['close'] - data['open']) / np.where(data['daily_range'] == 0, 1e-6, data['daily_range'])
    
    # Gap filling vs continuation
    data['gap_filled'] = np.where(
        (data['gap_direction'] > 0) & (data['low'] <= data['close'].shift(1)), 1,
        np.where((data['gap_direction'] < 0) & (data['high'] >= data['close'].shift(1)), 1, 0)
    )
    
    # Volatility Regime Classification
    # Average True Range calculation
    data['tr1'] = data['high'] - data['low']
    data['tr2'] = (data['high'] - data['close'].shift(1)).abs()
    data['tr3'] = (data['low'] - data['close'].shift(1)).abs()
    data['true_range'] = data[['tr1', 'tr2', 'tr3']].max(axis=1)
    data['atr_10'] = data['true_range'].rolling(window=10, min_periods=5).mean()
    data['atr_50'] = data['true_range'].rolling(window=50, min_periods=25).mean()
    
    # Close-to-Close volatility
    data['returns'] = data['close'].pct_change()
    data['vol_10'] = data['returns'].rolling(window=10, min_periods=5).std()
    data['vol_50'] = data['returns'].rolling(window=50, min_periods=25).std()
    
    # Volatility regime classification
    data['high_vol_regime'] = (data['atr_10'] > data['atr_50'] * 1.2).astype(int)
    data['low_vol_regime'] = (data['vol_10'] < data['vol_50'] * 0.8).astype(int)
    data['vol_regime'] = np.where(data['high_vol_regime'] == 1, 1, np.where(data['low_vol_regime'] == 1, -1, 0))
    
    # Volume-Pressure Accumulation Framework
    data['buy_pressure'] = np.maximum(0, data['close'] - data['open']) * data['volume']
    data['sell_pressure'] = np.maximum(0, data['open'] - data['close']) * data['volume']
    
    data['buy_pressure_5d'] = data['buy_pressure'].rolling(window=5, min_periods=3).sum()
    data['sell_pressure_5d'] = data['sell_pressure'].rolling(window=5, min_periods=3).sum()
    data['pressure_ratio'] = data['buy_pressure_5d'] / (data['sell_pressure_5d'] + 1e-6)
    
    # Volume trend analysis
    data['volume_sma_5'] = data['volume'].rolling(window=5, min_periods=3).mean()
    data['volume_sma_20'] = data['volume'].rolling(window=20, min_periods=10).mean()
    data['volume_trend_ratio'] = data['volume_sma_5'] / (data['volume_sma_20'] + 1e-6)
    data['volume_stress'] = data['volume'] / (data['volume_sma_20'] + 1e-6)
    
    # Volume-Gap Alignment
    data['gap_volume_alignment'] = data['gap_direction'] * np.sign(data['volume_stress'] - 1)
    
    # Regime-Adaptive Gap Processing
    # Momentum calculations
    data['momentum_3d'] = data['close'] / data['close'].shift(3) - 1
    data['momentum_15d'] = data['close'] / data['close'].shift(15) - 1
    
    # Regime-specific momentum weighting
    data['gap_momentum_high_vol'] = data['momentum_3d'] * data['reaction_strength']
    data['gap_momentum_low_vol'] = data['momentum_15d'] * data['reaction_strength']
    
    # Pressure component with volume scaling
    data['scaled_pressure'] = data['pressure_ratio'] * data['volume_trend_ratio']
    
    # Regime-weighted combination
    high_vol_weight = np.where(data['vol_regime'] == 1, 0.7, 0.3)
    low_vol_weight = np.where(data['vol_regime'] == -1, 0.7, 0.3)
    
    data['regime_gap_component'] = (
        high_vol_weight * data['gap_momentum_high_vol'] + 
        low_vol_weight * data['gap_momentum_low_vol']
    )
    
    data['regime_pressure_component'] = (
        high_vol_weight * (2 - data['scaled_pressure']) +  # Mean reversion emphasis in high vol
        low_vol_weight * data['scaled_pressure']  # Momentum emphasis in low vol
    )
    
    # Signal Generation with Multi-Timeframe Confirmation
    # Gap-Pressure Alignment
    data['gap_pressure_alignment'] = data['gap_direction'] * np.sign(data['pressure_ratio'] - 1)
    
    # Multi-timeframe pressure consistency
    data['pressure_ratio_3d'] = data['pressure_ratio'].rolling(window=3, min_periods=2).mean()
    data['pressure_alignment_3d'] = data['gap_direction'] * np.sign(data['pressure_ratio_3d'] - 1)
    
    # Regime-adaptive signal weighting
    volatility_scaling = np.where(data['vol_regime'] == 1, 0.6, np.where(data['vol_regime'] == -1, 1.2, 1.0))
    
    # Final Alpha Synthesis
    data['gap_momentum_factor'] = (
        data['regime_gap_component'] * 0.4 +
        data['regime_pressure_component'] * 0.3 +
        data['gap_pressure_alignment'] * 0.15 +
        data['pressure_alignment_3d'] * 0.15
    ) * volatility_scaling * data['gap_magnitude_rank']
    
    # Apply volume confirmation filter
    volume_confirmation = np.where(data['volume_stress'] > 1.2, 1.2, 
                                  np.where(data['volume_stress'] < 0.8, 0.8, 1.0))
    data['gap_momentum_factor'] *= volume_confirmation
    
    # Penalize gap filling patterns in momentum context
    gap_filling_penalty = np.where(data['gap_filled'] == 1, 0.8, 1.0)
    data['gap_momentum_factor'] *= gap_filling_penalty
    
    return data['gap_momentum_factor']
