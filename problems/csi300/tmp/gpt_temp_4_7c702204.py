import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate basic price movements
    data['price_change'] = data['close'] - data['open']
    data['prev_close'] = data['close'].shift(1)
    data['day_return'] = data['close'] / data['prev_close'] - 1
    
    # Asymmetric Price Behavior Analysis
    # Upside-Downside Asymmetry
    data['up_intensity'] = np.where(
        data['close'] > data['open'],
        (data['high'] - data['open']) / np.maximum(data['close'] - data['low'], 1e-6),
        0
    )
    data['down_intensity'] = np.where(
        data['close'] < data['open'],
        (data['open'] - data['low']) / np.maximum(data['high'] - data['close'], 1e-6),
        0
    )
    data['asymmetry_ratio'] = np.where(
        data['down_intensity'] > 0,
        data['up_intensity'] / np.maximum(data['down_intensity'], 1e-6),
        1.0
    )
    
    # Gap Asymmetry Patterns
    data['overnight_gap'] = np.abs(data['open'] - data['prev_close']) / np.maximum(
        data['high'].shift(1) - data['low'].shift(1), 1e-6
    )
    data['gap_filling'] = np.where(
        np.abs(data['open'] - data['prev_close']) > 1e-6,
        (data['close'] - data['open']) / np.maximum(data['open'] - data['prev_close'], 1e-6),
        0
    )
    
    # Gap persistence calculation
    gap_sign = np.sign(data['open'] - data['prev_close'])
    close_open_sign = np.sign(data['close'] - data['open'])
    gap_persistence = (gap_sign == close_open_sign).rolling(window=5, min_periods=1).sum()
    data['gap_persistence'] = gap_persistence
    
    # Intraday Reversal Patterns
    data['morning_reversal'] = (data['open'] - data['low']) / np.maximum(data['high'] - data['open'], 1e-6)
    data['afternoon_reversal'] = (data['high'] - data['close']) / np.maximum(data['close'] - data['low'], 1e-6)
    
    # Reversal consistency (rolling correlation)
    reversal_consistency = data['morning_reversal'].rolling(window=5, min_periods=3).corr(data['afternoon_reversal'])
    data['reversal_consistency'] = reversal_consistency.fillna(0)
    
    # Volume Asymmetry Detection
    # Volume-Price Divergence
    up_days = data['close'] > data['prev_close']
    down_days = data['close'] < data['prev_close']
    
    up_volume_5d = up_days.rolling(window=5, min_periods=1).apply(
        lambda x: (data.loc[x.index, 'volume'] * x).sum() / np.maximum(data.loc[x.index, 'volume'].sum(), 1e-6), 
        raw=False
    )
    down_volume_5d = down_days.rolling(window=5, min_periods=1).apply(
        lambda x: (data.loc[x.index, 'volume'] * x).sum() / np.maximum(data.loc[x.index, 'volume'].sum(), 1e-6), 
        raw=False
    )
    
    data['up_volume_concentration'] = up_volume_5d
    data['down_volume_concentration'] = down_volume_5d
    data['volume_divergence'] = data['up_volume_concentration'] - data['down_volume_concentration']
    
    # Volume Spike Asymmetry
    high_volume_up = ((data['close'] > data['prev_close']) & 
                     (data['volume'] > 2 * data['volume'].shift(1))).rolling(window=5, min_periods=1).sum()
    high_volume_down = ((data['close'] < data['prev_close']) & 
                       (data['volume'] > 2 * data['volume'].shift(1))).rolling(window=5, min_periods=1).sum()
    
    data['high_volume_up_days'] = high_volume_up
    data['high_volume_down_days'] = high_volume_down
    data['spike_asymmetry_ratio'] = np.where(
        data['high_volume_down_days'] > 0,
        data['high_volume_up_days'] / np.maximum(data['high_volume_down_days'], 1e-6),
        1.0
    )
    
    # Volume Efficiency Patterns
    data['volume_efficiency_gains'] = np.where(
        data['close'] > data['open'],
        (data['close'] - data['open']) / np.maximum(data['volume'], 1e-6),
        0
    )
    data['volume_efficiency_losses'] = np.where(
        data['close'] < data['open'],
        (data['open'] - data['close']) / np.maximum(data['volume'], 1e-6),
        0
    )
    
    efficiency_gains_5d = data['volume_efficiency_gains'].rolling(window=5, min_periods=3).mean()
    efficiency_losses_5d = data['volume_efficiency_losses'].rolling(window=5, min_periods=3).mean()
    data['efficiency_divergence'] = efficiency_gains_5d - efficiency_losses_5d
    
    # Market Regime Classification
    data['volatility_asymmetry'] = data['asymmetry_ratio'].rolling(window=10, min_periods=5).std()
    data['high_vol_regime'] = data['volatility_asymmetry'] > 0.5
    data['low_vol_regime'] = data['volatility_asymmetry'] < 0.2
    
    # Trend Asymmetry Patterns
    data['strong_uptrend'] = (data['asymmetry_ratio'] > 1.2) & (data['close'] > data['close'].shift(5))
    data['strong_downtrend'] = (data['asymmetry_ratio'] < 0.8) & (data['close'] < data['close'].shift(5))
    data['neutral_trend'] = (data['asymmetry_ratio'] >= 0.9) & (data['asymmetry_ratio'] <= 1.1)
    
    # Volume Asymmetry Regimes
    data['high_volume_div'] = np.abs(data['volume_divergence']) > 0.3
    data['low_volume_div'] = np.abs(data['volume_divergence']) < 0.1
    
    # Asymmetry-Based Signal Construction
    # High Volatility Asymmetry Signals
    data['gap_reversal_signal'] = -data['gap_filling'] * data['gap_persistence']
    data['volume_confirmation'] = data['volume_divergence'] * data['spike_asymmetry_ratio']
    data['intraday_momentum'] = data['morning_reversal'] * data['afternoon_reversal']
    
    # Trend Asymmetry Signals
    data['uptrend_continuation'] = data['asymmetry_ratio'] * efficiency_gains_5d
    data['downtrend_continuation'] = (1 / np.maximum(data['asymmetry_ratio'], 1e-6)) * efficiency_losses_5d
    data['trend_reversal'] = -data['asymmetry_ratio'] * data['efficiency_divergence']
    
    # Volume Asymmetry Signals
    data['volume_exhaustion'] = -data['volume_divergence'] * data['spike_asymmetry_ratio']
    data['accumulation_signal'] = data['up_volume_concentration'] * efficiency_gains_5d
    data['distribution_signal'] = data['down_volume_concentration'] * efficiency_losses_5d
    
    # Cross-Asymmetry Synchronization
    data['price_volume_alignment'] = data['asymmetry_ratio'] * data['volume_divergence']
    data['gap_volume_coordination'] = data['gap_filling'] * data['spike_asymmetry_ratio']
    
    # Multi-timeframe asymmetry correlation
    multi_timeframe_corr = data['asymmetry_ratio'].rolling(window=5, min_periods=3).corr(data['volume_divergence'])
    data['multi_timeframe_asymmetry'] = multi_timeframe_corr.fillna(0)
    
    # Adaptive Alpha Generation
    # Regime-Dependent Signal Blending
    data['high_vol_alpha'] = (data['gap_reversal_signal'] + 
                             data['volume_confirmation'] + 
                             data['intraday_momentum'])
    
    data['trend_asymmetry_alpha'] = (data['uptrend_continuation'] + 
                                   data['downtrend_continuation'] + 
                                   data['trend_reversal'])
    
    data['volume_regime_alpha'] = (data['volume_exhaustion'] + 
                                 data['accumulation_signal'] + 
                                 data['distribution_signal'])
    
    data['synchronization_bonus'] = data['price_volume_alignment'] + data['gap_volume_coordination']
    
    # Dynamic Regime Selection
    def select_regime_alpha(row):
        if row['high_vol_regime']:
            return row['high_vol_alpha']
        elif row['strong_uptrend'] or row['strong_downtrend']:
            return row['trend_asymmetry_alpha']
        elif row['high_volume_div']:
            return row['volume_regime_alpha']
        else:
            # Mixed regime - blend all components
            return (row['high_vol_alpha'] + row['trend_asymmetry_alpha'] + 
                   row['volume_regime_alpha']) / 3
    
    data['selected_regime_alpha'] = data.apply(select_regime_alpha, axis=1)
    
    # Final Alpha Output
    data['primary_alpha'] = data['selected_regime_alpha'] + data['synchronization_bonus']
    data['confirmation_signal'] = data['multi_timeframe_asymmetry'] * data['selected_regime_alpha']
    
    # Risk adjustment
    vol_adjustment = 1 / np.maximum(data['volatility_asymmetry'], 0.1)
    data['final_alpha'] = data['primary_alpha'] * vol_adjustment + 0.3 * data['confirmation_signal']
    
    # Clean up and return
    alpha_series = data['final_alpha'].copy()
    alpha_series = alpha_series.replace([np.inf, -np.inf], np.nan).fillna(method='ffill').fillna(0)
    
    return alpha_series
