import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Chaotic Momentum-Volatility Resonance with Fractal Regime Detection
    """
    # Copy dataframe to avoid modifying original
    data = df.copy()
    
    # Calculate basic price changes and ranges
    data['prev_close'] = data['close'].shift(1)
    data['prev_high'] = data['high'].shift(1)
    data['prev_low'] = data['low'].shift(1)
    data['prev_volume'] = data['volume'].shift(1)
    
    # 1. Multi-Timeframe Momentum Divergence
    # Short-term momentum acceleration
    data['mom_accel_st'] = (data['close'] / data['close'].shift(2) - 1) - (data['close'].shift(1) / data['close'].shift(3) - 1)
    
    # Medium-term momentum persistence
    data['ret_3d'] = data['close'] / data['close'].shift(3) - 1
    data['ret_5d'] = data['close'] / data['close'].shift(5) - 1
    data['mom_persistence_mt'] = np.sign(data['ret_5d']) * np.sign(data['ret_3d']) * np.abs(data['ret_5d'])
    
    # Multi-timeframe momentum signal
    data['momentum_signal'] = data['mom_accel_st'] * data['mom_persistence_mt']
    
    # 2. Volatility Regime Resonance
    # Daily range
    data['daily_range'] = data['high'] - data['low']
    data['avg_range_5d'] = data['daily_range'].rolling(window=5).mean()
    data['avg_volume_5d'] = data['volume'].rolling(window=5).mean()
    
    # Chaotic volatility clustering
    data['vol_clustering'] = (data['daily_range'] / data['avg_range_5d']) * (data['volume'] / data['avg_volume_5d'])
    
    # ATR calculation
    data['tr'] = np.maximum(data['high'] - data['low'], 
                           np.maximum(np.abs(data['high'] - data['close'].shift(1)), 
                                     np.abs(data['low'] - data['close'].shift(1))))
    data['atr_10'] = data['tr'].rolling(window=10).mean()
    data['fractal_vol_persistence'] = data['daily_range'] / data['atr_10']
    
    # Volatility resonance signal
    data['volatility_signal'] = data['vol_clustering'] * data['fractal_vol_persistence']
    
    # 3. Price-Volume Fractal Alignment
    # Multi-scale volume momentum
    data['volume_momentum'] = (data['volume'] / data['volume'].shift(1)) - (data['volume'].shift(1) / data['volume'].shift(2))
    
    # Fractal price efficiency across multiple periods
    data['price_efficiency_3d'] = np.abs(data['close'] - data['open']) / (data['high'] - data['low'])
    data['price_efficiency_5d'] = np.abs(data['close'].rolling(window=5).mean() - data['open'].rolling(window=5).mean()) / \
                                 (data['high'].rolling(window=5).max() - data['low'].rolling(window=5).min())
    data['price_efficiency_8d'] = np.abs(data['close'].rolling(window=8).mean() - data['open'].rolling(window=8).mean()) / \
                                 (data['high'].rolling(window=8).max() - data['low'].rolling(window=8).min())
    
    # Combined price efficiency
    data['fractal_efficiency'] = (data['price_efficiency_3d'] + data['price_efficiency_5d'] + data['price_efficiency_8d']) / 3
    
    # Price-volume alignment signal
    data['price_volume_signal'] = data['volume_momentum'] * data['fractal_efficiency']
    
    # 4. Chaotic Breakout Detection
    data['high_15d'] = data['high'].rolling(window=15).max()
    data['atr_5'] = data['tr'].rolling(window=5).mean()
    
    # Fractal breakout strength
    data['breakout_strength'] = np.abs(data['close'] - data['high_15d']) / data['atr_10']
    
    # Chaotic confirmation
    data['chaotic_confirmation'] = (data['close'] - data['open']) * (data['high'] - data['low']) / (data['prev_close'] * data['atr_5'])
    
    # Breakout signal
    data['breakout_signal'] = data['breakout_strength'] * data['chaotic_confirmation']
    
    # 5. Multi-Timeframe Regime Switching
    data['volume_change_2d'] = data['volume'] / data['volume'].shift(2) - 1
    data['volume_trend_5d'] = data['volume'].rolling(window=5).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0], raw=True)
    
    # Regime signals
    data['regime_st'] = np.sign(data['ret_3d']) * np.sign(data['volume_change_2d'])
    data['regime_mt'] = np.sign(data['ret_5d']) * np.sign(data['volume_trend_5d'])
    
    # Regime alignment with persistence
    regime_alignment = data['regime_st'] * data['regime_mt']
    regime_persistence = regime_alignment.rolling(window=3).mean()
    data['regime_signal'] = regime_alignment * regime_persistence
    
    # 6. Fractal Momentum Exhaustion
    data['price_change_t'] = data['close'] / data['prev_close'] - 1
    data['price_change_t1'] = data['prev_close'] / data['close'].shift(2) - 1
    
    # Momentum decay
    data['momentum_decay'] = data['price_change_t'] / data['price_change_t1']
    
    # Volume support weakening
    data['volume_support'] = (data['volume'] / data['prev_volume']) / np.abs(data['price_change_t'] / data['price_change_t1'])
    
    # Exhaustion signal
    data['exhaustion_signal'] = data['momentum_decay'] * data['volume_support']
    
    # 7. Chaotic Volatility Expansion
    data['intraday_vol_spike'] = data['daily_range'] / data['open'] / data['avg_range_5d']
    data['interday_vol_persistence'] = data['atr_5'] / data['atr_10']
    
    # Volatility expansion signal
    data['vol_expansion_signal'] = data['intraday_vol_spike'] * data['interday_vol_persistence']
    
    # 8. Fractal Price Efficiency Patterns
    data['price_efficiency_2d'] = np.abs(data['close'] - data['open']) / (data['high'] - data['low'])
    data['price_efficiency_4d'] = np.abs(data['close'].rolling(window=4).mean() - data['open'].rolling(window=4).mean()) / \
                                 (data['high'].rolling(window=4).max() - data['low'].rolling(window=4).min())
    data['price_efficiency_6d'] = np.abs(data['close'].rolling(window=6).mean() - data['open'].rolling(window=6).mean()) / \
                                 (data['high'].rolling(window=6).max() - data['low'].rolling(window=6).min())
    
    # Multi-scale efficiency ratio
    data['multi_scale_efficiency'] = (data['price_efficiency_2d'] + data['price_efficiency_4d'] + data['price_efficiency_6d']) / 3
    data['efficiency_momentum'] = data['multi_scale_efficiency'] / data['multi_scale_efficiency'].rolling(window=3).mean()
    
    # Efficiency signal
    data['efficiency_signal'] = data['multi_scale_efficiency'] * data['efficiency_momentum']
    
    # 9. Chaotic Regime Transition Scoring
    data['vol_momentum_transition'] = data['vol_clustering'] * data['mom_accel_st']
    data['efficiency_regime_shift'] = data['fractal_efficiency'] * data['volume_momentum']
    
    # Regime transition score
    data['regime_transition_score'] = data['vol_momentum_transition'] * data['efficiency_regime_shift']
    
    # 10. Composite Chaotic-Fractal Alpha
    # Core components
    core_momentum_vol = data['momentum_signal'] * data['volatility_signal']
    fractal_efficiency = data['price_volume_signal'] * data['efficiency_signal']
    chaotic_regime = data['regime_signal'] * data['regime_transition_score']
    
    # Enhanced signals during volatility expansion
    vol_expansion_enhancement = 1 + np.tanh(data['vol_expansion_signal'])
    
    # Momentum acceleration in high efficiency regimes
    efficiency_enhancement = 1 + np.tanh(data['fractal_efficiency'] * data['mom_accel_st'])
    
    # Volume confirmation
    volume_confirmation = np.tanh(data['volume_momentum'])
    
    # Multi-timeframe alignment
    timeframe_alignment = np.tanh(data['regime_signal'])
    
    # Final composite signal
    chaotic_fractal_alpha = (
        core_momentum_vol * vol_expansion_enhancement +
        fractal_efficiency * efficiency_enhancement +
        chaotic_regime * volume_confirmation * timeframe_alignment -
        data['exhaustion_signal'] -  # Subtract exhaustion signals
        data['breakout_signal'] * 0.5  # Moderate breakout influence
    )
    
    # Normalize and return
    alpha_series = chaotic_fractal_alpha.fillna(0)
    alpha_series = alpha_series.replace([np.inf, -np.inf], 0)
    
    return alpha_series
