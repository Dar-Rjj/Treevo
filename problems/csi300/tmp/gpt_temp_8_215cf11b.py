import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Generate novel alpha factors using momentum-efficiency convergence, pressure-volume breakout confidence,
    volatility-regime adaptive reversal, multi-timeframe pressure diffusion, efficiency-modified momentum persistence,
    volume-spike regime detection, liquidity-enhanced breakout, and momentum-volatility alignment.
    """
    data = df.copy()
    
    # Initialize result series
    result = pd.Series(index=data.index, dtype=float)
    
    # Calculate basic technical indicators
    data['returns_3d'] = data['close'] / data['close'].shift(3) - 1
    data['returns_10d'] = data['close'] / data['close'].shift(10) - 1
    data['momentum_divergence'] = data['returns_3d'] / data['returns_10d']
    
    # Efficiency metrics
    data['movement_efficiency'] = np.abs(data['close'] - data['open']) / (data['high'] - data['low'])
    data['movement_efficiency'] = data['movement_efficiency'].replace([np.inf, -np.inf], np.nan)
    
    # Trend efficiency (5-day)
    data['price_change'] = data['close'] - data['close'].shift(1)
    data['daily_range'] = data['high'] - data['low']
    data['trend_efficiency'] = data['price_change'].rolling(window=5).sum() / data['daily_range'].rolling(window=5).sum()
    
    data['efficiency_momentum'] = data['movement_efficiency'] / data['movement_efficiency'].rolling(window=5).mean()
    
    # Volume metrics
    data['volume_5d_avg'] = data['volume'].rolling(window=5).mean()
    data['volume_concentration'] = data['volume'] / data['volume_5d_avg']
    
    # Momentum-Efficiency Convergence
    data['raw_convergence'] = data['momentum_divergence'] * data['efficiency_momentum']
    data['momentum_efficiency_signal'] = data['raw_convergence'] * data['volume_concentration']
    
    # Pressure-Volume Breakout Confidence
    data['opening_pressure'] = data['open'] / data['close'].shift(1) - 1
    data['closing_pressure'] = (data['close'] - data['low']) / (data['high'] - data['low'])
    data['daily_pressure_score'] = data['opening_pressure'] * data['closing_pressure']
    
    # Volume clustering
    data['volume_3d_corr'] = data['volume'].rolling(window=3).corr(data['volume'].shift(1))
    data['cluster_strength'] = data['volume_concentration'] * data['volume_3d_corr']
    
    # Breakout conditions
    data['high_10d'] = data['high'].rolling(window=10).max()
    data['low_10d'] = data['low'].rolling(window=10).min()
    data['range_position'] = (data['close'] - data['low_10d']) / (data['high_10d'] - data['low_10d'])
    data['pressure_volume_alignment'] = data['daily_pressure_score'] * data['cluster_strength']
    data['breakout_confidence'] = data['range_position'] * data['pressure_volume_alignment']
    
    # Volatility-Regime Adaptive Reversal
    data['atr_5d'] = (data['high'] - data['low']).rolling(window=5).mean()
    data['atr_20d'] = (data['high'] - data['low']).rolling(window=20).mean()
    data['volatility_regime'] = data['atr_5d'] / data['atr_20d']
    
    data['momentum_3d_dir'] = np.sign(data['returns_3d'])
    data['momentum_10d_dir'] = np.sign(data['returns_10d'])
    data['momentum_regime'] = data['momentum_3d_dir'] * data['momentum_10d_dir']
    data['regime_alignment'] = data['volatility_regime'] * data['momentum_regime']
    
    # Reversal setup
    data['price_extreme'] = np.minimum(data['close'] - data['low_10d'], data['high_10d'] - data['close']) / (data['high_10d'] - data['low_10d'])
    data['efficiency_drop'] = data['movement_efficiency'] / data['movement_efficiency'].rolling(window=5).mean()
    data['reversal_potential'] = data['price_extreme'] * data['efficiency_drop']
    data['regime_weighted_reversal'] = data['reversal_potential'] / data['regime_alignment']
    data['volatility_reversal_signal'] = data['regime_weighted_reversal'] * data['volume_concentration']
    
    # Multi-Timeframe Pressure Diffusion
    data['intraday_pressure'] = (data['close'] - data['open']) / (data['high'] - data['low'])
    data['short_term_pressure'] = data['returns_3d']
    data['medium_term_pressure'] = data['returns_10d']
    
    # Calculate pressure diffusion (standard deviation across timeframes)
    pressures = pd.DataFrame({
        'intraday': data['intraday_pressure'],
        'short_term': data['short_term_pressure'],
        'medium_term': data['medium_term_pressure']
    })
    data['pressure_diffusion'] = pressures.std(axis=1)
    
    # Volume alignment
    data['volume_3d_slope'] = data['volume'].rolling(window=3).apply(lambda x: (x.iloc[-1] - x.iloc[0]) / len(x) if len(x) == 3 else np.nan)
    data['volume_10d_slope'] = data['volume'].rolling(window=10).apply(lambda x: (x.iloc[-1] - x.iloc[0]) / len(x) if len(x) == 10 else np.nan)
    data['multi_timeframe_volume'] = data['volume_3d_slope'] * data['volume_10d_slope']
    
    # Historical volume-pressure correlation (20-day rolling)
    data['volume_pressure_corr'] = data['volume'].rolling(window=20).corr(data['intraday_pressure'])
    data['alignment_score'] = data['pressure_diffusion'] * data['volume_pressure_corr']
    
    data['concentrated_pressure'] = 1 / (data['pressure_diffusion'] + 1e-8)
    data['volume_validated_signal'] = data['concentrated_pressure'] * data['alignment_score']
    data['multi_timeframe_signal'] = data['volume_validated_signal'].rolling(window=3).mean()
    
    # Efficiency-Modified Momentum Persistence
    data['range_utilization'] = (data['close'] - data['low']) / (data['high'] - data['low'])
    data['trend_efficiency_5d'] = (data['close'] - data['close'].shift(5)) / (data['high'].rolling(window=5).max() - data['low'].rolling(window=5).min())
    
    # Efficiency persistence (consecutive days with efficiency > 0.5)
    data['high_efficiency'] = (data['movement_efficiency'] > 0.5).astype(int)
    data['efficiency_persistence'] = data['high_efficiency'] * (data['high_efficiency'].groupby((data['high_efficiency'] != data['high_efficiency'].shift()).cumsum()).cumcount() + 1)
    
    data['price_momentum_5d'] = data['close'] / data['close'].shift(5) - 1
    data['volume_momentum'] = data['volume'] / data['volume_5d_avg']
    data['momentum_efficiency_alignment'] = data['price_momentum_5d'] * data['efficiency_persistence']
    
    data['efficiency_weighted_momentum'] = data['price_momentum_5d'] * data['trend_efficiency_5d']
    data['volume_confirmed_persistence'] = data['efficiency_weighted_momentum'] * data['volume_momentum']
    data['efficiency_momentum_signal'] = data['volume_confirmed_persistence'] * data['range_utilization']
    
    # Volume-Spike Regime Detection
    data['volume_20d_avg'] = data['volume'].rolling(window=20).mean()
    data['volume_spike'] = data['volume'] / data['volume_20d_avg']
    data['atr_20d_avg'] = data['atr_20d'].rolling(window=20).mean()
    data['volatility_spike'] = data['atr_5d'] / data['atr_20d_avg']
    data['spike_intensity'] = data['volume_spike'] * data['volatility_spike']
    
    data['high_20d'] = data['high'].rolling(window=20).max()
    data['low_20d'] = data['low'].rolling(window=20).min()
    data['price_position'] = (data['close'] - data['low_20d']) / (data['high_20d'] - data['low_20d'])
    data['recent_return'] = data['close'] / data['close'].shift(3) - 1
    data['market_state'] = data['price_position'] * data['recent_return']
    
    data['spike_regime_interaction'] = data['spike_intensity'] * data['market_state']
    data['historical_pattern'] = data['spike_regime_interaction'].rolling(window=20).corr(data['returns_3d'].shift(-3))
    data['volume_spike_signal'] = data['spike_regime_interaction'] * data['historical_pattern']
    
    # Liquidity-Enhanced Breakout
    data['price_impact'] = np.abs(data['close'] - data['close'].shift(1)) / data['volume']
    data['turnover_efficiency'] = data['amount'] / data['volume']
    data['liquidity_score'] = (1 / (data['price_impact'] + 1e-8)) * data['turnover_efficiency']
    
    data['breakout_direction'] = np.sign(data['close'] - data['close'].shift(1))
    data['breakout_distance'] = (data['close'] - data['low_10d']) / (data['high_10d'] - data['low_10d'])
    data['raw_breakout'] = data['breakout_distance'] * data['breakout_direction']
    
    data['liquidity_adjusted_breakout'] = data['raw_breakout'] * data['liquidity_score']
    data['liquidity_breakout_signal'] = data['liquidity_adjusted_breakout'] * data['volume_concentration']
    data['liquidity_signal_strength'] = data['liquidity_breakout_signal'].rolling(window=3).mean()
    
    # Momentum-Volatility Alignment
    data['momentum_alignment'] = data['returns_3d'] * data['returns_10d']
    data['momentum_strength'] = np.abs(data['returns_3d']) * np.abs(data['returns_10d'])
    
    data['volatility_regime_mom'] = data['atr_5d'] / data['atr_20d']
    data['volatility_scaled_momentum'] = data['momentum_alignment'] / (data['volatility_regime_mom'] + 1e-8)
    data['strength_confirmation'] = data['volatility_scaled_momentum'] * data['momentum_strength']
    data['momentum_volatility_signal'] = data['strength_confirmation'] * data['volume_concentration']
    
    # Combine all signals with equal weighting
    signals = [
        data['momentum_efficiency_signal'],
        data['breakout_confidence'],
        data['volatility_reversal_signal'],
        data['multi_timeframe_signal'],
        data['efficiency_momentum_signal'],
        data['volume_spike_signal'],
        data['liquidity_signal_strength'],
        data['momentum_volatility_signal']
    ]
    
    # Normalize each signal and combine
    combined_signal = pd.Series(0, index=data.index)
    for signal in signals:
        normalized_signal = (signal - signal.mean()) / (signal.std() + 1e-8)
        combined_signal += normalized_signal
    
    result = combined_signal / len(signals)
    
    return result
