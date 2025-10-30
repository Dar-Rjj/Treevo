import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    data = df.copy()
    
    # Multi-Scale Efficiency Synchronization
    # Short-term directional efficiency (3-day)
    data['efficiency'] = (data['close'] - data['open']) / (data['high'] - data['low']).replace(0, np.nan)
    data['eff_momentum'] = data['efficiency'] / data['efficiency'].rolling(window=3, min_periods=1).mean()
    data['eff_consistency'] = 1 - (data['efficiency'].rolling(window=3, min_periods=1).std() / 
                                 data['efficiency'].rolling(window=3, min_periods=1).mean().abs()).replace([np.inf, -np.inf], np.nan)
    
    # Medium-term movement efficiency (8-day)
    data['total_movement'] = (data['high'] - data['low']).rolling(window=8, min_periods=1).sum()
    data['net_change'] = data['close'] - data['close'].shift(8)
    data['medium_efficiency'] = data['net_change'] / data['total_movement'].replace(0, np.nan)
    
    # Efficiency synchronization score
    data['direction_alignment'] = (np.sign(data['efficiency']) == np.sign(data['medium_efficiency'])).astype(int)
    data['magnitude_convergence'] = 1 - (data['efficiency'] - data['medium_efficiency']).abs()
    data['sync_factor'] = data['direction_alignment'] * data['magnitude_convergence']
    
    # Volume-Efficiency Alignment Matrix
    # Volume regime assessment
    data['volume_expansion'] = data['volume'] / data['volume'].shift(1).replace(0, np.nan)
    vol_mean_5d = data['volume'].rolling(window=5, min_periods=1).mean()
    data['volume_consistency'] = ((data['volume'] > vol_mean_5d).rolling(window=5, min_periods=1).sum() / 5)
    data['volume_regime'] = data['volume_expansion'] * data['volume_consistency']
    
    # Efficiency-volume directional alignment
    data['price_volume_dir'] = np.sign(data['close'] - data['open']) * np.sign(data['volume'] - data['volume'].shift(1))
    data['alignment_strength'] = (data['close'] - data['open']).abs() * data['volume']
    data['confirmation_score'] = data['price_volume_dir'] * data['alignment_strength']
    
    # Volume-weighted efficiency enhancement
    data['volume_validated_eff'] = data['sync_factor'] * data['confirmation_score'] * data['volume_regime']
    
    # Multi-Timeframe Convergence Divergence
    data['short_term_accel'] = data['efficiency'] - data['medium_efficiency']
    long_efficiency = (data['close'] - data['close'].shift(15)) / (data['high'] - data['low']).rolling(window=15, min_periods=1).sum().replace(0, np.nan)
    data['medium_term_momentum'] = data['medium_efficiency'] - long_efficiency
    data['convergence_signal'] = data['short_term_accel'] * data['medium_term_momentum']
    
    # Regime-Adaptive Volatility Synchronization
    # Volatility regime classification
    data['daily_range'] = data['high'] - data['low']
    data['gap_high'] = (data['high'] - data['close'].shift(1)).abs()
    data['gap_low'] = (data['low'] - data['close'].shift(1)).abs()
    data['true_range'] = data[['daily_range', 'gap_high', 'gap_low']].max(axis=1)
    
    returns = data['close'] / data['close'].shift(1) - 1
    data['current_volatility'] = returns.rolling(window=10, min_periods=1).std()
    vol_mean_5d = data['current_volatility'].rolling(window=5, min_periods=1).mean()
    data['volatility_momentum'] = data['current_volatility'] / vol_mean_5d.replace(0, np.nan)
    data['high_vol_regime'] = (data['volatility_momentum'] > 1.2).astype(int)
    
    # Volatility-Efficiency Synchronization
    range_mean_5d = data['daily_range'].rolling(window=5, min_periods=1).mean()
    data['range_compression'] = data['daily_range'] / range_mean_5d.replace(0, np.nan)
    data['compression_persistence'] = 1 - (data['daily_range'].rolling(window=5, min_periods=1).std() / 
                                          data['daily_range'].rolling(window=5, min_periods=1).mean()).replace([np.inf, -np.inf], np.nan)
    data['vol_momentum_signal'] = data['range_compression'] * data['compression_persistence']
    
    # Efficiency-volatility alignment
    data['vol_dir_alignment'] = np.sign(data['sync_factor']) * np.sign(data['vol_momentum_signal'])
    
    # Calculate rolling correlation between efficiency and daily range
    eff_vol_corr = []
    for i in range(len(data)):
        if i >= 4:
            window_eff = data['efficiency'].iloc[i-4:i+1]
            window_range = data['daily_range'].iloc[i-4:i+1]
            corr = window_eff.corr(window_range)
            eff_vol_corr.append(corr if not np.isnan(corr) else 0)
        else:
            eff_vol_corr.append(0)
    data['magnitude_correlation'] = eff_vol_corr
    
    data['vol_sync_quality'] = data['vol_dir_alignment'] * data['magnitude_correlation']
    
    # Volatility-adaptive signal scaling
    data['vol_adaptive_momentum'] = np.where(
        data['high_vol_regime'] == 1,
        data['volume_validated_eff'] * (1 + data['vol_sync_quality']),
        data['volume_validated_eff'] * data['vol_sync_quality']
    )
    
    # Intraday Momentum Integration Framework
    data['opening_momentum'] = (data['open'] - data['close'].shift(1)) / data['close'].shift(1).replace(0, np.nan)
    data['intraday_efficiency'] = (data['close'] - data['open']) / (data['high'] - data['low']).replace(0, np.nan)
    data['closing_momentum'] = (data['close'] - data['open']) / data['open'].replace(0, np.nan)
    data['momentum_integration'] = data['opening_momentum'] * data['intraday_efficiency'] * data['closing_momentum']
    
    # Efficiency-weighted momentum enhancement
    data['weighted_momentum'] = data['momentum_integration'] * data['efficiency'] * data['confirmation_score']
    
    # Trend Quality and Persistence Assessment
    data['short_term_trend'] = data['close'] / data['close'].shift(3) - 1
    data['medium_term_trend'] = data['close'] / data['close'].shift(8) - 1
    data['long_term_trend'] = data['close'] / data['close'].shift(15) - 1
    
    trend_alignment = ((np.sign(data['short_term_trend']) == np.sign(data['medium_term_trend'])) & 
                      (np.sign(data['medium_term_trend']) == np.sign(data['long_term_trend']))).astype(int)
    
    data['trend_efficiency'] = data['efficiency'] * np.sign(data['close'] - data['open'])
    data['trend_persistence'] = 1 - ((data['short_term_trend'] / data['medium_term_trend'].replace(0, np.nan)) - 1).abs()
    data['trend_quality'] = data['trend_efficiency'] * data['trend_persistence']
    
    # Volume-trend synchronization
    vol_mean_5d = data['volume'].rolling(window=5, min_periods=1).mean()
    data['volume_trend_alignment'] = np.sign(data['volume'] - vol_mean_5d) * np.sign(data['trend_efficiency'])
    data['trend_confirmation'] = data['trend_efficiency'].abs() * data['volume']
    data['volume_validated_trend'] = data['volume_trend_alignment'] * data['trend_confirmation']
    
    # Adaptive Composite Alpha Synthesis
    # Multi-dimensional signal integration with regime-adaptive weighting
    efficiency_volume_factor = data['volume_validated_eff'] * data['convergence_signal']
    volatility_momentum_factor = data['vol_adaptive_momentum']
    trend_quality_factor = data['trend_quality'] * data['volume_validated_trend']
    intraday_momentum_factor = data['weighted_momentum']
    
    # Regime-adaptive weighting
    high_vol_weight = np.where(data['high_vol_regime'] == 1, 0.5, 0.2)
    low_vol_weight = np.where(data['high_vol_regime'] == 0, 0.5, 0.2)
    
    composite_alpha = (
        high_vol_weight * efficiency_volume_factor +
        high_vol_weight * volatility_momentum_factor +
        low_vol_weight * trend_quality_factor +
        low_vol_weight * intraday_momentum_factor
    )
    
    # Signal quality validation - filter weak signals
    signal_strength = (
        data['eff_consistency'].abs() + 
        data['volume_regime'].abs() + 
        data['vol_sync_quality'].abs() + 
        trend_alignment
    ) / 4
    
    final_factor = composite_alpha * signal_strength
    
    return final_factor
