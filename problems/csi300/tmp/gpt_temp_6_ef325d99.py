import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Volume-Price Divergence Detection
    # Directional Volume Analysis
    data['volume_weighted_price_change'] = (data['close'] - data['close'].shift(1)) * data['volume']
    
    # Up-day and down-day volume accumulation (5-day window)
    up_volume = []
    down_volume = []
    for i in range(len(data)):
        if i < 4:
            up_volume.append(np.nan)
            down_volume.append(np.nan)
            continue
            
        window_data = data.iloc[i-4:i+1]
        up_vol = 0
        down_vol = 0
        for j in range(1, len(window_data)):
            if window_data['close'].iloc[j] > window_data['close'].iloc[j-1]:
                up_vol += window_data['volume'].iloc[j]
            elif window_data['close'].iloc[j] < window_data['close'].iloc[j-1]:
                down_vol += window_data['volume'].iloc[j]
        
        up_volume.append(up_vol)
        down_volume.append(down_vol)
    
    data['up_day_volume_accumulation'] = up_volume
    data['down_day_volume_accumulation'] = down_volume
    data['accumulation_divergence_ratio'] = data['up_day_volume_accumulation'] / data['down_day_volume_accumulation']
    
    # Divergence Strength Measurement
    data['price_momentum_5d'] = data['close'] / data['close'].shift(5) - 1
    data['volume_price_momentum_divergence'] = data['volume_weighted_price_change'] - (data['price_momentum_5d'] * data['volume'])
    
    # Divergence persistence (count consecutive days with positive divergence)
    divergence_persistence = []
    count = 0
    for val in data['volume_price_momentum_divergence']:
        if pd.isna(val):
            count = 0
            divergence_persistence.append(np.nan)
        elif val > 0:
            count += 1
            divergence_persistence.append(count)
        else:
            count = 0
            divergence_persistence.append(count)
    data['divergence_persistence'] = divergence_persistence
    
    # Abnormal Volume Patterns
    data['volume_median_21d'] = data['volume'].rolling(window=21, min_periods=1).median()
    data['volume_burst_intensity'] = data['volume'] / data['volume_median_21d']
    
    data['volume_avg_21d'] = data['volume'].rolling(window=21, min_periods=1).mean()
    data['quiet_period_volume'] = np.where(data['volume'] < 0.7 * data['volume_avg_21d'], data['volume'], 0)
    
    data['volume_avg_5d'] = data['volume'].rolling(window=5, min_periods=1).mean()
    data['volume_regime_shift'] = data['volume_avg_5d'] / data['volume_avg_21d']
    
    # Microstructure Momentum Indicators
    # Trade Size Momentum
    data['avg_trade_size'] = data['amount'] / data['volume']
    data['avg_trade_size_5d'] = data['avg_trade_size'].rolling(window=5, min_periods=1).mean()
    data['large_trade_momentum'] = data['avg_trade_size'] / data['avg_trade_size_5d']
    
    data['trade_size_acceleration'] = data['large_trade_momentum'] - data['large_trade_momentum'].shift(1)
    
    # Institutional momentum persistence
    institutional_persistence = []
    for i in range(len(data)):
        if i < 4:
            institutional_persistence.append(np.nan)
            continue
        window_data = data['large_trade_momentum'].iloc[i-4:i+1]
        count = (window_data > 1).sum()
        institutional_persistence.append(count)
    data['institutional_momentum_persistence'] = institutional_persistence
    
    # Price Efficiency Signals
    data['daily_range'] = data['high'] - data['low']
    data['range_utilization_efficiency'] = (data['close'] - data['close'].shift(1)) / data['daily_range']
    data['gap_efficiency'] = (data['open'] - data['close'].shift(1)) / data['daily_range']
    data['intraday_momentum_consistency'] = (data['close'] - data['open']) * (data['open'] - data['close'].shift(1))
    
    # Microstructure Quality Assessment
    data['price_impact_per_volume'] = data['daily_range'] / data['volume']
    data['trade_concentration_quality'] = data['avg_trade_size'] * data['range_utilization_efficiency']
    data['micro_noise_ratio'] = abs(data['close'] - data['open']) / data['daily_range']
    
    # Divergence-Momentum Integration
    # Volume-Driven Momentum Signals
    data['confirmed_volume_momentum'] = data['volume_price_momentum_divergence'] * data['large_trade_momentum']
    data['divergence_breakout_signal'] = data['accumulation_divergence_ratio'] * data['trade_size_acceleration']
    data['volume_quality_momentum'] = data['confirmed_volume_momentum'] * data['trade_concentration_quality']
    
    # Microstructure-Enhanced Divergence
    data['efficient_divergence'] = data['volume_price_momentum_divergence'] * data['range_utilization_efficiency']
    data['quiet_period_signal'] = data['quiet_period_volume'] * data['intraday_momentum_consistency']
    data['burst_confirmation'] = data['volume_burst_intensity'] * data['price_impact_per_volume']
    
    # Persistence-Based Filtering
    data['divergence_momentum_persistence'] = data['divergence_persistence'] * data['institutional_momentum_persistence']
    data['regime_adaptive_weighting'] = data['volume_regime_shift'] * data['micro_noise_ratio']
    data['quality_adjusted_divergence'] = data['efficient_divergence'] * (1 - data['micro_noise_ratio'])
    
    # Multi-Timeframe Confirmation
    # Short-Term vs Medium-Term Alignment
    data['volume_timeframe_alignment'] = data['volume_avg_5d'] / data['volume_avg_21d']
    
    data['price_momentum_10d'] = data['close'] / data['close'].shift(10) - 1
    data['momentum_timeframe_consistency'] = data['price_momentum_5d'] * data['price_momentum_10d']
    
    data['large_trade_momentum_5d_avg'] = data['large_trade_momentum'].rolling(window=5, min_periods=1).mean()
    data['trade_size_timeframe_alignment'] = data['large_trade_momentum'] * data['large_trade_momentum_5d_avg']
    
    # Divergence Confirmation Across Periods
    data['multi_period_volume_confirmation'] = data['volume_burst_intensity'] * data['volume_regime_shift']
    data['price_momentum_timeframe_divergence'] = data['price_momentum_5d'] - data['price_momentum_10d']
    
    data['trade_concentration_quality_5d_avg'] = data['trade_concentration_quality'].rolling(window=5, min_periods=1).mean()
    data['microstructure_consistency'] = data['trade_concentration_quality'] * data['trade_concentration_quality_5d_avg']
    
    # Convergence-Divergence Patterns
    data['volume_weighted_price_change_5d_avg'] = data['volume_weighted_price_change'].rolling(window=5, min_periods=1).mean()
    data['volume_price_convergence'] = abs(data['volume_weighted_price_change'] - data['volume_weighted_price_change_5d_avg'])
    data['trade_size_convergence'] = data['large_trade_momentum'] / data['large_trade_momentum_5d_avg']
    data['multi_timeframe_quality'] = data['microstructure_consistency'] * data['range_utilization_efficiency']
    
    # Advanced Signal Construction
    # Core Divergence Factors
    data['primary_divergence_signal'] = data['volume_price_momentum_divergence'] * data['accumulation_divergence_ratio']
    data['microstructure_enhanced_divergence'] = data['primary_divergence_signal'] * data['trade_concentration_quality']
    data['persistence_weighted_divergence'] = data['microstructure_enhanced_divergence'] * data['divergence_momentum_persistence']
    
    # Momentum Confirmation Layers
    data['volume_confirmed_momentum'] = data['confirmed_volume_momentum'] * data['volume_quality_momentum']
    data['efficiency_adjusted_momentum'] = data['volume_confirmed_momentum'] * (1 - data['micro_noise_ratio'])
    data['timeframe_aligned_momentum'] = data['efficiency_adjusted_momentum'] * data['momentum_timeframe_consistency']
    
    # Composite Factor Generation
    data['divergence_momentum_integration'] = data['persistence_weighted_divergence'] * data['timeframe_aligned_momentum']
    data['quality_filtering'] = data['divergence_momentum_integration'] * data['multi_timeframe_quality']
    data['final_alpha'] = data['quality_filtering'] * data['regime_adaptive_weighting']
    
    # Final output
    return data['final_alpha']
