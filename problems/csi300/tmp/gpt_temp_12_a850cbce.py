import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    data = df.copy()
    
    # Volatility Regime Classification with Range Dynamics
    # Calculate True Range
    data['prev_close'] = data['close'].shift(1)
    data['tr1'] = data['high'] - data['low']
    data['tr2'] = abs(data['high'] - data['prev_close'])
    data['tr3'] = abs(data['low'] - data['prev_close'])
    data['true_range'] = data[['tr1', 'tr2', 'tr3']].max(axis=1)
    
    # Compute volatility regime indicators
    data['avg_tr_20'] = data['true_range'].rolling(window=20, min_periods=20).mean()
    data['median_tr_60'] = data['true_range'].rolling(window=60, min_periods=60).median()
    data['high_vol_regime'] = data['avg_tr_20'] > data['median_tr_60']
    
    # Multi-Timeframe Gap Analysis with Pressure Components
    data['gap'] = (data['open'] / data['prev_close']) - 1
    
    # Gap persistence (count consecutive same-sign gaps over past 5 days)
    gap_sign = np.sign(data['gap'])
    gap_persistence = []
    for i in range(len(data)):
        if i < 5:
            gap_persistence.append(1)
        else:
            current_sign = gap_sign.iloc[i]
            count = 1
            for j in range(1, 5):
                if gap_sign.iloc[i-j] == current_sign:
                    count += 1
                else:
                    break
            gap_persistence.append(count)
    data['gap_persistence'] = gap_persistence
    
    # Gap fill percentage
    data['gap_fill_pct'] = (data['close'] - data['open']) / (data['prev_close'] - data['open'])
    data['gap_fill_pct'] = data['gap_fill_pct'].replace([np.inf, -np.inf], np.nan)
    
    # Intraday pressure components
    data['opening_pressure'] = (data['open'] - data['prev_close']) * data['volume']
    data['closing_pressure'] = (data['close'] - data['open']) * data['volume']
    data['net_pressure'] = data['closing_pressure'] - data['opening_pressure']
    
    # Gap type classification
    data['breakaway_gap'] = (abs(data['gap']) > 0.02) & (data['net_pressure'] > 0)
    data['exhaustion_gap'] = (abs(data['gap']) > 0.02) & (data['net_pressure'] < 0)
    data['common_gap'] = (abs(data['gap']) <= 0.02)
    
    # Regime-Specific Momentum-Turnover Components
    data['turnover'] = data['volume'] * data['close']
    
    # High-volatility component
    data['momentum_3d'] = (data['close'] / data['close'].shift(2)) - 1
    data['turnover_avg_3d'] = data['turnover'].rolling(window=3, min_periods=3).mean()
    data['turnover_avg_9d'] = data['turnover'].rolling(window=9, min_periods=9).mean()
    data['turnover_ratio_3d'] = (data['turnover_avg_3d'] / data['turnover_avg_9d']) - 1
    data['high_vol_component'] = data['momentum_3d'] * data['turnover_ratio_3d']
    
    # Low-volatility component
    data['momentum_5d'] = (data['close'] / data['close'].shift(4)) - 1
    data['turnover_avg_5d'] = data['turnover'].rolling(window=5, min_periods=5).mean()
    data['turnover_avg_15d'] = data['turnover'].rolling(window=15, min_periods=15).mean()
    data['turnover_ratio_5d'] = (data['turnover_avg_5d'] / data['turnover_avg_15d']) - 1
    data['low_vol_component'] = data['momentum_5d'] * data['turnover_ratio_5d']
    
    # Volume-Price Asymmetry Assessment
    # Upside volume confirmation
    data['up_day'] = data['close'] > data['prev_close']
    
    def calc_up_volume_ratio(window_data):
        up_days = window_data[window_data['up_day']]
        if len(up_days) == 0:
            return 0
        up_volume_avg = up_days['volume'].mean()
        total_volume_avg = window_data['volume'].mean()
        return up_volume_avg / total_volume_avg if total_volume_avg > 0 else 0
    
    up_volume_ratios = []
    for i in range(len(data)):
        if i < 10:
            up_volume_ratios.append(0.5)
        else:
            window = data.iloc[i-9:i+1]
            up_volume_ratios.append(calc_up_volume_ratio(window))
    data['up_volume_ratio'] = up_volume_ratios
    
    # Price movement asymmetry
    data['return'] = data['close'] / data['prev_close'] - 1
    
    def calc_price_asymmetry(window_data):
        pos_returns = window_data[window_data['return'] > 0]['return'].sum()
        neg_returns = abs(window_data[window_data['return'] < 0]['return'].sum())
        return np.log(1 + pos_returns) - np.log(1 + neg_returns)
    
    price_asymmetries = []
    for i in range(len(data)):
        if i < 10:
            price_asymmetries.append(0)
        else:
            window = data.iloc[i-9:i+1]
            price_asymmetries.append(calc_price_asymmetry(window))
    data['price_asymmetry'] = price_asymmetries
    
    # Combine asymmetry components
    data['volume_price_asymmetry'] = np.sqrt(data['up_volume_ratio'] * data['price_asymmetry'])
    
    # Momentum Efficiency with Volume Asymmetry
    # Short-term efficiency (3-day)
    data['abs_return_1d'] = abs(data['close'] - data['prev_close'])
    data['price_change_3d'] = abs(data['close'] - data['close'].shift(3))
    data['cum_abs_3d'] = data['abs_return_1d'].rolling(window=3, min_periods=3).sum()
    data['efficiency_3d'] = data['price_change_3d'] / data['cum_abs_3d']
    data['efficiency_3d'] = data['efficiency_3d'].replace([np.inf, -np.inf], 0)
    
    # Medium-term efficiency (5-day)
    data['price_change_5d'] = abs(data['close'] - data['close'].shift(5))
    data['cum_abs_5d'] = data['abs_return_1d'].rolling(window=5, min_periods=5).sum()
    data['efficiency_5d'] = data['price_change_5d'] / data['cum_abs_5d']
    data['efficiency_5d'] = data['efficiency_5d'].replace([np.inf, -np.inf], 0)
    
    data['efficiency_gradient'] = data['efficiency_5d'] - data['efficiency_3d']
    
    # Momentum convexity
    mom_3d = (data['close'] / data['close'].shift(3)) - 1
    mom_8d = (data['close'] / data['close'].shift(8)) - 1
    data['momentum_convexity'] = mom_3d / mom_8d
    data['momentum_convexity'] = data['momentum_convexity'].replace([np.inf, -np.inf], 0)
    
    # Volume acceleration
    data['volume_acceleration'] = data['volume'] / data['volume'].shift(5)
    data['volume_acceleration'] = data['volume_acceleration'].replace([np.inf, -np.inf], 1)
    
    # Intraday Strength with Range Expansion
    data['intraday_efficiency'] = (data['close'] - data['low']) / (data['high'] - data['low'])
    data['intraday_efficiency'] = data['intraday_efficiency'].replace([np.inf, -np.inf], 0.5)
    
    data['daily_range'] = data['high'] - data['low']
    data['avg_range_10d'] = data['daily_range'].rolling(window=10, min_periods=10).mean()
    data['range_expansion'] = data['daily_range'] / data['avg_range_10d']
    data['range_expansion'] = data['range_expansion'].replace([np.inf, -np.inf], 1)
    
    data['opening_strength'] = data['gap'] * data['intraday_efficiency']
    data['pressure_efficiency'] = data['net_pressure'] * np.sqrt(data['intraday_efficiency'] * data['range_expansion'])
    
    # Momentum Quality Filter
    data['momentum_10d'] = (data['close'] / data['close'].shift(9)) - 1
    
    def calc_consistency_score(row):
        signs = [np.sign(row['momentum_3d']), np.sign(row['momentum_5d']), np.sign(row['momentum_10d'])]
        target_sign = np.sign(row['momentum_5d'])
        return sum(1 for sign in signs if sign == target_sign)
    
    consistency_scores = []
    for i in range(len(data)):
        if i < 10:
            consistency_scores.append(3)
        else:
            row = data.iloc[i]
            consistency_scores.append(calc_consistency_score(row))
    data['momentum_consistency'] = consistency_scores
    data['momentum_quality_filter'] = data['momentum_consistency'] >= 2
    
    # Volatility Scaling Adjustment
    data['high_20d'] = data['high'].rolling(window=20, min_periods=20).max()
    data['low_20d'] = data['low'].rolling(window=20, min_periods=20).min()
    data['range_ratio'] = (data['high_20d'] / data['low_20d']) - 1
    
    # Adaptive Signal Synthesis
    factor_values = []
    
    for i in range(len(data)):
        if i < 20:  # Ensure sufficient data for calculations
            factor_values.append(0)
            continue
            
        row = data.iloc[i]
        
        # Select regime-specific component
        if row['high_vol_regime']:
            regime_component = row['high_vol_component']
            gap_component = row['gap'] * row['momentum_3d'] * row['net_pressure']
        else:
            regime_component = row['low_vol_component']
            gap_component = row['gap'] * row['gap_persistence'] * row['volume_price_asymmetry']
        
        # Combine components
        base_signal = regime_component * gap_component * row['volume_price_asymmetry']
        
        # Apply momentum quality filter
        if not row['momentum_quality_filter']:
            base_signal = 0
        
        # Apply gap type adjustments
        if row['breakaway_gap']:
            base_signal *= 1.5
        elif row['exhaustion_gap']:
            base_signal *= -1
        
        # Multiply by intraday strength components
        base_signal *= row['opening_strength'] * row['pressure_efficiency']
        
        # Apply efficiency enhancement
        base_signal *= row['efficiency_gradient'] * row['momentum_convexity']
        
        # Volatility scaling
        base_signal *= row['range_ratio']
        
        # Final normalization
        final_factor = np.cbrt(base_signal) if base_signal != 0 else 0
        factor_values.append(final_factor)
    
    # Create output series
    factor_series = pd.Series(factor_values, index=data.index)
    return factor_series
