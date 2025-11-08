import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Volatility Regime Classification
    # Calculate True Range
    data['prev_close'] = data['close'].shift(1)
    data['tr1'] = data['high'] - data['low']
    data['tr2'] = abs(data['high'] - data['prev_close'])
    data['tr3'] = abs(data['low'] - data['prev_close'])
    data['true_range'] = data[['tr1', 'tr2', 'tr3']].max(axis=1)
    
    # 5-day True Range Volatility
    data['vol_5d'] = data['true_range'].rolling(window=5).mean()
    data['vol_20d_avg'] = data['true_range'].rolling(window=20).mean()
    
    # Regime Assignment
    data['high_vol_regime'] = data['vol_5d'] > data['vol_20d_avg']
    
    # Multi-Timeframe Momentum Analysis
    # Price Momentum
    data['mom_5d'] = data['close'] / data['close'].shift(5) - 1
    data['mom_20d'] = data['close'] / data['close'].shift(20) - 1
    
    # Intraday strength
    data['intraday_strength'] = (data['close'] - data['open']) / (data['high'] - data['low']).replace(0, np.nan)
    data['intraday_strength'] = data['intraday_strength'].fillna(0)
    
    # Volume Momentum Analysis
    data['volume_5d_change'] = data['volume'] / data['volume'].shift(5) - 1
    data['volume_ma_5d'] = data['volume'].rolling(window=5).mean()
    data['volume_deviation'] = data['volume'] / data['volume_ma_5d']
    
    # Volume direction comparison
    data['volume_ma_10d'] = data['volume'].rolling(window=10).mean()
    data['volume_direction'] = (data['volume_ma_5d'] > data['volume_ma_5d'].shift(1)) & (data['volume_ma_10d'] > data['volume_ma_10d'].shift(1))
    
    # Convergence-Divergence Engine
    # Price Momentum Alignment
    data['momentum_alignment'] = ((data['mom_5d'] > 0) == (data['mom_20d'] > 0)).astype(int)
    data['intraday_trend_alignment'] = ((data['intraday_strength'] > 0) == (data['mom_5d'] > 0)).astype(int)
    
    # Price-Volume Correlation
    data['price_change_1d'] = data['close'].pct_change()
    data['volume_change_1d'] = data['volume'].pct_change()
    
    # 5-day correlation
    data['price_volume_corr_5d'] = data['price_change_1d'].rolling(window=5).corr(data['volume_change_1d'])
    
    # 10-day correlation
    data['price_volume_corr_10d'] = data['price_change_1d'].rolling(window=10).corr(data['volume_change_1d'])
    
    # Multi-Signal Consistency
    data['signal_consistency'] = (
        data['momentum_alignment'] + 
        data['intraday_trend_alignment'] + 
        (data['price_volume_corr_5d'] > 0).astype(int) + 
        (data['price_volume_corr_10d'] > 0).astype(int) +
        data['volume_direction'].astype(int)
    )
    
    # Regime-Adaptive Signal Generation
    factor_values = []
    
    for i in range(len(data)):
        if i < 20:  # Need enough data for calculations
            factor_values.append(0)
            continue
            
        row = data.iloc[i]
        
        if row['high_vol_regime']:
            # High Volatility: Mean reversion focus
            volume_weight = 0.6
            convergence_threshold = 4  # Stronger threshold
            momentum_breakdown = (row['mom_5d'] * row['mom_20d'] < 0)  # Momentum breakdown
            
            if row['signal_consistency'] >= convergence_threshold:
                if momentum_breakdown:
                    # Mean reversion signal
                    signal = -np.sign(row['mom_5d']) * (abs(row['mom_5d']) + volume_weight * abs(row['volume_5d_change']))
                else:
                    # Strong convergence with volume confirmation
                    signal = np.sign(row['mom_5d']) * (abs(row['mom_5d']) + volume_weight * abs(row['volume_5d_change']))
            else:
                # Weak or divergent signals
                signal = -np.sign(row['mom_5d']) * abs(row['mom_5d']) * 0.5
                
        else:
            # Low Volatility: Trend continuation focus
            volume_weight = 0.3  # Reduced volume sensitivity
            convergence_threshold = 3  # Moderate threshold
            momentum_acceleration = (abs(row['mom_5d']) > abs(row['mom_20d']))  # Momentum acceleration
            
            if row['signal_consistency'] >= convergence_threshold:
                if momentum_acceleration:
                    # Strong trend continuation
                    signal = np.sign(row['mom_5d']) * (abs(row['mom_5d']) * 1.2 + volume_weight * row['volume_deviation'])
                else:
                    # Stable trend
                    signal = np.sign(row['mom_5d']) * (abs(row['mom_5d']) + volume_weight * row['volume_deviation'])
            else:
                # Weak signals
                signal = np.sign(row['mom_5d']) * abs(row['mom_5d']) * 0.3
        
        factor_values.append(signal)
    
    # Create output series
    factor_series = pd.Series(factor_values, index=data.index, name='regime_adaptive_convergence_factor')
    
    # Handle any remaining NaN values
    factor_series = factor_series.fillna(0)
    
    return factor_series
