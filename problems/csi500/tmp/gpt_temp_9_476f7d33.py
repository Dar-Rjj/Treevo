import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

def heuristics_v2(df):
    """
    Regime-Adaptive Momentum Convergence Alpha Factor
    """
    # Copy data to avoid modifying original
    data = df.copy()
    
    # Multi-Timeframe Feature Extraction
    # Price-Based Features
    data['mom_5d'] = data['close'] / data['close'].shift(5) - 1
    data['mom_10d'] = data['close'] / data['close'].shift(10) - 1
    data['mom_20d'] = data['close'] / data['close'].shift(20) - 1
    data['price_range_eff'] = (data['close'] - data['low']) / (data['high'] - data['low'] + 1e-8)
    
    # Volume-Based Features
    data['vol_mom_5d'] = data['volume'] / data['volume'].shift(5) - 1
    data['vol_mom_10d'] = data['volume'] / data['volume'].shift(10) - 1
    data['vol_mom_20d'] = data['volume'] / data['volume'].shift(20) - 1
    data['vol_concentration'] = data['volume'] / data['volume'].rolling(5).mean()
    
    # Volatility Features
    data['daily_range'] = (data['high'] - data['low']) / data['close']
    data['close_vol'] = abs(data['close'] / data['close'].shift(1) - 1)
    data['gap_vol'] = abs(data['open'] / data['close'].shift(1) - 1)
    
    # Regime Detection & Classification
    # Volatility Regime
    range_median = data['daily_range'].rolling(20).median()
    data['vol_regime'] = 'normal'
    data.loc[data['daily_range'] > 1.5 * range_median, 'vol_regime'] = 'high'
    data.loc[data['daily_range'] < 0.7 * range_median, 'vol_regime'] = 'low'
    
    # Trend Regime
    def calculate_trend_regime(series, window=20):
        results = []
        for i in range(len(series)):
            if i < window:
                results.append('range_bound')
                continue
            y = series.iloc[i-window:i].values
            x = np.arange(window).reshape(-1, 1)
            model = LinearRegression()
            model.fit(x, y)
            slope = model.coef_[0]
            y_pred = model.predict(x)
            r_squared = 1 - np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2)
            
            if slope > 0.001 and r_squared > 0.3:
                results.append('strong_uptrend')
            elif slope < -0.001 and r_squared > 0.3:
                results.append('strong_downtrend')
            else:
                results.append('range_bound')
        return results
    
    data['trend_regime'] = calculate_trend_regime(data['close'])
    
    # Volume Regime
    vol_concentration_median = data['vol_concentration'].rolling(20).median()
    data['volume_regime'] = 'normal'
    data.loc[data['vol_concentration'] > 1.3 * vol_concentration_median, 'volume_regime'] = 'high'
    data.loc[data['vol_concentration'] < 0.8 * vol_concentration_median, 'volume_regime'] = 'low'
    
    # Adaptive Feature Weighting
    def get_timeframe_weights(vol_regime):
        if vol_regime == 'high':
            return {'short': 0.6, 'medium': 0.3, 'long': 0.1}
        elif vol_regime == 'low':
            return {'short': 0.2, 'medium': 0.5, 'long': 0.3}
        else:  # normal
            return {'short': 0.4, 'medium': 0.4, 'long': 0.2}
    
    def get_trend_multiplier(trend_regime):
        if trend_regime == 'strong_uptrend':
            return 1.2
        elif trend_regime == 'strong_downtrend':
            return 1.1
        else:  # range_bound
            return 1.3
    
    def get_volume_adjustment(volume_regime):
        if volume_regime == 'high':
            return 1.5
        elif volume_regime == 'low':
            return 1.2
        else:  # normal
            return 1.0
    
    # Momentum Convergence Analysis
    def calculate_convergence(row):
        mom_signs = [np.sign(row['mom_5d']), np.sign(row['mom_10d']), np.sign(row['mom_20d'])]
        unique_signs = len(set(mom_signs))
        
        if unique_signs == 1 and mom_signs[0] != 0:
            strength = abs(row['mom_5d'] + row['mom_10d'] + row['mom_20d'])
            if strength > 0.1:
                return 'strong_positive', 1.5
            else:
                return 'weak_positive', 1.2
        elif unique_signs == 3:
            return 'strong_negative', 0.5
        else:
            return 'mixed', 0.8
    
    def calculate_divergence(row):
        price_sign = np.sign(row['mom_5d'])
        volume_sign = np.sign(row['vol_mom_5d'])
        
        if price_sign > 0 and volume_sign < 0:
            return 'bearish', -0.3
        elif price_sign < 0 and volume_sign > 0:
            return 'bullish', 0.3
        else:
            return 'none', 0.0
    
    # Calculate convergence and divergence
    convergence_data = data.apply(calculate_convergence, axis=1, result_type='expand')
    data[['convergence_type', 'convergence_mult']] = convergence_data
    data[['convergence_type', 'convergence_mult']] = data[['convergence_type', 'convergence_mult']].fillna({'convergence_type': 'mixed', 'convergence_mult': 0.8})
    
    divergence_data = data.apply(calculate_divergence, axis=1, result_type='expand')
    data[['divergence_type', 'divergence_adj']] = divergence_data
    data[['divergence_type', 'divergence_adj']] = data[['divergence_type', 'divergence_adj']].fillna({'divergence_type': 'none', 'divergence_adj': 0.0})
    
    # Final Factor Construction
    base_signals = []
    
    for idx, row in data.iterrows():
        if pd.isna(row['mom_5d']) or pd.isna(row['mom_10d']) or pd.isna(row['mom_20d']):
            base_signals.append(np.nan)
            continue
            
        # Get regime-based weights and adjustments
        weights = get_timeframe_weights(row['vol_regime'])
        trend_mult = get_trend_multiplier(row['trend_regime'])
        vol_adj = get_volume_adjustment(row['volume_regime'])
        
        # Calculate weighted momentum features
        price_momentum = (weights['short'] * row['mom_5d'] + 
                         weights['medium'] * row['mom_10d'] + 
                         weights['long'] * row['mom_20d'])
        
        volume_momentum = (weights['short'] * row['vol_mom_5d'] + 
                          weights['medium'] * row['vol_mom_10d'] + 
                          weights['long'] * row['vol_mom_20d'])
        
        # Apply regime adjustments
        if row['trend_regime'] in ['strong_uptrend', 'strong_downtrend']:
            price_momentum *= trend_mult
            volume_momentum *= trend_mult
        else:  # range_bound
            vol_features = (row['daily_range'] + row['close_vol'] + row['gap_vol']) / 3
            vol_features *= trend_mult
        
        # Apply volume regime adjustments
        if row['volume_regime'] == 'high':
            volume_momentum *= vol_adj
        elif row['volume_regime'] == 'low':
            price_momentum *= vol_adj
        
        # Combine features
        base_signal = (price_momentum + volume_momentum + row['price_range_eff'] + 
                      row['vol_concentration']) / 4
        
        # Apply convergence multiplier and divergence adjustment
        final_signal = base_signal * row['convergence_mult'] + row['divergence_adj']
        
        base_signals.append(final_signal)
    
    data['base_signal'] = base_signals
    
    # Volatility Scaling
    returns = data['close'].pct_change()
    vol_20d = returns.rolling(20).std()
    data['scaled_signal'] = data['base_signal'] / (vol_20d + 0.001)
    
    # Stationarity Enhancement
    data['diff_signal'] = data['scaled_signal'].diff()
    data['final_signal'] = data['diff_signal'] - data['diff_signal'].rolling(5).mean()
    
    # Signal Quality Check
    signal_mean = data['final_signal'].mean()
    signal_std = data['final_signal'].std()
    upper_bound = signal_mean + 3 * signal_std
    lower_bound = signal_mean - 3 * signal_std
    
    data['final_signal'] = np.clip(data['final_signal'], lower_bound, upper_bound)
    
    return data['final_signal']
