import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import linregress

def heuristics_v2(df):
    """
    Cross-Association Fractal Pattern Recognition with Dynamic Regime Switching
    """
    data = df.copy()
    
    # Helper function for Hurst exponent calculation
    def hurst_exponent(ts, window):
        """Calculate Hurst exponent using R/S analysis"""
        if len(ts) < window:
            return np.nan
        
        lags = range(2, min(window, len(ts)//2))
        tau = []
        for lag in lags:
            # Calculate R/S for each lag
            rs_values = []
            for i in range(0, len(ts) - lag, lag):
                segment = ts[i:i+lag]
                if len(segment) < 2:
                    continue
                mean_segment = np.mean(segment)
                cum_dev = segment - mean_segment
                r = np.max(cum_dev) - np.min(cum_dev)
                s = np.std(segment)
                if s > 0:
                    rs_values.append(r / s)
            
            if rs_values:
                tau.append(np.log(np.mean(rs_values)))
            else:
                tau.append(np.nan)
        
        # Remove NaN values
        valid_lags = [l for l, t in zip(lags, tau) if not np.isnan(t)]
        valid_tau = [t for t in tau if not np.isnan(t)]
        
        if len(valid_tau) < 2:
            return 0.5
        
        # Linear regression to get Hurst exponent
        try:
            slope, _, _, _, _ = linregress(np.log(valid_lags), valid_tau)
            return slope
        except:
            return 0.5
    
    # Calculate returns and volume changes
    data['returns'] = data['close'].pct_change()
    data['volume_change'] = data['volume'].pct_change()
    
    # Multi-Scale Volatility Fractal Analysis
    hurst_5 = []
    hurst_20 = []
    
    for i in range(len(data)):
        if i >= 20:
            # 5-day Hurst
            price_5 = data['close'].iloc[i-4:i+1].values
            h5 = hurst_exponent(price_5, 5)
            hurst_5.append(h5)
            
            # 20-day Hurst
            price_20 = data['close'].iloc[i-19:i+1].values
            h20 = hurst_exponent(price_20, 20)
            hurst_20.append(h20)
        else:
            hurst_5.append(np.nan)
            hurst_20.append(np.nan)
    
    data['hurst_5'] = hurst_5
    data['hurst_20'] = hurst_20
    data['fractal_dim_5'] = 2 - data['hurst_5']
    data['fractal_dim_20'] = 2 - data['hurst_20']
    
    # Price-Volume Correlation Fractal
    data['corr_5'] = data['returns'].rolling(window=5, min_periods=3).corr(data['volume_change'])
    data['corr_10'] = data['returns'].rolling(window=10, min_periods=5).corr(data['volume_change'])
    data['corr_20'] = data['returns'].rolling(window=20, min_periods=10).corr(data['volume_change'])
    
    # Correlation persistence using autocorrelation
    data['corr_persistence'] = data['corr_5'].rolling(window=10, min_periods=5).apply(
        lambda x: x.autocorr(lag=1) if len(x.dropna()) >= 5 else np.nan
    )
    
    # Market Regime Classification
    def classify_regime(row):
        if pd.isna(row['hurst_20']) or pd.isna(row['corr_20']):
            return 0
        
        # Trending regime: High Hurst + Strong correlation
        if row['hurst_20'] > 0.6 and abs(row['corr_20']) > 0.3:
            return 1
        # Mean-reverting regime: Low Hurst + Weak correlation
        elif row['hurst_20'] < 0.4 and abs(row['corr_20']) < 0.2:
            return -1
        # Random walk regime
        else:
            return 0
    
    data['regime'] = data.apply(classify_regime, axis=1)
    
    # Intraday Pattern Association
    # Opening Gap Pattern Strength
    data['gap_magnitude'] = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
    data['gap_filling_ratio'] = (data['close'] - data['open']) / (data['gap_magnitude'].replace(0, np.nan))
    data['gap_persistence'] = data['gap_magnitude'].rolling(window=3, min_periods=2).apply(
        lambda x: np.sum(np.sign(x.dropna()) == np.sign(x.dropna().iloc[-1])) if len(x.dropna()) >= 2 else 0
    )
    
    # High-Low Reversal Patterns
    data['daily_range_efficiency'] = (data['close'] - data['open']) / (data['high'] - data['low']).replace(0, np.nan)
    data['extreme_rejection'] = (data['high'] - data['high'].shift(1)) * (data['low'] - data['low'].shift(1))
    data['midpoint'] = (data['high'] + data['low']) / 2
    data['intraday_momentum_shift'] = (data['close'] - data['midpoint']) / (data['high'] - data['low']).replace(0, np.nan)
    
    # Volume-Price Association Patterns
    data['volume_zscore'] = (data['volume'] - data['volume'].rolling(window=20, min_periods=10).mean()) / data['volume'].rolling(window=20, min_periods=10).std()
    data['volume_spike_persistence'] = data['volume_zscore'].rolling(window=5, min_periods=3).apply(
        lambda x: np.sum(x.dropna() > 1) if len(x.dropna()) >= 3 else 0
    )
    data['price_volume_efficiency'] = (data['close'] - data['open']) / data['volume'].replace(0, np.nan)
    
    # Multi-Fractal Momentum Calculation
    data['momentum_5'] = data['close'].pct_change(5)
    data['momentum_20'] = data['close'].pct_change(20)
    data['momentum_60'] = data['close'].pct_change(60)
    
    # Fractal momentum based on price path complexity
    def fractal_momentum(price_series, window):
        if len(price_series) < window:
            return np.nan
        
        # Calculate path efficiency (straight-line distance vs actual path)
        start_price = price_series.iloc[0]
        end_price = price_series.iloc[-1]
        straight_distance = abs(end_price - start_price)
        
        actual_path = 0
        for i in range(1, len(price_series)):
            actual_path += abs(price_series.iloc[i] - price_series.iloc[i-1])
        
        if actual_path > 0:
            return straight_distance / actual_path
        else:
            return 0
    
    data['fractal_momentum_5'] = data['close'].rolling(window=5, min_periods=3).apply(
        lambda x: fractal_momentum(x, 5)
    )
    data['fractal_momentum_20'] = data['close'].rolling(window=20, min_periods=10).apply(
        lambda x: fractal_momentum(x, 20)
    )
    
    # Regime-Specific Pattern Scoring
    def regime_pattern_score(row):
        if pd.isna(row['regime']):
            return 0
        
        score = 0
        
        # Trending regime patterns
        if row['regime'] == 1:
            # Strong gap persistence with momentum
            if abs(row['gap_magnitude']) > 0.01 and row['gap_persistence'] >= 2:
                score += row['gap_magnitude'] * 2
            # High range efficiency with trend
            if abs(row['daily_range_efficiency']) > 0.7:
                score += row['daily_range_efficiency'] * 1.5
            # Volume confirmation
            if row['volume_zscore'] > 1 and row['momentum_5'] * row['gap_magnitude'] > 0:
                score += row['volume_zscore'] * 0.5
        
        # Mean-reverting regime patterns
        elif row['regime'] == -1:
            # Extreme rejection signals
            if row['extreme_rejection'] < 0:  # Opposite signs indicate rejection
                score += -np.sign(row['extreme_rejection']) * 1.2
            # Intraday momentum reversal
            if abs(row['intraday_momentum_shift']) > 0.8:
                score += -row['intraday_momentum_shift'] * 1.0
            # Low fractal momentum (choppy markets)
            if row['fractal_momentum_5'] < 0.3:
                score += (0.3 - row['fractal_momentum_5']) * 2
        
        # Random walk regime patterns
        else:
            # Breakout patterns with volume
            if abs(row['gap_magnitude']) > 0.02 and row['volume_zscore'] > 1.5:
                score += row['gap_magnitude'] * row['volume_zscore'] * 0.8
            # High fractal dimension (complex patterns)
            if row['fractal_dim_5'] > 1.6:
                score += (row['fractal_dim_5'] - 1.5) * 1.5
        
        return score
    
    data['regime_pattern_score'] = data.apply(regime_pattern_score, axis=1)
    
    # Cross-Association Signal Integration
    def cross_association_signal(row):
        if pd.isna(row['regime']) or pd.isna(row['corr_20']):
            return 0
        
        signal = 0
        
        # Intraday pattern alignment
        gap_signal = row['gap_magnitude'] * (1 + 0.2 * row['gap_persistence'])
        reversal_signal = -row['intraday_momentum_shift'] * (1 + abs(row['daily_range_efficiency']))
        volume_signal = row['volume_zscore'] * np.sign(row['price_volume_efficiency'])
        
        # Multi-timeframe fractal alignment
        fractal_alignment = 0
        if not pd.isna(row['fractal_momentum_5']) and not pd.isna(row['fractal_momentum_20']):
            if np.sign(row['fractal_momentum_5']) == np.sign(row['fractal_momentum_20']):
                fractal_alignment = (row['fractal_momentum_5'] + row['fractal_momentum_20']) / 2
        
        # Regime-weighted combination
        if row['regime'] == 1:  # Trending
            signal = (gap_signal * 0.4 + volume_signal * 0.3 + fractal_alignment * 0.3)
        elif row['regime'] == -1:  # Mean-reverting
            signal = (reversal_signal * 0.5 + gap_signal * 0.2 + volume_signal * 0.3)
        else:  # Random walk
            signal = (gap_signal * 0.3 + volume_signal * 0.4 + fractal_alignment * 0.3)
        
        return signal
    
    data['cross_association_signal'] = data.apply(cross_association_signal, axis=1)
    
    # Final Adaptive Alpha Generation
    def adaptive_alpha(row):
        if pd.isna(row['regime']) or pd.isna(row['cross_association_signal']):
            return 0
        
        base_signal = row['cross_association_signal']
        regime_strength = abs(row['regime_pattern_score'])
        
        # Enhance signal based on regime strength and pattern reliability
        enhanced_signal = base_signal * (1 + 0.5 * regime_strength)
        
        # Apply correlation persistence filter
        if not pd.isna(row['corr_persistence']):
            if row['corr_persistence'] > 0.3:  # High persistence
                enhanced_signal *= 1.2
            elif row['corr_persistence'] < -0.3:  # Low/negative persistence
                enhanced_signal *= 0.8
        
        # Apply fractal dimension filter for signal stability
        if not pd.isna(row['fractal_dim_20']):
            if row['fractal_dim_20'] > 1.7:  # Very complex patterns
                enhanced_signal *= 0.7
            elif row['fractal_dim_20'] < 1.3:  # Very smooth patterns
                enhanced_signal *= 1.3
        
        return enhanced_signal
    
    data['alpha'] = data.apply(adaptive_alpha, axis=1)
    
    # Normalize the final alpha
    alpha_series = data['alpha'].copy()
    alpha_series = (alpha_series - alpha_series.rolling(window=60, min_periods=20).mean()) / alpha_series.rolling(window=60, min_periods=20).std()
    
    return alpha_series
