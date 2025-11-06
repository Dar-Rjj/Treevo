import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy import stats

def heuristics_v2(df):
    """
    Fractal Correlation Dimension Breakout factor
    Detects dimension collapse preceding volatility explosions by measuring
    fractal structure stability across scales weighted by dimension change rate asymmetry
    """
    data = df.copy()
    
    # Calculate daily returns and volatility
    data['returns'] = data['close'].pct_change()
    data['volatility'] = data['returns'].rolling(window=20).std()
    
    # Multi-scale correlation dimension calculation
    def correlation_dimension(price_series, scale_factor=0.1):
        """Calculate correlation dimension using box-counting method"""
        if len(price_series) < 10:
            return np.nan
            
        # Normalize price series
        normalized_prices = (price_series - price_series.min()) / (price_series.max() - price_series.min() + 1e-12)
        
        # Multiple scales for fractal analysis
        scales = [0.01, 0.02, 0.05, 0.1, 0.2]
        counts = []
        
        for scale in scales:
            # Box counting at different scales
            bins = np.arange(0, 1 + scale, scale)
            hist, _ = np.histogram(normalized_prices, bins=bins)
            non_empty_boxes = np.sum(hist > 0)
            counts.append(non_empty_boxes)
        
        # Linear regression in log-log space for dimension estimation
        if len([c for c in counts if c > 0]) < 3:
            return np.nan
            
        log_scales = np.log([s for s, c in zip(scales, counts) if c > 0])
        log_counts = np.log([c for c in counts if c > 0])
        
        if len(log_scales) > 1:
            slope, _, r_value, _, _ = stats.linregress(log_scales, log_counts)
            return -slope if r_value**2 > 0.8 else np.nan
        return np.nan
    
    # Calculate correlation dimension over rolling windows
    window_size = 20
    data['corr_dim'] = data['close'].rolling(window=window_size).apply(
        lambda x: correlation_dimension(x), raw=False
    )
    
    # Dimension change rate and asymmetry
    data['dim_change'] = data['corr_dim'].diff()
    data['dim_change_abs'] = data['dim_change'].abs()
    
    # Asymmetric dimension collapse detection
    data['dim_collapse_signal'] = 0
    collapse_condition = (
        (data['dim_change'] < -0.1) & 
        (data['dim_change_abs'] > data['dim_change_abs'].rolling(10).mean())
    )
    data.loc[collapse_condition, 'dim_collapse_signal'] = 1
    
    # Volatility explosion detection
    data['vol_explosion'] = (
        data['volatility'] > data['volatility'].rolling(50).quantile(0.8)
    ).astype(int)
    
    # Lead-lag relationship between dimension collapse and volatility
    data['collapse_lead_vol'] = data['dim_collapse_signal'].shift(1).rolling(5).sum()
    
    # Factor construction: dimension stability weighted by change rate asymmetry
    data['dimension_stability'] = 1 / (1 + data['corr_dim'].rolling(10).std())
    data['change_asymmetry'] = (
        data['dim_change'].rolling(10).apply(lambda x: np.mean(x[x > 0]) - np.mean(x[x < 0]), raw=False)
    ).fillna(0)
    
    # Final factor: combination of dimension collapse signals and stability measures
    factor = (
        data['collapse_lead_vol'] * 
        data['dimension_stability'] * 
        (1 + data['change_asymmetry'])
    )
    
    # Normalize the factor
    factor = (factor - factor.rolling(252).mean()) / (factor.rolling(252).std() + 1e-12)
    
    return factor
