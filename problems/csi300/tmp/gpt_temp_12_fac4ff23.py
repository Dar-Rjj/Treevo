import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Transactional Liquidity Assessment
    data['volume_liquidity'] = data['volume'] / (data['high'] - data['low']).replace(0, np.nan)
    data['amount_liquidity'] = data['amount'] / (data['high'] - data['low']).replace(0, np.nan)
    
    # Price Impact Liquidity
    data['price_impact'] = (data['close'] - data['open']) / data['volume'].replace(0, np.nan)
    data['range_impact'] = (data['high'] - data['low']) / data['amount'].replace(0, np.nan)
    
    # Liquidity Regime Classification
    volume_median = data['volume_liquidity'].rolling(window=20, min_periods=10).median()
    impact_median = data['price_impact'].abs().rolling(window=20, min_periods=10).median()
    
    data['high_volume_low_impact'] = ((data['volume_liquidity'] > volume_median) & 
                                    (data['price_impact'].abs() < impact_median)).astype(int)
    data['low_volume_high_impact'] = ((data['volume_liquidity'] < volume_median) & 
                                    (data['price_impact'].abs() > impact_median)).astype(int)
    
    # Multi-Timeframe Liquidity Momentum
    # Short-term (3-day)
    data['volume_liquidity_momentum'] = (data['volume_liquidity'] / 
                                       data['volume_liquidity'].shift(2)).replace([np.inf, -np.inf], np.nan)
    data['amount_liquidity_momentum'] = (data['amount_liquidity'] / 
                                       data['amount_liquidity'].shift(2)).replace([np.inf, -np.inf], np.nan)
    
    # Medium-term (10-day trends using linear regression slope)
    def linear_slope(series, window):
        slopes = pd.Series(index=series.index, dtype=float)
        for i in range(window-1, len(series)):
            if i >= window-1:
                y = series.iloc[i-window+1:i+1].values
                x = np.arange(len(y))
                if len(y) == window and not np.any(np.isnan(y)):
                    slope = np.polyfit(x, y, 1)[0]
                    slopes.iloc[i] = slope
        return slopes
    
    data['volume_liquidity_trend'] = linear_slope(data['volume_liquidity'], 10)
    data['amount_liquidity_trend'] = linear_slope(data['amount_liquidity'], 10)
    
    # Price-Liquidity Interaction Patterns
    data['range_efficiency'] = (data['close'] - data['open']).abs() / (data['high'] - data['low']).replace(0, np.nan)
    
    # 5-day correlation between range efficiency and volume liquidity
    data['efficiency_liquidity_corr'] = data['range_efficiency'].rolling(window=5).corr(data['volume_liquidity'])
    
    # Liquidity Compression-Expansion
    data['range_liquidity_ratio'] = (data['high'] - data['low']) / ((data['volume'] + data['amount']) / 2).replace(0, np.nan)
    
    # Regime-Adaptive Liquidity Signals
    data['high_volume_signal'] = data['volume_liquidity_momentum'] * data['efficiency_liquidity_corr']
    data['low_volume_signal'] = data['amount_liquidity_momentum'] * data['range_liquidity_ratio']
    
    # Liquidity Quality Assessment
    data['volume_amount_efficiency'] = data['volume'] / data['amount'].replace(0, np.nan)
    data['range_efficiency_momentum'] = data['range_efficiency'].pct_change(5)
    data['volume_amount_trend'] = data['volume_amount_efficiency'].pct_change(5)
    
    # Composite Liquidity Alpha Factor
    # Multi-regime signal integration
    regime_signal = (data['high_volume_low_impact'] * data['high_volume_signal'] + 
                    data['low_volume_high_impact'] * data['low_volume_signal'])
    
    # Liquidity momentum synthesis
    momentum_signal = (data['volume_liquidity_momentum'].rank(pct=True) + 
                      data['amount_liquidity_momentum'].rank(pct=True) + 
                      data['volume_liquidity_trend'].rank(pct=True) + 
                      data['amount_liquidity_trend'].rank(pct=True)) / 4
    
    # Quality and consistency filters
    quality_filter = (data['range_efficiency_momentum'].rank(pct=True) + 
                     data['volume_amount_trend'].rank(pct=True)) / 2
    
    # Final composite factor
    composite_factor = (regime_signal.fillna(0) * 0.4 + 
                       momentum_signal.fillna(0) * 0.4 + 
                       quality_filter.fillna(0) * 0.2)
    
    # Normalize and clean
    final_factor = composite_factor.replace([np.inf, -np.inf], np.nan)
    final_factor = (final_factor - final_factor.rolling(window=20, min_periods=10).mean()) / final_factor.rolling(window=20, min_periods=10).std()
    
    return final_factor
