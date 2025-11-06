import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import linregress

def heuristics_v2(df):
    """
    Multi-Scale Liquidity Momentum Factor
    Combines volume liquidity, amount liquidity, price impact, and quality metrics
    across multiple timeframes with regime adaptation
    """
    data = df.copy()
    
    # Basic liquidity proxies
    data['volume_liquidity'] = data['volume'] / (data['high'] - data['low']).replace(0, np.nan)
    data['amount_liquidity'] = data['amount'] / (data['high'] - data['low']).replace(0, np.nan)
    data['liquidity_efficiency'] = data['volume_liquidity'] / data['amount_liquidity'].replace(0, np.nan)
    
    # Price impact measures
    data['daily_price_impact'] = (data['close'] - data['open']) / data['volume'].replace(0, np.nan)
    data['range_based_impact'] = (data['high'] - data['low']) / data['amount'].replace(0, np.nan)
    
    # Impact momentum (5-day changes)
    data['price_impact_momentum_5d'] = data['daily_price_impact'] - data['daily_price_impact'].shift(5)
    data['range_impact_momentum_5d'] = data['range_based_impact'] - data['range_based_impact'].shift(5)
    
    # Short-term liquidity dynamics (3-day)
    data['volume_liquidity_momentum_3d'] = data['volume_liquidity'] / data['volume_liquidity'].shift(3).replace(0, np.nan)
    data['amount_liquidity_momentum_3d'] = data['amount_liquidity'] / data['amount_liquidity'].shift(3).replace(0, np.nan)
    data['price_impact_momentum_3d'] = data['daily_price_impact'] - data['daily_price_impact'].shift(3)
    data['range_impact_momentum_3d'] = data['range_based_impact'] - data['range_based_impact'].shift(3)
    
    # Medium-term trends (10-day slopes)
    def calc_slope(series, window=10):
        slopes = pd.Series(index=series.index, dtype=float)
        for i in range(window-1, len(series)):
            if i >= window-1:
                y_values = series.iloc[i-window+1:i+1].values
                if len(y_values) == window and not np.any(np.isnan(y_values)):
                    x_values = np.arange(window)
                    slope, _, _, _, _ = linregress(x_values, y_values)
                    slopes.iloc[i] = slope
        return slopes
    
    data['volume_liquidity_trend_10d'] = calc_slope(data['volume_liquidity'], 10)
    data['amount_liquidity_trend_10d'] = calc_slope(data['amount_liquidity'], 10)
    data['price_impact_trend_10d'] = calc_slope(data['daily_price_impact'], 10)
    data['range_impact_trend_10d'] = calc_slope(data['range_based_impact'], 10)
    
    # Liquidity momentum convergence
    data['liquidity_convergence'] = (
        data['volume_liquidity_momentum_3d'] * data['volume_liquidity_trend_10d'] +
        data['amount_liquidity_momentum_3d'] * data['amount_liquidity_trend_10d']
    )
    
    # Price-liquidity interactions
    def rolling_corr(series1, series2, window=5):
        return series1.rolling(window).corr(series2)
    
    data['price_efficiency'] = abs(data['close'] - data['open']) / (data['high'] - data['low']).replace(0, np.nan)
    data['efficiency_liquidity_corr_5d'] = rolling_corr(data['price_efficiency'], data['volume_liquidity'], 5)
    data['amount_impact_corr_5d'] = rolling_corr(data['amount_liquidity'], data['daily_price_impact'], 5)
    
    # Liquidity compression-expansion
    data['range_liquidity_ratio'] = (data['high'] - data['low']) / ((data['volume'] + data['amount']) / 2).replace(0, np.nan)
    
    # Volume-amount coordination
    data['volume_amount_corr_5d'] = rolling_corr(data['volume'], data['amount'], 5)
    
    # Transaction quality metrics
    data['volume_amount_efficiency'] = data['volume'] / data['amount'].replace(0, np.nan)
    data['impact_consistency_5d'] = data['daily_price_impact'].rolling(5).std()
    data['range_efficiency'] = abs(data['close'] - data['open']) / (data['high'] - data['low']).replace(0, np.nan)
    
    # Quality momentum
    data['efficiency_trend_5d'] = data['volume_amount_efficiency'] - data['volume_amount_efficiency'].shift(5)
    data['impact_consistency_momentum'] = data['impact_consistency_5d'] - data['impact_consistency_5d'].shift(5)
    data['range_efficiency_momentum'] = data['range_efficiency'] - data['range_efficiency'].shift(5)
    
    # Regime classification and adaptive signals
    volume_median = data['volume'].rolling(20).median()
    impact_median = data['daily_price_impact'].rolling(20).median()
    amount_median = data['amount'].rolling(20).median()
    efficiency_median = data['liquidity_efficiency'].rolling(20).median()
    
    # High Volume, Low Impact regime
    high_volume_low_impact = (
        (data['volume'] > volume_median) & 
        (abs(data['daily_price_impact']) < abs(impact_median))
    )
    
    # Low Volume, High Impact regime  
    low_volume_high_impact = (
        (data['volume'] < volume_median) & 
        (abs(data['daily_price_impact']) > abs(impact_median))
    )
    
    # High Amount, Efficient regime
    high_amount_efficient = (
        (data['amount'] > amount_median) & 
        (data['liquidity_efficiency'] > efficiency_median)
    )
    
    # Low Amount, Inefficient regime
    low_amount_inefficient = (
        (data['amount'] < amount_median) & 
        (data['liquidity_efficiency'] < efficiency_median)
    )
    
    # Regime-adaptive signals
    regime_signals = pd.Series(0, index=data.index)
    
    # High Volume, Low Impact: Focus on volume momentum with efficiency correlation
    regime_signals[high_volume_low_impact] = (
        data['volume_liquidity_momentum_3d'] * data['efficiency_liquidity_corr_5d']
    )[high_volume_low_impact]
    
    # Low Volume, High Impact: Amount momentum with range liquidity
    regime_signals[low_volume_high_impact] = (
        data['amount_liquidity_momentum_3d'] * data['range_liquidity_ratio']
    )[low_volume_high_impact]
    
    # High Amount, Efficient: Amount trend with price efficiency
    regime_signals[high_amount_efficient] = (
        data['amount_liquidity_trend_10d'] * data['price_efficiency']
    )[high_amount_efficient]
    
    # Low Amount, Inefficient: Compression signals with impact momentum
    regime_signals[low_amount_inefficient] = (
        (1 / data['range_liquidity_ratio']) * data['price_impact_momentum_5d']
    )[low_amount_inefficient]
    
    # Quality enhancement
    quality_filter = (
        (data['volume_amount_efficiency'] > data['volume_amount_efficiency'].rolling(20).quantile(0.3)) &
        (data['impact_consistency_5d'] < data['impact_consistency_5d'].rolling(20).quantile(0.7)) &
        (data['range_efficiency'] > 0.1)
    )
    
    # Composite factor with quality filtering
    composite_factor = (
        regime_signals * 
        data['liquidity_convergence'] * 
        data['volume_amount_corr_5d'] *
        (1 + data['efficiency_trend_5d'])
    )
    
    # Apply quality filter
    final_factor = composite_factor.where(quality_filter, composite_factor * 0.5)
    
    # Normalize and clean
    final_factor = final_factor.replace([np.inf, -np.inf], np.nan)
    final_factor = (final_factor - final_factor.rolling(60).mean()) / final_factor.rolling(60).std()
    
    return final_factor
