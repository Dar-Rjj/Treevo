import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Multi-Timeframe Price-Volume Divergence with Liquidity Acceleration factor
    """
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Initialize result series
    result = pd.Series(index=df.index, dtype=float)
    
    # Price-Volume Divergence Patterns
    # Directional Divergence
    df['price_5d_return'] = df['close'].pct_change(5)
    df['volume_5d_change'] = df['volume'].pct_change(5)
    df['price_10d_return'] = df['close'].pct_change(10)
    df['volume_10d_change'] = df['volume'].pct_change(10)
    
    # Directional divergence signals
    df['div_up_price_down_vol'] = ((df['price_5d_return'] > 0) & (df['volume_5d_change'] < 0)).astype(int)
    df['div_down_price_up_vol'] = ((df['price_10d_return'] < 0) & (df['volume_10d_change'] > 0)).astype(int)
    
    # Rolling price-volume correlation (20-day)
    df['price_volume_corr'] = df['close'].rolling(window=20).corr(df['volume'])
    
    # Magnitude Divergence
    df['return_magnitude'] = df['close'].pct_change().abs()
    df['volume_magnitude'] = df['volume'].pct_change().abs()
    df['magnitude_ratio'] = df['return_magnitude'] / (df['volume_magnitude'] + 1e-8)
    
    # High-low range expansion with volume contraction
    df['daily_range'] = (df['high'] - df['low']) / df['close']
    df['range_change'] = df['daily_range'].pct_change()
    df['range_exp_vol_cont'] = ((df['range_change'] > 0) & (df['volume_5d_change'] < 0)).astype(int)
    
    # Close-to-close volatility vs volume volatility
    df['price_volatility'] = df['close'].pct_change().rolling(window=10).std()
    df['volume_volatility'] = df['volume'].pct_change().rolling(window=10).std()
    df['vol_vol_ratio'] = df['price_volatility'] / (df['volume_volatility'] + 1e-8)
    
    # Liquidity Acceleration Signals
    # Bid-Ask Spread Proxy
    df['spread_estimate'] = (df['high'] - df['low']) / df['close']
    df['range_volume_ratio'] = df['daily_range'] / (df['volume'] + 1e-8)
    df['range_efficiency'] = df['close'].pct_change().abs() / (df['daily_range'] + 1e-8)
    
    # Liquidity Momentum
    df['range_eff_accel'] = df['range_efficiency'].diff(5)
    df['volume_clustering'] = (df['daily_range'].rolling(window=5).std() / 
                              (df['volume'].rolling(window=5).std() + 1e-8))
    
    # Amount per trade concentration
    df['amount_per_trade'] = df['amount'] / (df['volume'] + 1e-8)
    df['amount_trend'] = df['amount_per_trade'].rolling(window=5).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 5 else np.nan
    )
    
    # Adaptive Signal Integration
    # Market State Recognition
    # Price path complexity (turning points count)
    def count_turning_points(prices):
        if len(prices) < 3:
            return 0
        turning_points = 0
        for i in range(1, len(prices)-1):
            if (prices[i] > prices[i-1] and prices[i] > prices[i+1]) or \
               (prices[i] < prices[i-1] and prices[i] < prices[i+1]):
                turning_points += 1
        return turning_points
    
    df['price_complexity'] = df['close'].rolling(window=10).apply(
        count_turning_points, raw=True
    )
    
    # Volume distribution skewness
    df['volume_skew'] = df['volume'].rolling(window=20).skew()
    
    # Intraday persistence patterns
    df['intraday_persistence'] = ((df['close'] > df['open']).rolling(window=5).sum() / 5)
    
    # Dynamic Weighting Components
    # Divergence strength scoring
    df['divergence_strength'] = (
        df['div_up_price_down_vol'] + 
        df['div_down_price_up_vol'] + 
        (1 - df['price_volume_corr'].abs()) +
        df['range_exp_vol_cont']
    )
    
    # Liquidity signal reliability
    df['liquidity_reliability'] = (
        df['range_eff_accel'].abs() +
        (1 / (df['volume_clustering'] + 1e-8)) +
        df['amount_trend'].abs()
    )
    
    # Time-varying combination weights
    market_volatility = df['close'].pct_change().rolling(window=20).std()
    df['market_regime_weight'] = 1 / (market_volatility + 1e-8)
    
    # Final factor calculation with adaptive weighting
    price_volume_component = (
        df['divergence_strength'] * 0.3 +
        df['magnitude_ratio'] * 0.2 +
        df['vol_vol_ratio'] * 0.2
    )
    
    liquidity_component = (
        df['range_efficiency'] * 0.15 +
        df['range_eff_accel'] * 0.15
    )
    
    # Apply market regime adjustment
    regime_adjusted_pv = price_volume_component * df['market_regime_weight']
    regime_adjusted_liq = liquidity_component * df['liquidity_reliability']
    
    # Final factor with complexity adjustment
    complexity_adjustment = 1 / (df['price_complexity'] + 1)
    final_factor = (regime_adjusted_pv + regime_adjusted_liq) * complexity_adjustment
    
    # Fill NaN values with 0
    result = final_factor.fillna(0)
    
    return result
