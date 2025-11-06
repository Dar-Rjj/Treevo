import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(data):
    """
    Cross-Asset Relative Pressure & Liquidity Dynamics alpha factor
    """
    df = data.copy()
    
    # Calculate basic components
    df['net_tick_pressure'] = (df['amount'] * (df['close'] - df['open']) / (df['high'] - df['low'])).fillna(0)
    df['price_change'] = df['close'] - df['open']
    df['return_3d'] = df['close'] / df['close'].shift(3) - 1
    df['return_5d'] = df['close'] - df['close'].shift(5)
    
    # Cross-asset calculations (using rolling cross-sectional statistics)
    def cross_asset_median(series, window=5):
        return series.rolling(window).apply(lambda x: np.median(x), raw=True)
    
    def cross_asset_mean(series, window=5):
        return series.rolling(window).apply(lambda x: np.mean(x), raw=True)
    
    # Cross-asset pressure components
    df['cross_asset_net_pressure'] = cross_asset_mean(df['net_tick_pressure'])
    df['cross_asset_median_volume'] = cross_asset_median(df['volume'])
    df['cross_asset_median_return'] = cross_asset_median(df['return_3d'])
    df['cross_asset_median_close'] = cross_asset_median(df['close'])
    
    # Relative Tick Pressure
    df['relative_tick_pressure'] = df['net_tick_pressure'] - df['cross_asset_net_pressure']
    
    # Liquidity metrics
    df['relative_liquidity_score'] = df['volume'] / df['cross_asset_median_volume']
    
    # Liquidity volatility (5-day rolling)
    df['volume_std_5d'] = df['volume'].rolling(5).std()
    df['volume_mean_5d'] = df['volume'].rolling(5).mean()
    df['liquidity_volatility'] = df['volume_std_5d'] / df['volume_mean_5d']
    df['cross_asset_liquidity_vol'] = cross_asset_median(df['liquidity_volatility'])
    
    # Price Impact Efficiency
    df['price_impact_efficiency'] = np.abs(df['price_change']) / (df['volume'] / df['cross_asset_median_volume'])
    df['price_impact_efficiency'] = df['price_impact_efficiency'].replace([np.inf, -np.inf], 0).fillna(0)
    
    # Liquidity regime classification
    conditions = [
        (df['relative_liquidity_score'] > 1.3) & (df['liquidity_volatility'] < 0.8 * df['cross_asset_liquidity_vol']),
        (df['relative_liquidity_score'] < 0.7) | (df['liquidity_volatility'] > 1.5 * df['cross_asset_liquidity_vol'])
    ]
    choices = [1.3, 0.6]  # High, Low liquidity weights
    df['liquidity_regime_weight'] = np.select(conditions, choices, default=1.0)
    
    # Cross-asset momentum components
    df['relative_price_acceleration'] = df['return_3d'] - df['cross_asset_median_return']
    df['liquidity_adjusted_momentum'] = df['return_5d'] * df['relative_liquidity_score']
    
    # Cross-market trend persistence
    df['sign_vs_median'] = np.sign(df['close'] - df['cross_asset_median_close'])
    df['trend_persistence'] = df['sign_vs_median'].rolling(5).apply(
        lambda x: np.sum(x == x.iloc[-1]) if len(x) == 5 else 0, raw=False
    )
    
    # Volume-liquidity confirmation
    df['volume_sum_5d'] = df['volume'].rolling(5).sum()
    df['cross_asset_volume_5d'] = cross_asset_median(df['volume_sum_5d'])
    df['relative_volume_trend'] = (
        np.sign(df['volume_sum_5d'] - df['cross_asset_volume_5d']) * 
        np.sign(df['close'] - df['cross_asset_median_close'])
    )
    
    # Liquidity spike divergence
    df['liquidity_spike_divergence'] = (
        df['relative_liquidity_score'] - df['relative_liquidity_score'].shift(1)
    )
    
    # Cross-asset pressure correlation
    df['pressure_correlation'] = df['net_tick_pressure'].rolling(5).corr(df['cross_asset_net_pressure'])
    
    # Cross-market adjustment based on correlation
    conditions_corr = [
        df['pressure_correlation'] > 0.7,
        df['pressure_correlation'] < 0.3
    ]
    choices_corr = [1.4, 0.7]
    df['cross_market_adjustment'] = np.select(conditions_corr, choices_corr, default=1.0)
    
    # Core factor construction
    df['base_factor'] = df['liquidity_adjusted_momentum'] * df['relative_volume_trend']
    df['efficiency_enhancement'] = df['base_factor'] * df['price_impact_efficiency']
    
    # Primary signal
    df['primary_signal'] = (
        df['efficiency_enhancement'] * 
        df['liquidity_regime_weight'] * 
        df['cross_market_adjustment']
    )
    
    # Confidence assessment
    df['cross_asset_liquidity_spike'] = cross_asset_median(df['liquidity_spike_divergence'])
    
    confidence_conditions = [
        (df['trend_persistence'] >= 3) & (df['liquidity_spike_divergence'] > df['cross_asset_liquidity_spike']),
        (df['trend_persistence'] >= 2) | (df['liquidity_spike_divergence'] > df['cross_asset_liquidity_spike'])
    ]
    confidence_multipliers = [1.6, 1.1]
    df['confidence_multiplier'] = np.select(confidence_conditions, confidence_multipliers, default=0.9)
    
    # Final alpha signal
    df['alpha_signal'] = df['primary_signal'] * df['confidence_multiplier']
    
    # Handle any remaining NaN values
    df['alpha_signal'] = df['alpha_signal'].fillna(0)
    
    return df['alpha_signal']
