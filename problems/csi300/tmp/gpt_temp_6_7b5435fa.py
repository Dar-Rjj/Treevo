import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Multi-Scale Price-Volume Synchronization Divergence
    # Calculate Mid-Price Momentum
    data['mid_price'] = (data['high'] + data['low']) / 2
    data['mid_price_5d_return'] = (data['mid_price'] - data['mid_price'].shift(5)) / data['mid_price'].shift(5)
    data['mid_price_10d_return'] = (data['mid_price'] - data['mid_price'].shift(10)) / data['mid_price'].shift(10)
    
    # Calculate Directional Volume Flow
    data['price_range'] = data['high'] - data['low']
    data['directional_movement'] = (data['close'] - data['open']) / (data['price_range'] + 1e-8)
    data['directional_volume_flow'] = data['directional_movement'] * data['volume']
    
    # Detect Synchronization Divergence Across Timeframes
    # 5-day synchronization
    data['volume_flow_5d_trend'] = data['directional_volume_flow'].rolling(window=5).mean()
    data['sync_5d_corr'] = data['mid_price_5d_return'].rolling(window=5).corr(data['volume_flow_5d_trend'])
    
    # 10-day synchronization
    data['volume_flow_10d_trend'] = data['directional_volume_flow'].rolling(window=10).mean()
    data['sync_10d_corr'] = data['mid_price_10d_return'].rolling(window=10).corr(data['volume_flow_10d_trend'])
    
    # Multi-Timeframe Synchronization Analysis
    data['positive_sync_count'] = ((data['sync_5d_corr'] > 0).astype(int) + 
                                  (data['sync_10d_corr'] > 0).astype(int))
    data['negative_sync_count'] = ((data['sync_5d_corr'] < 0).astype(int) + 
                                  (data['sync_10d_corr'] < 0).astype(int))
    data['net_synchronization'] = data['positive_sync_count'] - data['negative_sync_count']
    
    # Volatility-Regime Identification with Asymmetry
    # Calculate Bidirectional Volatility
    data['returns'] = data['close'].pct_change()
    
    # Upside volatility (positive returns)
    upside_returns = data['returns'].copy()
    upside_returns[upside_returns < 0] = np.nan
    data['upside_volatility'] = upside_returns.rolling(window=30).std()
    
    # Downside volatility (negative returns)
    downside_returns = data['returns'].copy()
    downside_returns[downside_returns > 0] = np.nan
    data['downside_volatility'] = downside_returns.rolling(window=30).std()
    
    # Calculate Volatility Momentum
    # True Range calculation
    data['true_range'] = np.maximum(
        data['high'] - data['low'],
        np.maximum(
            abs(data['high'] - data['close'].shift(1)),
            abs(data['low'] - data['close'].shift(1))
        )
    )
    data['atr_10d'] = data['true_range'].rolling(window=10).mean()
    data['volatility_momentum'] = (data['atr_10d'] - data['atr_10d'].shift(5)) / (data['atr_10d'].shift(5) + 1e-8)
    
    # Regime Classification
    data['volatility_asymmetry'] = (data['upside_volatility'] / (data['downside_volatility'] + 1e-8)) - 1
    
    # Volatility Persistence Assessment
    squared_returns = data['returns'] ** 2
    data['volatility_persistence'] = squared_returns.rolling(window=20).apply(
        lambda x: x.autocorr(), raw=False
    )
    
    # Assign Regime Flags
    conditions = [
        (data['volatility_asymmetry'] > 0.2) & (data['volatility_momentum'] > 0),
        (data['volatility_asymmetry'] < -0.2) & (data['volatility_momentum'] > 0)
    ]
    choices = [1, -1]  # 1: Bull, -1: Bear
    data['regime'] = np.select(conditions, choices, default=0)  # 0: Neutral
    
    # Efficiency-Enhanced Signal Processing
    # Price-Volume Efficiency
    data['price_efficiency'] = (data['close'] - data['open']) / (data['price_range'] + 1e-8)
    data['volume_weighted_efficiency'] = data['price_efficiency'] * data['volume']
    data['vw_efficiency_5d'] = data['volume_weighted_efficiency'].rolling(window=5).mean()
    
    # Breakout Efficiency
    data['high_5d_max'] = data['high'].rolling(window=5).max().shift(1)
    data['low_5d_min'] = data['low'].rolling(window=5).min().shift(1)
    data['breakout_magnitude'] = (
        np.maximum(0, data['high'] - data['high_5d_max']) + 
        np.maximum(0, data['low_5d_min'] - data['low'])
    )
    data['breakout_efficiency'] = data['breakout_magnitude'] / (data['true_range'] + 1e-8)
    
    # Intraday Behavior
    data['opening_gap'] = (data['open'] - data['close'].shift(1)) / (data['close'].shift(1) + 1e-8)
    data['gap_persistence'] = np.sign(data['opening_gap']) * np.sign(data['close'] - data['open'])
    
    data['morning_momentum'] = (data['high'] - data['open']) / (data['open'] + 1e-8)
    data['afternoon_momentum'] = (data['close'] - data['low']) / (data['low'] + 1e-8)
    data['intraday_momentum_ratio'] = data['morning_momentum'] / (data['afternoon_momentum'] + 1e-8)
    
    # Generate Efficiency Score
    data['efficiency_score'] = np.cbrt(
        (data['vw_efficiency_5d'] + data['breakout_efficiency']) * 
        (1 + data['gap_persistence'])
    )
    
    # Regime-Adaptive Signal Combination
    # Bull Market Processing
    bull_mask = data['regime'] == 1
    data['regime_adjusted_sync'] = data['net_synchronization'].copy()
    data.loc[bull_mask, 'regime_adjusted_sync'] = data.loc[bull_mask, 'net_synchronization'] * 1.5
    data.loc[bull_mask, 'regime_adjusted_sync'] = data.loc[bull_mask, 'regime_adjusted_sync'] * (1 + data.loc[bull_mask, 'breakout_efficiency'])
    
    # Bear Market Adjustment
    bear_mask = data['regime'] == -1
    data.loc[bear_mask, 'regime_adjusted_sync'] = data.loc[bear_mask, 'net_synchronization'] * 0.7
    
    # Neutral Market Optimization
    neutral_mask = data['regime'] == 0
    efficiency_threshold = data['efficiency_score'].rolling(window=20).mean()
    data.loc[neutral_mask, 'regime_adjusted_sync'] = data.loc[neutral_mask, 'net_synchronization'] * (
        1 + data.loc[neutral_mask, 'volatility_persistence']
    )
    
    # Final Composite Alpha Generation
    # Calculate Momentum Strength
    data['momentum_strength'] = np.cbrt(
        (abs(data['mid_price_5d_return']) + abs(data['mid_price_10d_return'])) / 2
    )
    
    # Incorporate Liquidity Dynamics
    data['price_impact'] = data['price_range'] / (data['volume'] + 1e-8)
    data['vwap_impact_10d'] = data['price_impact'].rolling(window=10).mean()
    data['liquidity_pressure'] = (data['vwap_impact_10d'] / 
                                 data['price_impact'].rolling(window=60).median()) - 1
    
    # Regime-Weighted Signal Integration
    data['composite_signal'] = (
        data['regime_adjusted_sync'] * 
        data['efficiency_score'] * 
        data['momentum_strength'] * 
        (1 - data['liquidity_pressure'])
    )
    
    # Apply hyperbolic tangent for bounded output
    data['final_factor'] = np.tanh(data['composite_signal'])
    
    return data['final_factor']
