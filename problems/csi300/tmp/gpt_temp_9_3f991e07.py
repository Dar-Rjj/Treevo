import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Cross-Asset Relative Momentum with Microstructure Anchoring alpha factor
    """
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate basic price returns
    data['ret_1'] = data['close'] / data['close'].shift(1) - 1
    data['ret_5'] = data['close'] / data['close'].shift(5) - 1
    data['price_range'] = (data['high'] - data['low']) / data['close']
    
    # Cross-Asset Relative Positioning Components
    # Since we don't have sector/market indices, use rolling market proxies
    market_ret_5 = data['close'].pct_change(5).rolling(window=20, min_periods=10).mean()
    market_ret_1 = data['close'].pct_change(1).rolling(window=20, min_periods=10).mean()
    
    # Intra-Sector Relative Strength (using market proxy)
    data['intra_sector_rs'] = (data['close'] / data['close'].shift(5)) / (1 + market_ret_5)
    
    # Cross-Sector Momentum Flow
    data['cross_sector_momentum'] = np.sign(data['ret_5'] - market_ret_5) * np.abs(data['ret_5'] - market_ret_5)
    
    # Sector Leadership Persistence
    leadership_count = pd.Series(index=data.index, dtype=float)
    for i in range(5, len(data)):
        window = data.iloc[i-4:i+1]
        count = ((window['close'] / window['close'].shift(1)) > 
                (1 + market_ret_1.loc[window.index])).sum()
        leadership_count.iloc[i] = count
    data['sector_leadership'] = leadership_count
    
    # Market Regime Relative Behavior
    market_range_5 = data['price_range'].rolling(window=20, min_periods=10).mean()
    data['risk_on_sensitivity'] = (data['price_range'] / data['price_range'].shift(5)) / (1 + market_range_5.pct_change(5))
    
    # Market Correlation Regime
    market_volume_1 = data['volume'].pct_change(1).rolling(window=20, min_periods=10).mean()
    data['market_correlation'] = np.sign(data['ret_1'] - market_ret_1) * (data['volume'] / data['volume'].shift(1))
    
    # Microstructure Anchoring Framework
    # Opening Auction Dynamics
    gap_threshold = 0.02 * data['close'].shift(1)
    gap_condition = np.abs(data['open'] - data['close'].shift(1)) > gap_threshold
    data['opening_gap_absorption'] = np.where(
        gap_condition,
        (data['close'] - data['open']) / (data['open'] - data['close'].shift(1)),
        0
    )
    
    # Since we don't have intraday data, approximate first hour range using opening dynamics
    data['first_hour_range_capture'] = np.abs(data['open'] - data['close'].shift(1)) / data['price_range']
    
    # Closing Auction Dynamics
    # Approximate final hour momentum using last hour of trading
    data['final_hour_momentum'] = (data['close'] - data['open']) / data['price_range']
    
    # End-of-Day Price Discovery (using VWAP approximation)
    vwap_approx = (data['high'] + data['low'] + data['close']) / 3
    data['eod_price_discovery'] = np.abs(data['close'] - vwap_approx) / data['price_range']
    
    # Liquidity Regime Classification
    # Approximate spread using price range (since we don't have bid-ask)
    data['relative_spread'] = data['price_range'] * 0.1  # Approximation
    
    # Market Depth Analysis (using volume as proxy)
    volume_ma_5 = data['volume'].rolling(window=5, min_periods=3).mean()
    data['depth_imbalance'] = (data['volume'] - volume_ma_5) / (data['volume'] + volume_ma_5)
    
    # Depth-Momentum Divergence
    data['depth_momentum_divergence'] = np.sign(data['ret_1']) * data['depth_imbalance']
    
    # Cross-Timeframe Momentum Alignment
    data['multi_scale_convergence'] = np.sign(data['ret_1']) * np.sign(data['ret_5'])
    
    # Momentum Acceleration
    data['momentum_acceleration'] = data['ret_1'] - data['ret_1'].shift(1)
    
    # Volatility-Adjusted Momentum
    data['vol_adj_momentum'] = data['ret_1'] / data['price_range']
    
    # Relative Value Anchoring
    # Spread-Adjusted Returns
    data['spread_adj_returns'] = data['ret_1'] / (data['relative_spread'] + 1e-8)
    
    # Depth-Weighted Momentum
    data['depth_weighted_momentum'] = (data['close'] - data['close'].shift(1)) * data['depth_imbalance']
    
    # Auction Efficiency
    data['auction_efficiency'] = ((data['close'] - data['open']) / data['price_range']) * data['first_hour_range_capture']
    
    # Composite Alpha Construction
    
    # Cross-Asset Relative Momentum Score
    sector_component = data['intra_sector_rs'] * data['sector_leadership']
    market_regime_component = data['risk_on_sensitivity'] * data['market_correlation']
    timeframe_component = data['multi_scale_convergence'] * data['momentum_acceleration']
    
    cross_asset_momentum = (
        0.4 * sector_component + 
        0.3 * market_regime_component + 
        0.3 * timeframe_component
    )
    
    # Microstructure Anchoring Score
    auction_quality = data['opening_gap_absorption'] * data['final_hour_momentum']
    liquidity_regime = data['relative_spread'] * data['depth_momentum_divergence']
    value_signals = data['spread_adj_returns'] * data['auction_efficiency']
    
    microstructure_anchoring = (
        0.4 * auction_quality + 
        0.3 * liquidity_regime + 
        0.3 * value_signals
    )
    
    # Final Alpha Integration
    # Liquidity regime weighting
    liquidity_weight = 1 / (1 + np.abs(data['depth_imbalance']))
    
    # Relative value adjustment
    value_adjustment = data['vol_adj_momentum'] * data['depth_weighted_momentum']
    
    # Multi-timeframe validation
    timeframe_validation = np.sign(data['ret_1']) * np.sign(data['ret_5']) * np.sign(data['momentum_acceleration'])
    
    # Final alpha factor
    alpha = (
        cross_asset_momentum * microstructure_anchoring * 
        liquidity_weight * (1 + 0.1 * value_adjustment) * 
        timeframe_validation
    )
    
    # Clean and normalize
    alpha = alpha.replace([np.inf, -np.inf], np.nan)
    alpha = alpha.fillna(method='ffill').fillna(0)
    
    # Remove any potential lookahead bias by ensuring no future data
    alpha = alpha.shift(1)  # Use previous day's calculated alpha
    
    return alpha
