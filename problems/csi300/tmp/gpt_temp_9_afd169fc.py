import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Residual Momentum with Volume Confirmation factor
    Combines residual returns from market model with volume confirmation signals
    """
    # Calculate daily returns
    returns = df['close'].pct_change()
    
    # Create market proxy (equally weighted portfolio of all stocks)
    # In practice, this would be replaced with actual market index returns
    market_returns = returns.rolling(window=20, min_periods=10).mean()
    
    # Calculate residual returns using rolling market regression
    residuals = pd.Series(index=df.index, dtype=float)
    
    for i in range(len(df)):
        if i < 60:  # Need sufficient history for regression
            residuals.iloc[i] = 0
            continue
            
        # Use past 60 days for regression (current and historical only)
        lookback = 60
        start_idx = max(0, i - lookback + 1)
        end_idx = i + 1
        
        stock_ret = returns.iloc[start_idx:end_idx]
        market_ret = market_returns.iloc[start_idx:end_idx]
        
        # Remove NaN values
        valid_mask = (~stock_ret.isna()) & (~market_ret.isna())
        if valid_mask.sum() < 30:  # Minimum observations
            residuals.iloc[i] = 0
            continue
            
        stock_ret_valid = stock_ret[valid_mask]
        market_ret_valid = market_ret[valid_mask]
        
        # Simple regression: stock_return = alpha + beta * market_return
        beta = np.cov(stock_ret_valid, market_ret_valid)[0, 1] / np.var(market_ret_valid)
        alpha = np.mean(stock_ret_valid) - beta * np.mean(market_ret_valid)
        
        # Calculate residual for current period
        predicted_return = alpha + beta * market_returns.iloc[i]
        residuals.iloc[i] = returns.iloc[i] - predicted_return
    
    # Calculate volume trend and confirmation
    volume_ma = df['volume'].rolling(window=20, min_periods=10).mean()
    volume_ratio = df['volume'] / volume_ma
    volume_signal = np.where(volume_ratio > 1.2, 1.0,  # High volume confirmation
                   np.where(volume_ratio < 0.8, 0.5,   # Low volume discount
                   0.8))  # Normal volume
    
    # Volume-weighted residuals
    volume_weighted_residuals = residuals * volume_signal
    
    # Calculate recent residual momentum (5-day average)
    residual_momentum = volume_weighted_residuals.rolling(window=5, min_periods=3).mean()
    
    # Final alpha factor: recent residual momentum
    alpha_factor = residual_momentum
    
    return alpha_factor
