# cross-chain-liquidity-analysis
Empirical analysis of cross-chain liquidity provision and arbitrage dynamics using USDC/ETH pools across Ethereum, Arbitrum, and Polygon with SVAR and event studies.

This repository contains my MSc Finance dissertation project analyzing **cross-chain liquidity provision** and **arbitrage dynamics** using the USDC/ETH 0.05% pool across **Ethereum, Arbitrum, and Polygon**.  The Uniswap v3 data was gathered from The Graph subgraph API requests.

## 🔹 Overview
- Examines how arbitrage affects cross-chain price convergence.  
- Tests **which chain leads in price discovery**.  
- Studies **liquidity provider (LP) behavior** across chains.  
- Applies **event study methods** and **Structural VAR (SVAR)** analysis.  

## 🔹 Data
- Uniswap v3 pool data (swaps, mints, burns).  
- Time-series datasets across Ethereum, Arbitrum, and Polygon.  
*(Data not included — please use provided scripts to fetch from source.)*

## 🔹 Methodology
- **Event Study** on cross-chain arbitrage events.  
- **Structural VAR (SVAR)** to test lead-lag dynamics.  
- **Liquidity analysis** across LP actions.  

## 🔹 Tools
- Python (pandas, numpy, statsmodels, matplotlib, seaborn)  
- SQL  
- Jupyter Notebooks  

## 🔹 Results
- Price convergence is faster on L2 chains during arbitrage shocks.  
- Ethereum often leads in discovery but L2 dominance emerges in high-volatility periods.  
- LPs rebalance liquidity significantly after arbitrage events.
  
## 🔹 Author
Shambhvi Lokre — MSc Finance, Warwick Business School  


