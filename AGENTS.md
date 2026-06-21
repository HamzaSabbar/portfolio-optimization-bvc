# AGENTS.md

## Project context
This project is an academic portfolio optimization application for stocks listed on the Casablanca Stock Exchange.

The system follows this pipeline:
Data collection → data cleaning → financial features → logistic regression scoring → constrained portfolio optimization → historical bootstrap Monte Carlo simulation → Streamlit application.

## Technical stack
Use Python 3.11.
Use pandas, numpy, scikit-learn, scipy, streamlit, plotly, joblib and pytest.
Prefer simple, readable and well-documented code.

## Financial methodology
Do not invent another methodology unless explicitly requested.
Use:
- historical daily returns
- financial indicators: past return, momentum, volatility, RSI, MACD, moving averages, average volume, drawdown
- target variable: future 20-day return greater than 2%
- chronological train/test split
- logistic regression for scoring
- SLSQP optimization
- no short selling
- maximum weight per asset
- risk-aversion parameter based on user risk profile
- historical bootstrap Monte Carlo simulation

## Coding rules
Each module must be independent and testable.
Avoid hardcoded absolute paths.
All outputs must be saved in data/processed or data/outputs.
Add docstrings to every public function.
Add error handling for missing columns, empty dataframes, and invalid user inputs.

## Streamlit app
The app must allow the user to enter:
- budget
- investment horizon
- risk profile
- target return

The app must display:
- stock scores
- recommended allocation
- amounts to invest
- portfolio risk
- Monte Carlo simulation results
- probability of reaching the target