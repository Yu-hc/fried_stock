# Documentation and structure for project

This is the project main structure, please follow the following format and architecture. For additional information, refer to [this conversation](https://gemini.google.com/app/e99eeef9ca53a39a)

---

## File structure

``` text
stock_proj/
├── modules/                # Core logic and Python classes
│   ├── __init__.py
│   ├── data_processor.py   # Module A: Preprocessing & Strategy
│   ├── environment.py      # Module B: Gymnasium Wrapper
│   ├── rewards.py          # Module C: Reward Logic
│   ├── model_agent.py      # Module D: RL Architecture
│   └── evaluator.py        # Module E: Backtesting & Metrics
│
├── data/                   # Raw and processed CSV datasets
├── models/                 # Saved weights (.zip or .pth)
├── notebooks/              # Jupyter notebooks for experimentation
│   ├── 01_EDA_and_Test.ipynb
│   └── 02_Training_Run.ipynb
│
├── main.py                 # (Optional) Main entry point for execution
├── .venv/
├── .gitignore
├── documentation.md
├── requirements.txt
└── README.md

```

---

## 🟢 Module A: Data Preprocessing & Strategy Signals

**Role:** The "Feature Factory." It cleans raw data and adds "heuristic" signals (Buy/Sell indicators) that the RL agent can use as hints.

* **Input Format:**
  * `Raw_Data` (Pandas DataFrame): Standard OHLCV (Open, High, Low, Close, Volume) data.
  * *Columns:* `['Date', 'Open', 'High', 'Low', 'Close', 'Volume']`

* **Output Format:**
  * `Processed_DF` (Pandas DataFrame):
    * **Original Columns:** OHLCV.
    * **Technical Features:** Normalized indicators (e.g., `RSI / 100`, `MACD_scaled`).
    * **Heuristic Signals:** Binary or categorical columns:
      * `signal_buy`: (1 if logic met, else 0)
      * `signal_sell`: (1 if logic met, else 0)
  * *Index:* `DatetimeIndex`.

---

## 🔵 Module B: The Trading Environment (Gymnasium)

**Role:** The "Simulation Engine." It tracks the portfolio, executes trades, and moves time forward.

* **Input Format:**
  * `data` (Module A's Output): The processed DataFrame.
  * `initial_balance` (Float): e.g., `10000.0`.
  * `action` (Integer): Received from the RL Agent (Module D) during each `.step()`.

* **Output Format (via `.step()` function):**
  * `observation` (NumPy Array): A slice of the features (e.g., last 30 days of data) with shape `(window_size, num_features)`.
  * `reward` (Float): The value calculated by Module C.
  * `done` (Boolean): True if end of data or account bankrupt.
  * `info` (Dictionary): `{'net_worth': 10500, 'shares_held': 10, 'trades_count': 5}`.

---

## 🟡 Module C: Reward & Penalty Function

**Role:** The "Feedback Loop." It interprets the environment's state changes into a mathematical score.

* **Input Format:**
  * `current_nav` (Float): Current Net Asset Value.
  * `previous_nav` (Float): Net Asset Value from the last time step.
  * `transaction_costs` (Float): Cost incurred by the current action.
  * `risk_metrics` (Optional): Current drawdown or volatility measure.

* **Output Format:**
  * `reward` (Float): A single scalar value (e.g., `0.05` for profit, `-1.0` for a bad trade).

---

## 🔴 Module D: Reinforcement Learning Model

**Role:** The "Decision Maker." It learns which states lead to the highest cumulative rewards.

* **Input Format:**
  * `observation_space` (Shape from Module B): e.g., `Box(low=-inf, high=inf, shape=(30, 15))`.

* **Output Format:**
  * `action` (Integer):
    * `0`: Hold (Stay in cash or maintain current position).
    * `1`: Buy/Long (Enter market or increase position).
    * `2`: Sell/Close (Exit market or go flat).
  * `action_probability` (Optional): Confidence score for the action (useful for analysis).

---

## 🟣 Module E: Backtesting & Result Evaluation

**Role:** The "Audit Department." It produces the final reports.

* **Input Format:**
  * `trade_log` (List of Dicts): A record of every trade made during the test.
  * `[{'step': 5, 'action': 'BUY', 'price': 150.0, 'balance': 9850}, ...]`

  * `price_history` (Series): The actual closing prices for the test period.

* **Output Format:**
  * `Performance_Report` (JSON/Dict):
    * `{'Total_Return': 0.15, 'Sharpe_Ratio': 1.8, 'Max_Drawdown': -0.05}`

  * `Visualizations` (Matplotlib/Plotly objects): Cumulative returns curve vs. Buy-and-Hold.

---

### Data Flow Visualization

### Summary Table for Quick Reference

| Module | Receives From | Sends To | Format |
| --- | --- | --- | --- |
| **A (Data)** | CSV/API | B (Env) | `pd.DataFrame` |
| **B (Env)** | D (Action) | D (State), C (Reward) | `np.array`, `float` |
| **C (Reward)** | B (Env) | B (Env)  D (Agent) | `float` |
| **D (Agent)** | B (State/Reward) | B (Action) | `int` |
| **E (Test)** | B & D (Logs) | User | Plots & Metrics |
