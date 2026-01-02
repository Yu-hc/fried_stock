# RL-Trader-Taiwan: Reinforcement Learning for TWSE

This repository contains a Reinforcement Learning (RL) framework designed to automate trading strategies specifically for the **Taiwan Stock Market**. The project utilizes deep Q-learning (or PPO/A2C) to navigate price action and optimize portfolio returns using historical TWSE data.


## ⚙️ Installation & Setup

Follow these steps to set up your local development environment.

### 1. Clone the Repository

```bash
git clone https://github.com/Yu-hc/fried_stock

```

### 2. Set Up a Virtual Environment

It is highly recommended to use a virtual environment to manage dependencies and avoid conflicts.

**On macOS/Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate

```

**On Windows:**

```bash
python -m venv .venv
.venv\Scripts\activate

```

### 3. Install Dependencies

Once the virtual environment is active, install the required packages:

```bash
pip install --upgrade pip
pip install -r requirements.txt

```

> **Note:** If you are using **TA-Lib**, ensure you have the C++ binary installed on your system before running the pip install.

### 4. Run the Main Script

Use run_test.ipynb under notebooks folder or under python entry run

```bash
python3 run_test.py

```

---

## 📈 Project Structure

```text
├── data/               # Historical CSV files / Database scripts
├── .venv/              # Custom Trading Environment logic
├── models/             # Trained RL agents and weights
├── notebooks/          # Exploratory Data Analysis (EDA)
├── python_entry/       # python entry point for execution
├── modules/            # Indicators and preprocessing helpers
├── report/             # Store training results
└── requirements.txt    # Project dependencies
```
