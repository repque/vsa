# VSA - Volume Spread Analysis Stock Prediction System

An automated stock market prediction and trading system that analyzes 240+ stocks using Volume Spread Analysis (VSA) techniques combined with machine learning to identify potential price movements.

## Overview

VSA is a Python-based trading system that:
- Predicts stocks likely to rise in the next 4 weeks based on historical price/volume patterns
- Analyzes 240+ stocks across various market sectors
- Generates automated daily email reports with predictions and sector distribution
- Tracks trading positions and calculates P&L (profit and loss)
- Uses ensemble machine learning models for robust predictions

## Features

### Machine Learning Pipeline
- **Ensemble Model**: Combines 4 classifiers using soft voting
  - Logistic Regression
  - Random Forest Classifier
  - Bernoulli Naive Bayes
  - K-Nearest Neighbors
- **Cross-Validation**: TimeSeriesSplit with 3 folds for robust evaluation
- **Feature Engineering**: VSA-based technical indicators (ratios, patterns, lagged features)

### Volume Spread Analysis Indicators
- **Range Ratio**: Bar size relative to average (wide bar detection)
- **Close Ratio**: Close position within bar (up/down close strength)
- **Volume Ratio**: Volume relative to average (high/low volume detection)
- **Bar Patterns**: Up bars, down bars, inside bars, outside bars
- **Lagged Features**: Previous 4 periods for temporal context

### Automated Reporting
- Daily email reports via Gmail API
- Sector-based analysis with word cloud visualization
- Filters high-probability predictions (>98%)
- HTML-formatted tables with stock tickers and probabilities

### Position Management
- Track open and closed positions
- Calculate mark-to-market (MTM) and realized P&L
- Monitor position expiration (31-day tracking)
- Persistent storage using TinyDB

## Project Structure

```
vsa/
├── __init__.py          # Package initialization
├── main.py              # CLI entry point with argument parsing
├── model.py             # ML training, prediction, and visualization
├── data.py              # Data fetching from Yahoo Finance with caching
├── features.py          # VSA feature engineering
├── positions.py         # Position tracking and P&L management
├── mail.py              # Gmail API integration and report generation
├── config.py            # Configuration management
├── logger.py            # Logging setup
├── exceptions.py        # Custom exception classes
├── .env.example         # Example environment configuration
├── requirements.txt     # Python dependencies
├── pytest.ini           # Pytest configuration
├── tests/               # Test suite
│   ├── test_features.py
│   ├── test_data.py
│   └── test_positions.py
└── README.md            # This file
```

## Installation

### Prerequisites
- Python 3.x
- Gmail API credentials (client_secret.json)

### Dependencies

Install all required packages using requirements.txt:

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install pandas numpy pandas-datareader scikit-learn plotly matplotlib wordcloud tinydb google-auth-oauthlib google-api-python-client joblib python-dotenv pytest pytest-cov
```

### Gmail API Setup

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select existing one
3. Enable Gmail API
4. Create OAuth 2.0 credentials (Desktop application)
5. Download `client_secret.json` and place in configured path

## Configuration

VSA uses environment-based configuration for easy setup across different platforms.

### Setup Configuration

1. Copy the example environment file:
   ```bash
   cp .env.example .env
   ```

2. Edit `.env` with your settings:
   ```bash
   # Model Configuration
   MODEL_PATH=./models/vsa.sav

   # Data Configuration
   DATA_CACHE_DIR=./data/daily_data
   STOCKS_CSV_PATH=./data/stocks.csv

   # Gmail API Configuration
   GMAIL_CLIENT_SECRET_PATH=./credentials/client_secret.json
   GMAIL_TOKEN_PATH=./credentials/token.pickle
   EMAIL_RECIPIENT=your-email@example.com

   # Database Configuration
   POSITIONS_DB_PATH=./db.json

   # Application Settings
   LOG_LEVEL=INFO
   ```

3. The application will automatically create required directories

### Configuration Options

All settings are defined in `config.py` and can be overridden via environment variables:

- **MODEL_PATH**: Path to trained model file
- **DATA_CACHE_DIR**: Directory for cached stock data
- **STOCKS_CSV_PATH**: Path to CSV file with stock sectors
- **GMAIL_CLIENT_SECRET_PATH**: Path to Gmail API credentials
- **GMAIL_TOKEN_PATH**: Path to store Gmail OAuth token
- **EMAIL_RECIPIENT**: Email address for reports
- **POSITIONS_DB_PATH**: Path to TinyDB positions database
- **LOG_LEVEL**: Logging level (DEBUG, INFO, WARNING, ERROR)

## Usage

### Command-Line Interface

VSA provides a comprehensive CLI with multiple options:

#### Basic Usage

Run predictions and send email report:
```bash
python main.py
```

#### Available Options

```bash
# Show help and all available options
python main.py --help

# Train a new model
python main.py --train

# Run predictions without sending email
python main.py --no-email

# Use custom stock list
python main.py --stocks AAPL MSFT TSLA NVDA

# Set custom log level
python main.py --log-level DEBUG

# Use custom model path
python main.py --model-path /path/to/custom/model.sav

# Send email to custom recipient
python main.py --email-to custom@email.com

# Combine multiple options
python main.py --no-email --log-level DEBUG --stocks AAPL MSFT
```

### Programmatic Usage

#### Training a New Model

```python
from model import train, save

# Train model on stock list
model = train(['AAPL', 'MSFT', 'TSLA'])

# Save trained model
save(model, 'models/vsa.sav')
```

### Managing Positions

```python
from positions import Position

# Record a new position entry
Position.record('AAPL', 100, 150.50, 'entry')

# Get all open positions
open_positions = Position.all_open()

# View P&L as DataFrame
pnl_df = Position.as_df()

# Check expiring positions
expiring = Position.expiring()
```

### Fetching Historical Data

```python
from data import get_historical_data, compress_data

# Fetch daily data for stocks
daily_data = get_historical_data(['AAPL', 'MSFT'])

# Compress to weekly bars
weekly_data = compress_data(daily_data, 'W')
```

### Generating Predictions

```python
from model import make_predictions

# Get predictions for all configured stocks
predictions = make_predictions()

# Sort by probability
predictions.sort(reverse=True)

# Filter high-confidence predictions
high_prob = [p for p in predictions if p[0] > 0.98]
```

## Data Flow

1. **Data Acquisition**: Yahoo Finance API → Local CSV cache
2. **Data Processing**: Daily OHLCV → Weekly compressed bars
3. **Feature Engineering**: VSA ratios + lagged features
4. **Model Prediction**: Ensemble classifier → Probability scores
5. **Report Generation**: Filter + sector grouping → Word cloud + HTML table
6. **Email Delivery**: Gmail API → Automated report

## Stock Universe

The system analyzes 240+ stocks across multiple sectors including:
- Technology (AAPL, MSFT, NVDA, AMD, etc.)
- Finance (JPM, BAC, C, GS, etc.)
- Energy (XOM, CVX, COP, etc.)
- Healthcare (JNJ, PFE, ABBV, etc.)
- Consumer (WMT, TGT, AMZN, etc.)
- Industrials (BA, CAT, GE, etc.)

See `model.py:3-14` for complete stock list.

## Output

### Email Report Format
- **Subject**: Daily stock predictions
- **Content**:
  - Word cloud visualization of recommended sectors
  - HTML table with stock ticker and probability
  - Filtered to show only predictions >98% confidence

### Example Prediction Output
```
Probability | Ticker
------------|-------
0.992       | AAPL
0.987       | MSFT
0.981       | NVDA
```

## Technical Details

### Prediction Target
- **Label**: 4 consecutive weeks of price increases
- **Binary Classification**: 1 = bullish, 0 = bearish
- **Lookback Window**: Last 14 weeks of data per stock

### Model Performance
- Uses TimeSeriesSplit to prevent data leakage
- Soft voting aggregates probabilities from all classifiers
- Model saved/loaded using joblib serialization

### Caching Strategy
- Historical data cached locally as CSV files
- Automatic incremental updates for existing files
- Atomic file writes to prevent corruption

## Database Schema

### positions (TinyDB - db.json)
- `ticker`: Stock symbol
- `quantity`: Number of shares
- `price`: Entry/exit price
- `date`: Transaction date
- `kind`: 'entry' or 'exit'

## Testing

The project includes a comprehensive test suite using pytest.

### Running Tests

Run all tests:
```bash
pytest
```

Run with coverage report:
```bash
pytest --cov=. --cov-report=html
```

Run specific test file:
```bash
pytest tests/test_features.py
```

Run with verbose output:
```bash
pytest -v
```

### Test Structure

- `tests/test_features.py`: Tests for VSA feature engineering
- `tests/test_data.py`: Tests for data fetching and processing
- `tests/test_positions.py`: Tests for position tracking

## Logging

VSA includes comprehensive logging throughout the application:

- **Console Output**: Formatted log messages to stdout
- **File Output**: Detailed logs saved to `logs/vsa.log`
- **Log Levels**: Configurable via CLI or environment variable
- **Structured Logging**: Breadcrumb-style state change tracking

Example log output:
```
[INFO] main - [SYSTEM] Starting VSA prediction pipeline
[INFO] main - [SYSTEM] Generating predictions for 240 stocks
[INFO] data - Downloading AAPL: 2020-01-01 - 2025-01-09
[INFO] main - [SYSTEM] Found 15 positive predictions
[INFO] main -   AAPL: 99.20%
[INFO] main -   MSFT: 98.50%
```

## Recent Improvements

The following enhancements have been made to address previous limitations:

- ✅ **Configuration Management**: Environment-based configuration with `.env` file
- ✅ **Cross-Platform Support**: Platform-independent path handling using pathlib
- ✅ **Type Hints**: Full type annotations across all modules
- ✅ **Error Handling**: Custom exceptions and comprehensive try-catch blocks
- ✅ **Logging System**: Structured logging with file and console output
- ✅ **Test Suite**: pytest-based tests with 30+ test cases
- ✅ **CLI Arguments**: Full command-line interface with argparse
- ✅ **Dependencies**: requirements.txt for easy installation

## Future Enhancements

Potential improvements for future development:

- Build backtesting framework with historical performance metrics
- Add performance tracking and model evaluation dashboard
- Implement data validation and quality checks
- Create Docker container for easy deployment
- Add scheduled execution with cron/systemd
- Build web interface for predictions visualization
- Add support for multiple ML models and comparison
- Implement real-time prediction updates

## License

Not specified

## Author

Not specified

## Contributing

Not specified
