![image](https://github.com/user-attachments/assets/356f5b27-beb9-438f-bc86-8a007dd56d8a)


# Sundai Stocks

Sundai Stocks is an advanced stock analysis and prediction platform that combines technical analysis with AI-powered sentiment analysis to provide comprehensive market insights.

## Features

- NASDAQ stock data acquisition
- Stock screening and filtering
- Technical data analysis with temporal-spectral momentum indicators
- Historical model testing with accuracy tracking
- AI+ advanced predictions using Google's latest Gemini AI models
- News sentiment analysis
- Market research and intelligence reports

## Installation and Setup

### Prerequisites

- Docker and Docker Compose
- Internet connection for data downloads
- API keys (Gemini API, News API) - default keys are provided for testing

### Quick Start with Docker

1. Clone the repository:
```bash
git clone https://github.com/yourusername/sundai-stocks.git
cd sundai-stocks
```

2. Build the Docker image:
```bash
docker-compose build
```

3. Start the application:
```bash
docker-compose up -d
```

4. Access the application at http://localhost:8050
   - Default login: username `sundai`, password `Stocks123!`

### Manual Setup

If you prefer to run without Docker:

1. Install Python 3.11 or later
2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
python main.py
```

## Configuration

You can customize the application by setting environment variables:

- `DASH_USERNAME`: Dashboard login username (default: sundai)
- `DASH_PASSWORD_HASH`: Hashed password (default is for "Stocks123!")
- `SECRET_KEY`: Secret key for session encryption
- `GEMINI_API_KEY`: Your Google Gemini API key
- `NEWS_API_KEY`: Your News API key

## Usage Guide

1. **Data Download**: Start by downloading stock data from NASDAQ
2. **Stock Selection**: Use the screener to select stocks for analysis
3. **Technical Data**: Fetch technical indicators for selected stocks
4. **Test Model**: Validate the model's performance on historical data
5. **Predictions**: Generate market forecasts for selected securities
6. **Research**: Get sentiment analysis and news reports
7. **AI+ Analysis**: Use advanced AI to analyze stocks and predict movements

## Using AI+ Advanced Analysis

The AI+ tab leverages Google's Gemini AI to provide advanced stock analysis:

1. Select a stock symbol from the dropdown
2. Choose the technical data timeframe (1 month recommended for most cases)
3. Fetch the technical data by clicking "Fetch Technical Data"
4. Set the news analysis period (7 days recommended)
5. Fetch news data by clicking "Fetch News Data"
6. Select the prediction horizon (Next Day, Next Week, etc.)
7. Click "Begin AI Analysis" to generate an AI-enhanced prediction

The system uses local caching to store data, so you don't need to re-fetch data for repeated analyses of the same stock.

## Development

### Project Structure

- `modules/`: Core functionality modules
- `layouts/`: Dash layout components
- `callbacks/`: Dash callback logic
- `assets/`: CSS and static assets
- `stock_data/`: Data storage (created at runtime)

### Adding New Features

1. Update the relevant module in the `modules/` directory
2. Add UI components in the `layouts/` directory
3. Implement callbacks in the `callbacks/` directory
4. Update tests and documentation

## License

This project is licensed under the MIT License - see the LICENSE file for details.
