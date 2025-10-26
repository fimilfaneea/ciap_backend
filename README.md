# CIAP - Competitive Intelligence Automation Platform

An open-source competitive intelligence solution designed for Small and Medium-sized Enterprises (SMEs), providing enterprise-level intelligence capabilities at 70-90% lower cost.

## 🎯 Overview

CIAP automates the collection and analysis of competitive intelligence data using open-source web scrapers and Large Language Models (LLMs). It provides real-time insights on competitors, market trends, and customer sentiment through an intuitive dashboard interface.

## ✨ Features

- **Multi-Source Data Collection**: Automated scraping from Google and other search engines
- **AI-Powered Analysis**: LLM integration for sentiment analysis, competitor profiling, and trend identification
- **Real-Time Insights**: Automated processing pipeline for timely intelligence updates
- **Cost-Effective**: 70-90% cost reduction compared to commercial solutions
- **Easy to Use**: Simple web interface with no technical expertise required
- **Scalable Architecture**: Modular design for easy expansion and customization

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- Git

### Installation

1. Clone the repository:
```bash
git clone https://github.com/fimilfaneea/ciap_backend.git
cd ciap_backend
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env file with your API keys
```

5. Initialize the database:
```bash
python database.py
```

6. Run the application:
```bash
python main.py
```

7. Open your browser and navigate to:
```
http://localhost:8000
```

## 🔧 Configuration

Edit the `.env` file to configure:

- **LLM API Keys**: Set your OpenAI or Anthropic API keys
- **Scraping Settings**: Adjust delay and result limits
- **API Settings**: Configure host and port

## 📚 API Documentation

Once running, access the interactive API documentation at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## 🏗️ Architecture

```
ciap/
├── main.py              # FastAPI application
├── database.py          # SQLAlchemy models
├── config.py            # Configuration settings
├── scrapers/            # Web scraping modules
│   └── google_scraper.py
├── analysis/            # LLM analysis modules
│   └── llm_analyzer.py
├── api/                 # API service layer
│   └── search_service.py
└── static/              # Web interface
    └── index.html
```

## 🛠️ Technology Stack

- **Backend**: Python, FastAPI
- **Database**: SQLite (PostgreSQL ready)
- **Scraping**: BeautifulSoup, Requests
- **AI/ML**: OpenAI API, Anthropic Claude
- **Frontend**: HTML5, JavaScript, CSS3

## 📈 Use Cases

- **Competitor Analysis**: Track competitor products, pricing, and strategies
- **Market Intelligence**: Monitor industry trends and emerging technologies
- **Customer Insights**: Analyze customer sentiment and feedback
- **Strategic Planning**: Data-driven decision making for business strategy

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is open source and available under the MIT License.

## 👤 Author

**Fimil Faneea**
M.Tech IT Student (24102371)

## 🙏 Acknowledgments

- Built as part of academic research on democratizing competitive intelligence
- Designed to help SMEs compete with larger enterprises
- Leverages open-source tools and modern AI capabilities

## ⚠️ Disclaimer

This tool is for legitimate business intelligence purposes only. Users must comply with all applicable laws and website terms of service when using web scraping features.

## 📞 Support

For issues, questions, or suggestions, please open an issue on GitHub.