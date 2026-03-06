# Australia Unicom - Telecommunications Project Management Platform

A comprehensive telecommunications project management platform for Australia Unicom, combining ML-based pricing prediction, full project lifecycle management, AI-powered analytics, and real-time monitoring.

## Features

### AI Price Prediction Engine
- CatBoost machine learning model for telecom line pricing
- Supports multiple carriers: Telstra, Optus, Vocus, TPG, Superloop, Aussie Broadband
- Product types: Internet, WAN, VPN
- Automatic postcode-to-zone mapping for Australian regions
- Confidence scoring and price floor protection (minimum AUD $50)

### Project Lifecycle Management
- 7-stage workflow tracking:
  1. Requirement Confirmation
  2. Solution Design
  3. Supplier Inquiry
  4. Internal Approval
  5. Contract Signing
  6. Engineering Implementation
  7. Project Acceptance
- Gantt chart visualization for project timelines
- File upload/download for project attachments
- Manual review and approval workflows

### AI Smart Assistant
- Multi-model support:
  - Ollama (free, local deployment - recommended)
  - Google Gemini (generous free tier)
  - Alibaba Qwen / Tongyi Qianwen
  - DeepSeek
  - LM Studio (local)
- RAG (Retrieval-Augmented Generation) with project database context
- Web search integration via DuckDuckGo

### Monitoring & Analytics Dashboard
- Real-time project status tracking
- Analytics charts and data visualization
- Browser notification support
- Smart polling for live updates

### Content & Knowledge Base
- Network coverage articles
- Innovation leadership content
- SME (Small & Medium Enterprise) solutions
- Company knowledge base integration

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | Flask (Python) |
| ML Model | CatBoost Regressor |
| Frontend | HTML / CSS / JavaScript |
| Maps | Leaflet.js |
| Charts | Chart.js |
| AI Integration | Multi-model LLM API (Ollama, Gemini, Qwen, DeepSeek) |
| Web Search | DuckDuckGo Search API |
| Rate Limiting | Flask-Limiter |

## Project Structure

```
├── server.py                  # Flask backend (routes, API, AI integration)
├── trainadvance.py            # CatBoost model training script
├── telecom_model.cbm          # Trained ML model (binary)
├── training_mvp_v5.csv        # Training dataset
├── company_knowledge.txt      # Company knowledge base for AI assistant
├── requirements.txt           # Python dependencies
├── manual_reviews.csv         # Manual review records
├── notifications.json         # Notification data store
├── projects.json              # Project data store
├── user_feedback.csv          # User feedback records
├── static/
│   ├── css/
│   │   └── global.css         # Global design system & responsive styles
│   ├── js/
│   │   └── global.js          # Toast, modal, polling & notification utilities
│   ├── logo.png               # Brand logo
│   ├── app.ico                # Favicon
│   ├── opera-house-4k.jpg     # Hero image
│   └── sydney-tower-4k.jpg    # Hero image
└── templates/
    ├── homepage.html           # Landing page
    ├── index.html              # Price estimation interface with map
    ├── project_monitor.html    # Project monitoring dashboard
    ├── project_detail.html     # Individual project detail & Gantt chart
    ├── project_analytics.html  # Analytics dashboard
    ├── project_print.html      # Print/export template
    ├── article_coverage.html   # Network coverage article
    ├── article_innovation.html # Innovation leadership article
    └── article_sme.html        # SME solutions article
```

## Getting Started

### Prerequisites

- Python 3.9+
- pip

### Installation

1. **Clone the repository**

```bash
git clone <repository-url>
cd Telecommunication-sysytem-about-Australia-unicom
```

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

3. **Train the model** (skip if `telecom_model.cbm` already exists)

```bash
python trainadvance.py
```

4. **Start the server**

```bash
python server.py
```

The server will start at `http://localhost:5000`.

### AI Assistant Configuration (Optional)

The AI assistant supports multiple backends. Configure via environment variables or in-app settings:

| Provider | Setup | Cost |
|----------|-------|------|
| Ollama | Install [Ollama](https://ollama.ai), run locally | Free |
| Google Gemini | API key required | Free tier available |
| Alibaba Qwen | API key required | Free tier available |
| DeepSeek | API key required | Pay-per-use |
| LM Studio | Install [LM Studio](https://lmstudio.ai), run locally | Free |

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Homepage |
| GET | `/pricing` | Price estimation page |
| POST | `/predict` | Run price prediction |
| GET | `/health` | Health check |
| GET | `/projects` | List all projects |
| POST | `/projects` | Create a new project |
| GET | `/projects/<id>` | Project detail page |
| GET | `/monitor` | Project monitoring dashboard |
| GET | `/analytics` | Analytics dashboard |
| POST | `/chat` | AI assistant chat |
| GET | `/notifications` | Get notifications |
| POST | `/feedback` | Submit user feedback |

## Usage

1. **Price Estimation** — Navigate to the pricing page, fill in operator, bandwidth (Mbps), postcode, product type, and contract term, then click "Start Estimation" to get an AI-predicted price.

2. **Project Management** — Create and track telecommunications projects through a 7-stage workflow. Each stage supports notes, file attachments, and manual reviews.

3. **AI Assistant** — Chat with the AI assistant for data analysis, project insights, and general telecom questions. The assistant has access to the project database and company knowledge base.

4. **Monitoring & Analytics** — Use the dashboard to monitor project progress in real-time and view analytics across all projects.

## Notes

- Ensure `telecom_model.cbm` exists before starting the server; otherwise the prediction engine will not function.
- The server binds to `0.0.0.0:5000` by default, allowing LAN access.
- Rate limiting is set to 200 requests per hour per IP address.
- Data files (`projects.json`, `notifications.json`, `manual_reviews.csv`, `user_feedback.csv`) are stored locally as flat files.

## License

This project is proprietary to Australia Unicom.
