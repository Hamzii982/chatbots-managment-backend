# AI Control Panel - Backend

A powerful, enterprise-grade RAG-based Chatbot Management System built with FastAPI. This backend provides comprehensive APIs for managing multiple AI chatbots, handling document processing, managing vector embeddings, and orchestrating retrieval-augmented generation (RAG) pipelines.

## 🌟 Features

### Core Capabilities
- **Multi-Model Support**: Seamlessly integrate with OpenAI, Anthropic, and other LLM providers
- **RAG Pipeline**: Complete retrieval-augmented generation system with configurable components
- **Document Management**: Upload and process PDF documents with intelligent chunking
- **Vector Embeddings**: Leverage OpenAI embeddings with Chroma vector database for semantic search
- **Intelligent Retrieval**: Multiple retriever strategies (vector-based, hybrid, mock) for flexible retrieval patterns
- **Reranking**: Cross-encoder rerankers to enhance retrieval quality
- **Memory Management**: Support for both short-term and long-term conversation memory
- **User Authentication**: Secure JWT-based authentication and authorization
- **Performance Tracking**: Built-in statistics and usage analytics
- **Admin Panel**: Admin initialization and management capabilities

### Technical Highlights
- **Production-Ready**: Comprehensive error handling, validation, and logging
- **CORS Support**: Configurable cross-origin requests for seamless frontend integration
- **API Documentation**: Auto-generated interactive API docs (Swagger/OpenAPI)
- **Database Persistence**: PostgreSQL integration with SQLAlchemy ORM
- **Request Logging**: Detailed HTTP request/response logging for debugging
- **Modular Architecture**: Clean separation of concerns with organized code structure

## 🏗️ Architecture

### Directory Structure
```
app/
├── api/
│   └── routes/              # API endpoint definitions
│       ├── auth.py          # Authentication endpoints
│       ├── chatbots.py      # Chatbot management
│       ├── models.py        # LLM model configuration
│       ├── statistics.py    # Usage statistics
│       ├── system.py        # System health checks
│       └── usage.py         # User usage tracking
├── core/
│   ├── admin.py             # Admin initialization
│   ├── config.py            # Configuration management
│   └── security.py          # Security utilities
├── db/
│   ├── base.py              # Database base model
│   └── session.py           # Database session management
├── models/                  # SQLAlchemy models
│   ├── chatbot.py
│   ├── chatbot_documents.py
│   ├── message.py
│   ├── model_config.py
│   └── user.py
├── services/                # Business logic
│   ├── chatbot_service.py
│   └── langchain_rag.py     # RAG pipeline implementation
├── schemas/                 # Pydantic validation schemas
├── retrievers/              # Document retrieval implementations
│   ├── base.py
│   ├── factory.py
│   ├── hybrid_retriever.py
│   ├── mock_retriever.py
│   └── vector_retriever.py
├── rerankers/               # Result reranking implementations
│   ├── base.py
│   ├── cross_encoder_reranker.py
│   ├── factory.py
│   └── simple_reranker.py
└── main.py                  # FastAPI application setup

uploads/
└── chatbots/                # User-uploaded documents storage

.vectorstores/              # Chroma vector database storage
```

## 🛠️ Tech Stack

- **Framework**: FastAPI (Python 3.10+)
- **Database**: PostgreSQL with SQLAlchemy ORM
- **Vector DB**: Chroma with OpenAI embeddings
- **LLM Integration**: LangChain with OpenAI & Anthropic support
- **Authentication**: JWT with bcrypt password hashing
- **Validation**: Pydantic
- **API Style**: RESTful with Swagger/OpenAPI documentation

## 📋 Prerequisites

- Python 3.10 or higher
- PostgreSQL 12+
- Docker (optional, for running PostgreSQL)
- API keys for:
  - OpenAI (for LLMs and embeddings)
  - Anthropic (optional, for Claude models)

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone <repository-url>
cd assistant-backend
```

### 2. Create Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Set Up Database

**Option A: Docker (Recommended)**
```bash
docker run --name ai-postgres \
  -e POSTGRES_USER=user \
  -e POSTGRES_PASSWORD=password \
  -e POSTGRES_DB=ai_platform \
  -p 5432:5432 \
  -d postgres
```

**Option B: Local PostgreSQL**
- Ensure PostgreSQL is running on your system
- Create a database named `ai_platform`
- Update the `DATABASE_URL` in your `.env` file

### 5. Configure Environment Variables

Create a `.env` file in the project root:
```env
# Database
DATABASE_URL=postgresql://user:password@localhost:5432/ai_platform

# Security
SECRET_KEY=your-super-secret-key-change-me-in-production
ALGORITHM=HS256

# File Upload
MAX_UPLOAD_SIZE_MB=50
ALLOWED_FILE_EXTENSIONS=[".pdf"]

# LLM APIs
OPENAI_API_KEY=your-openai-api-key-here
ANTHROPIC_API_KEY=your-anthropic-api-key-here
```

### 6. Initialize Database
The database tables are automatically created on application startup via SQLAlchemy's metadata.

### 7. Run the Application
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The application will be available at:
- **API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/api/docs
- **ReDoc**: http://localhost:8000/api/redoc
- **Health Check**: http://localhost:8000/system/health

## 📚 API Endpoints Overview

### Authentication
- `POST /auth/login` - User login
- `POST /auth/register` - User registration
- `POST /auth/refresh` - Refresh access token

### Chatbots
- `GET /chatbots` - List all chatbots
- `POST /chatbots` - Create new chatbot
- `GET /chatbots/{id}` - Get chatbot details
- `PUT /chatbots/{id}` - Update chatbot configuration
- `DELETE /chatbots/{id}` - Delete chatbot
- `POST /chatbots/{id}/upload` - Upload documents
- `POST /chatbots/{id}/chat` - Send message to chatbot
- `WebSocket /chatbots/{id}/stream` - Stream real-time responses

### Models
- `GET /models` - List available LLM models
- `POST /models` - Configure new model
- `PUT /models/{id}` - Update model configuration

### Statistics & Usage
- `GET /statistics` - Get system statistics
- `GET /usage` - Get user usage metrics
- `GET /usage/chatbot/{id}` - Get chatbot-specific usage

### System
- `GET /system/health` - Health check endpoint
- `GET /system/info` - System information

## 🔐 Security Features

- **JWT Authentication**: Secure token-based authentication
- **Password Hashing**: Bcrypt password hashing for user credentials
- **CORS Protection**: Configurable cross-origin request handling
- **Input Validation**: Pydantic schemas for all request/response data
- **Error Handling**: Detailed validation error responses without exposing internals
- **Request Logging**: Comprehensive audit trail of API requests

## 🎛️ Configuration Options

### Model Configuration
Each chatbot supports granular RAG configuration:
- **Chunk Size**: Document chunk size for optimal retrieval (default: 500)
- **Chunk Overlap**: Overlap between chunks (default: 50)
- **Top K**: Number of documents to retrieve (default: 3)
- **Retriever Type**: Choose between `vector`, `hybrid`, or `mock`
- **Reranker Type**: `simple` or `cross_encoder` for result reranking
- **Memory**: Toggle short-term and long-term memory

### Supported Retrievers
- **Vector Retriever**: Semantic search using embeddings
- **Hybrid Retriever**: Combination of semantic and BM25 search
- **Mock Retriever**: Testing and development

### Supported Rerankers
- **Simple Reranker**: Basic relevance scoring
- **Cross-Encoder Reranker**: Fine-tuned semantic relevance ranking

## 🐳 Docker Support

Build and run with Docker:
```bash
docker build -t ai-assistant-backend .
docker run -p 8000:8000 \
  -e DATABASE_URL=postgresql://user:password@postgres:5432/ai_platform \
  -e OPENAI_API_KEY=your-key \
  ai-assistant-backend
```

## 📊 Database Schema

### Core Tables
- **users**: User accounts and authentication
- **chatbots**: Chatbot configurations
- **models**: LLM model configurations
- **messages**: Conversation message history
- **chatbot_documents**: Uploaded documents metadata
- **model_config**: Model-specific parameters

## 🔧 Development

### Code Structure
- **Services**: Business logic and RAG pipeline (`services/`)
- **Routes**: API endpoint handlers (`api/routes/`)
- **Models**: Database ORM models (`models/`)
- **Schemas**: Request/response validation (`schemas/`)

### Logging
The application uses Python's standard logging with INFO level by default. Output includes:
- Request method and path
- Response status codes
- Validation errors
- Errors and exceptions

## 🚨 Troubleshooting

### Common Issues

**Database Connection Error**
- Verify PostgreSQL is running
- Check DATABASE_URL in `.env`
- Ensure PostgreSQL user/password are correct

**API Key Errors**
- Verify OPENAI_API_KEY and ANTHROPIC_API_KEY are set in `.env`
- Check API key validity in provider dashboards

**Vector Store Issues**
- Clear `.vectorstores/` directory if corrupted
- Application will recreate vector stores on next chatbot interaction

**CORS Errors**
- Verify frontend URL is in ALLOWED_ORIGINS in `.env`
- Check that frontend is using correct API base URL

## 📈 Performance Considerations

- **Vector Embeddings**: Cached in Chroma DB for faster retrieval
- **Document Chunking**: Configurable for optimal token/retrieval balance
- **Memory Management**: Efficient garbage collection for vector store operations
- **Pooled Connections**: SQLAlchemy manages database connection pooling

## 🤝 Contributing

Contributions are welcome! Please:
1. Create a feature branch (`git checkout -b feature/amazing-feature`)
2. Commit changes (`git commit -m 'Add amazing feature'`)
3. Push to branch (`git push origin feature/amazing-feature`)
4. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📞 Support

For issues, feature requests, or questions:
- Open an issue on GitHub
- Check existing documentation in `/api/docs`
- Review error logs for debugging information

---

**Made with ❤️ for intelligent chatbot management**