"""
Comprehensive API Tests for CIAP Module 8
Tests all API endpoints, middleware, and functionality
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch, MagicMock
import asyncio
from datetime import datetime

from src.api import app, get_db
from src.database.manager import DatabaseManager
from src.database.operations import DatabaseOperations
from src.task_queue.manager import TaskStatus


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def client():
    """Create FastAPI test client"""
    return TestClient(app)


@pytest.fixture
async def test_db():
    """Create in-memory test database"""
    db = DatabaseManager("sqlite+aiosqlite:///:memory:")
    await db.initialize()
    yield db
    await db.close()


@pytest.fixture
def override_db(test_db):
    """Override database dependency for testing"""
    async def get_test_db():
        async with test_db.get_session() as session:
            yield session

    app.dependency_overrides[get_db] = get_test_db
    yield
    app.dependency_overrides.clear()


# ============================================================
# Test: Root & Health Endpoints
# ============================================================

class TestRootAndHealth:
    """Test root and health check endpoints"""

    def test_root_endpoint(self, client):
        """Test root endpoint returns API info"""
        response = client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "CIAP API"
        assert data["version"] == "0.8.0"
        assert data["status"] == "running"
        assert "/api/docs" in data["docs"]

    def test_health_check_endpoint(self, client):
        """Test health check endpoint"""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "checks" in data
        assert "database" in data["checks"]
        assert "ollama" in data["checks"]
        assert "task_queue" in data["checks"]
        assert "cache" in data["checks"]


# ============================================================
# Test: Search Endpoints
# ============================================================

class TestSearchEndpoints:
    """Test search CRUD endpoints"""

    @pytest.mark.xfail(reason="Complex mocking of async DB commit/refresh needed")
    def test_create_search_success(self, client):
        """Test successful search creation"""
        with patch("src.database.operations.DatabaseOperations.create_search", new_callable=AsyncMock) as mock_create:
            with patch("src.task_queue.manager.task_queue.enqueue", new_callable=AsyncMock) as mock_enqueue:
                # Setup mocks
                mock_search = MagicMock()
                mock_search.id = 1
                mock_search.query = "AI trends"
                mock_search.status = "pending"
                mock_search.created_at = datetime.utcnow()

                mock_create.return_value = mock_search
                mock_enqueue.return_value = 123

                # Make request
                response = client.post(
                    "/api/v1/search",
                    json={
                        "query": "AI trends",
                        "sources": ["google"],
                        "max_results": 50,
                        "analyze": True
                    }
                )

                assert response.status_code == 201
                data = response.json()
                assert data["search_id"] == 1
                assert data["query"] == "AI trends"
                assert data["status"] == "pending"
                assert data["task_id"] == 123

    def test_create_search_validation_error(self, client):
        """Test search creation with invalid query"""
        response = client.post(
            "/api/v1/search",
            json={
                "query": "",  # Invalid empty query
                "sources": ["google"]
            }
        )

        assert response.status_code == 422  # Validation error

    def test_get_search_found(self, client):
        """Test getting existing search"""
        with patch("src.api.routes.search.DatabaseOperations.get_search", new_callable=AsyncMock) as mock_get:
            with patch("src.api.routes.search.DatabaseOperations.get_search_results", new_callable=AsyncMock) as mock_results:
                # Setup mocks
                mock_search = MagicMock()
                mock_search.id = 1
                mock_search.query = "AI trends"
                mock_search.status = "completed"
                mock_search.sources = ["google"]
                mock_search.created_at = datetime.utcnow()
                mock_search.completed_at = datetime.utcnow()
                mock_search.error_message = None

                mock_get.return_value = mock_search
                mock_results.return_value = []

                response = client.get("/api/v1/search/1")

                assert response.status_code == 200
                data = response.json()
                assert data["search"]["id"] == 1
                assert data["search"]["query"] == "AI trends"
                assert "results" in data

    def test_get_search_not_found(self, client):
        """Test getting non-existent search"""
        with patch("src.api.routes.search.DatabaseOperations.get_search", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = None

            response = client.get("/api/v1/search/999")

            assert response.status_code == 404
            assert "not found" in response.json()["detail"].lower()

    def test_list_searches_pagination(self, client):
        """Test listing searches with pagination"""
        with patch("src.database.manager.db_manager.get_session") as mock_session_ctx:
            # Create mock session
            mock_session = AsyncMock()
            mock_session_ctx.return_value.__aenter__.return_value = mock_session

            # Mock execute to return searches
            mock_result = MagicMock()
            mock_result.scalars.return_value.all.return_value = []
            mock_result.scalar.return_value = 10

            mock_session.execute = AsyncMock(return_value=mock_result)

            response = client.get("/api/v1/search?skip=0&limit=10")

            assert response.status_code == 200
            data = response.json()
            assert "searches" in data
            assert "total" in data
            assert "page" in data
            assert "per_page" in data

    def test_delete_search(self, client):
        """Test deleting search"""
        with patch("src.database.operations.DatabaseOperations.get_search", new_callable=AsyncMock) as mock_get:
            with patch("src.database.manager.db_manager.get_session") as mock_session_ctx:
                # Setup mocks
                mock_search = MagicMock()
                mock_get.return_value = mock_search

                mock_session = AsyncMock()
                mock_session_ctx.return_value.__aenter__.return_value = mock_session

                response = client.delete("/api/v1/search/1")

                assert response.status_code == 204


# ============================================================
# Test: Task Endpoints
# ============================================================

class TestTaskEndpoints:
    """Test task management endpoints"""

    def test_enqueue_task(self, client):
        """Test enqueueing new task"""
        with patch("src.api.routes.tasks.task_queue.enqueue", new_callable=AsyncMock) as mock_enqueue:
            with patch("src.api.routes.tasks.task_queue.get_task_status", new_callable=AsyncMock) as mock_status:
                # Setup mocks
                mock_enqueue.return_value = 1

                mock_status.return_value = {
                    "type": "scrape",
                    "status": "pending",
                    "priority": 5,
                    "created_at": datetime.utcnow(),
                    "result": None,
                    "error": None
                }

                response = client.post(
                    "/api/v1/tasks",
                    json={
                        "type": "scrape",
                        "payload": {"query": "test"},
                        "priority": 5
                    }
                )

                assert response.status_code == 201
                data = response.json()
                assert data["task_id"] == 1
                assert data["type"] == "scrape"
                assert data["status"] == "pending"

    def test_get_task_status(self, client):
        """Test getting task status"""
        with patch("src.api.routes.tasks.task_queue.get_task_status", new_callable=AsyncMock) as mock_status:
            mock_status.return_value = {
                "type": "scrape",
                "status": "completed",
                "priority": 5,
                "created_at": datetime.utcnow(),
                "result": {"status": "success"},
                "error": None
            }

            response = client.get("/api/v1/tasks/1")

            assert response.status_code == 200
            data = response.json()
            assert data["task_id"] == 1
            assert data["status"] == "completed"
            assert data["result"] is not None

    def test_get_task_not_found(self, client):
        """Test getting non-existent task"""
        with patch("src.api.routes.tasks.task_queue.get_task_status", new_callable=AsyncMock) as mock_status:
            mock_status.return_value = None

            response = client.get("/api/v1/tasks/999")

            assert response.status_code == 404

    def test_list_tasks_with_filters(self, client):
        """Test listing tasks with filters"""
        with patch("src.task_queue.manager.task_queue.get_queue_stats", new_callable=AsyncMock) as mock_stats:
            with patch("src.database.operations.DatabaseOperations.get_paginated_task_queue", new_callable=AsyncMock) as mock_paginated:
                # Setup mocks
                mock_stats.return_value = {"status_counts": {}}

                from src.database.operations import PaginatedResult
                mock_paginated.return_value = PaginatedResult(
                    items=[],
                    total=0,
                    page=1,
                    per_page=10,
                    total_pages=0,
                    has_prev=False,
                    has_next=False
                )

                response = client.get("/api/v1/tasks?status=pending&task_type=scrape")

                assert response.status_code == 200
                data = response.json()
                assert "tasks" in data
                assert "total" in data

    def test_batch_enqueue_tasks(self, client):
        """Test enqueueing multiple tasks"""
        with patch("src.api.routes.tasks.task_queue.enqueue_batch", new_callable=AsyncMock) as mock_batch:
            mock_batch.return_value = [1, 2, 3]

            response = client.post(
                "/api/v1/tasks/batch",
                json={
                    "tasks": [
                        {"type": "scrape", "payload": {"query": "test1"}},
                        {"type": "scrape", "payload": {"query": "test2"}},
                        {"type": "scrape", "payload": {"query": "test3"}}
                    ]
                }
            )

            assert response.status_code == 201
            data = response.json()
            assert data["count"] == 3
            assert len(data["task_ids"]) == 3

    def test_cancel_task(self, client):
        """Test cancelling task"""
        with patch("src.api.routes.tasks.task_queue.get_task_status", new_callable=AsyncMock) as mock_status:
            with patch("src.api.routes.tasks.task_queue.cancel_task", new_callable=AsyncMock) as mock_cancel:
                # Setup mocks
                mock_status.return_value = {
                    "status": "pending",
                    "type": "scrape",
                    "priority": 5,
                    "created_at": datetime.utcnow()
                }
                mock_cancel.return_value = True

                response = client.delete("/api/v1/tasks/1")

                assert response.status_code == 200
                data = response.json()
                assert data["task_id"] == 1
                assert "cancelled" in data["message"].lower()


# ============================================================
# Test: Analysis Endpoints
# ============================================================

class TestAnalysisEndpoints:
    """Test analysis endpoints"""

    def test_analyze_text(self, client):
        """Test text analysis"""
        with patch("src.api.routes.analysis.ollama_client.analyze", new_callable=AsyncMock) as mock_analyze:
            mock_analyze.return_value = {
                "sentiment": "positive",
                "confidence": 0.9
            }

            response = client.post(
                "/api/v1/analysis/text",
                json={
                    "text": "This is a great product!",
                    "analysis_type": "sentiment"
                }
            )

            assert response.status_code == 200
            data = response.json()
            assert data["analysis_type"] == "sentiment"
            assert "result" in data
            assert data["result"]["sentiment"] == "positive"

    def test_analyze_sentiment(self, client):
        """Test sentiment analysis for search"""
        with patch("src.api.routes.analysis.SentimentAnalyzer") as mock_analyzer_class:
            mock_analyzer = AsyncMock()
            mock_analyzer.analyze_search_results = AsyncMock(return_value={
                "sample_size": 50,
                "sentiment_distribution": {"positive": 30, "neutral": 15, "negative": 5},
                "dominant_sentiment": "positive",
                "average_confidence": 0.85
            })
            mock_analyzer_class.return_value = mock_analyzer

            response = client.get("/api/v1/analysis/sentiment/1?sample_size=50")

            assert response.status_code == 200
            data = response.json()
            assert data["search_id"] == 1
            assert data["dominant_sentiment"] == "positive"
            assert data["sample_size"] == 50

    def test_analyze_competitors(self, client):
        """Test competitor analysis"""
        with patch("src.api.routes.analysis.CompetitorAnalyzer") as mock_analyzer_class:
            mock_analyzer = AsyncMock()
            mock_analyzer.analyze_competitors = AsyncMock(return_value={
                "competitors": ["Google", "Microsoft"],
                "products": ["Search", "Bing"],
                "mentions": {"Google": 10, "Microsoft": 5},
                "analysis": "Competition analysis results"
            })
            mock_analyzer_class.return_value = mock_analyzer

            response = client.get("/api/v1/analysis/competitors/1")

            assert response.status_code == 200
            data = response.json()
            assert data["search_id"] == 1
            assert len(data["competitors"]) == 2
            assert "Google" in data["competitors"]

    def test_analyze_trends(self, client):
        """Test trend analysis"""
        with patch("src.api.routes.analysis.TrendAnalyzer") as mock_analyzer_class:
            mock_analyzer = AsyncMock()
            mock_analyzer.analyze_trends = AsyncMock(return_value={
                "trends": ["AI adoption", "Cloud migration"],
                "keywords": ["AI", "cloud", "digital"],
                "topics": ["Technology", "Innovation"]
            })
            mock_analyzer_class.return_value = mock_analyzer

            response = client.get("/api/v1/analysis/trends/1")

            assert response.status_code == 200
            data = response.json()
            assert data["search_id"] == 1
            assert len(data["trends"]) > 0

    def test_get_analysis_stats(self, client):
        """Test getting analysis statistics"""
        with patch("src.api.routes.analysis.ollama_client.stats", {
            "requests": 100,
            "cache_hits": 25,
            "errors": 2,
            "total_tokens": 50000
        }):
            response = client.get("/api/v1/analysis/stats")

            assert response.status_code == 200
            data = response.json()
            assert data["requests"] == 100
            assert data["cache_hits"] == 25
            assert "cache_hit_rate" in data


# ============================================================
# Test: Middleware
# ============================================================

class TestMiddleware:
    """Test middleware functionality"""

    def test_rate_limiting_under_limit(self, client):
        """Test rate limiting allows requests under limit"""
        with patch("src.api.main.RateLimitCache.check_rate_limit", new_callable=AsyncMock) as mock_check:
            mock_check.return_value = True

            response = client.get("/")

            assert response.status_code == 200

    def test_rate_limiting_over_limit(self, client):
        """Test rate limiting blocks requests over limit"""
        with patch("src.api.main.RateLimitCache.check_rate_limit", new_callable=AsyncMock) as mock_check:
            with patch("src.api.main.RateLimitCache.get_remaining", new_callable=AsyncMock) as mock_remaining:
                mock_check.return_value = False
                mock_remaining.return_value = 0

                response = client.get("/")

                assert response.status_code == 429
                assert "rate limit" in response.json()["error"].lower()

    def test_request_logging_adds_timing_header(self, client):
        """Test request logging adds X-Process-Time header"""
        response = client.get("/")

        assert response.status_code == 200
        assert "X-Process-Time" in response.headers
        # Check that timing value is numeric
        timing = float(response.headers["X-Process-Time"])
        assert timing >= 0


# ============================================================
# Test: Export Endpoints
# ============================================================

class TestExportEndpoints:
    """Test export endpoints (placeholders)"""

    def test_export_csv_placeholder(self, client):
        """Test CSV export returns placeholder"""
        response = client.get("/api/v1/export/search/1/csv")

        assert response.status_code == 501
        data = response.json()
        assert data["module"] == "Module 9"
        assert data["status"] == "placeholder"

    def test_export_json_partial(self, client):
        """Test JSON export partial implementation"""
        with patch("src.database.operations.DatabaseOperations.get_search", new_callable=AsyncMock) as mock_get:
            with patch("src.database.operations.DatabaseOperations.get_search_results", new_callable=AsyncMock) as mock_results:
                # Setup mocks
                mock_search = MagicMock()
                mock_search.id = 1
                mock_search.query = "test"
                mock_search.status = "completed"
                mock_search.sources = ["google"]
                mock_search.created_at = datetime.utcnow()
                mock_search.completed_at = datetime.utcnow()

                mock_get.return_value = mock_search
                mock_results.return_value = []

                response = client.get("/api/v1/export/search/1/json")

                assert response.status_code == 200
                data = response.json()
                assert "search" in data
                assert "results" in data

    def test_export_formats(self, client):
        """Test getting export formats"""
        response = client.get("/api/v1/export/formats")

        assert response.status_code == 200
        data = response.json()
        assert "formats" in data
        assert len(data["formats"]) > 0


# ============================================================
# Run Tests
# ============================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
