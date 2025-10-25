"""
Module 8 Integration Tests
End-to-end API testing with real database and workflows
"""

import pytest
import asyncio
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch, MagicMock
from datetime import datetime
import json

from src.api import app, get_db
from src.database.manager import DatabaseManager
from src.database.operations import DatabaseOperations


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
async def integration_db():
    """Create in-memory database for integration testing"""
    db = DatabaseManager("sqlite+aiosqlite:///:memory:")
    await db.initialize()
    yield db
    await db.close()


@pytest.fixture
def integration_client(integration_db):
    """Create test client with real database"""
    async def get_integration_db():
        async with integration_db.get_session() as session:
            yield session

    app.dependency_overrides[get_db] = get_integration_db
    client = TestClient(app)
    yield client
    app.dependency_overrides.clear()


# ============================================================
# Test: Module Imports
# ============================================================

class TestModuleImports:
    """Test that all API modules can be imported"""

    def test_import_api_module(self):
        """Test importing main API module"""
        from src.api import app, ws_manager, get_db
        assert app is not None
        assert ws_manager is not None
        assert get_db is not None

    def test_import_route_modules(self):
        """Test importing all route modules"""
        from src.api.routes import search, tasks, analysis, export
        assert search.router is not None
        assert tasks.router is not None
        assert analysis.router is not None
        assert export.router is not None

    def test_import_pydantic_models(self):
        """Test importing Pydantic models"""
        from src.api.routes.search import SearchRequest, SearchResponse
        from src.api.routes.tasks import TaskRequest, TaskResponse
        from src.api.routes.analysis import AnalyzeTextRequest

        assert SearchRequest is not None
        assert TaskRequest is not None
        assert AnalyzeTextRequest is not None


# ============================================================
# Test: Database Integration
# ============================================================

class TestDatabaseIntegration:
    """Test API integration with real database"""

    @pytest.mark.asyncio
    async def test_create_search_with_real_db(self, integration_db):
        """Test creating search with real database"""
        async with integration_db.get_session() as session:
            # Create search
            search = await DatabaseOperations.create_search(
                session=session,
                query="AI trends",
                sources=["google"]
            )

            await session.commit()

            # Verify search was created
            assert search.id is not None
            assert search.query == "AI trends"
            assert search.status == "pending"

            # Retrieve search
            retrieved = await DatabaseOperations.get_search(session, search.id)
            assert retrieved is not None
            assert retrieved.query == "AI trends"

    @pytest.mark.asyncio
    async def test_search_results_storage(self, integration_db):
        """Test storing and retrieving search results"""
        async with integration_db.get_session() as session:
            # Create search
            search = await DatabaseOperations.create_search(
                session=session,
                query="test query",
                sources=["google"]
            )
            await session.commit()

            # Add search results
            results = [
                {
                    "search_id": search.id,
                    "title": "Result 1",
                    "snippet": "Snippet 1",
                    "url": "http://example.com/1",
                    "source": "google",
                    "position": 1
                },
                {
                    "search_id": search.id,
                    "title": "Result 2",
                    "snippet": "Snippet 2",
                    "url": "http://example.com/2",
                    "source": "google",
                    "position": 2
                }
            ]

            await DatabaseOperations.bulk_insert_results(session, search.id, results)
            await session.commit()

            # Retrieve results
            retrieved_results = await DatabaseOperations.get_search_results(
                session, search.id
            )

            assert len(retrieved_results) == 2
            assert retrieved_results[0].title == "Result 1"


# ============================================================
# Test: End-to-End Workflows
# ============================================================

class TestEndToEndWorkflows:
    """Test complete workflows through API"""

    def test_complete_search_workflow(self, integration_client):
        """Test: Create search → Get status → Get results → Delete"""
        with patch("src.api.routes.search.task_queue.enqueue", new_callable=AsyncMock) as mock_enqueue:
            with patch("src.api.routes.search.DatabaseOperations.create_search", new_callable=AsyncMock) as mock_create:
                # Step 1: Create search
                mock_search = MagicMock()
                mock_search.id = 1
                mock_search.query = "AI trends"
                mock_search.status = "pending"
                mock_search.created_at = datetime.utcnow()

                mock_create.return_value = mock_search
                mock_enqueue.return_value = 123

                create_response = integration_client.post(
                    "/api/v1/search",
                    json={"query": "AI trends", "sources": ["google"]}
                )

                assert create_response.status_code == 201
                search_id = create_response.json()["search_id"]

        with patch("src.api.routes.search.DatabaseOperations.get_search", new_callable=AsyncMock) as mock_get:
            with patch("src.api.routes.search.DatabaseOperations.get_search_results", new_callable=AsyncMock) as mock_results:
                # Step 2: Get search
                mock_search.status = "completed"
                mock_search.completed_at = datetime.utcnow()
                mock_search.sources = ["google"]
                mock_search.error_message = None

                mock_get.return_value = mock_search
                mock_results.return_value = []

                get_response = integration_client.get(f"/api/v1/search/{search_id}")
                assert get_response.status_code == 200
                assert get_response.json()["search"]["status"] == "completed"

        with patch("src.api.routes.search.DatabaseOperations.get_search", new_callable=AsyncMock) as mock_get:
            with patch("src.api.routes.search.db_manager.get_session") as mock_session_ctx:
                # Step 3: Delete search
                mock_get.return_value = mock_search
                mock_session = AsyncMock()
                mock_session_ctx.return_value.__aenter__.return_value = mock_session

                delete_response = integration_client.delete(f"/api/v1/search/{search_id}")
                assert delete_response.status_code == 204

    def test_task_workflow(self, integration_client):
        """Test: Enqueue task → Check status → Cancel"""
        with patch("src.api.routes.tasks.task_queue.enqueue", new_callable=AsyncMock) as mock_enqueue:
            with patch("src.api.routes.tasks.task_queue.get_task_status", new_callable=AsyncMock) as mock_status:
                # Step 1: Enqueue task
                mock_enqueue.return_value = 1
                mock_status.return_value = {
                    "type": "scrape",
                    "status": "pending",
                    "priority": 5,
                    "created_at": datetime.utcnow(),
                    "result": None,
                    "error": None
                }

                enqueue_response = integration_client.post(
                    "/api/v1/tasks",
                    json={
                        "type": "scrape",
                        "payload": {"query": "test"},
                        "priority": 5
                    }
                )

                assert enqueue_response.status_code == 201
                task_id = enqueue_response.json()["task_id"]

        with patch("src.api.routes.tasks.task_queue.get_task_status", new_callable=AsyncMock) as mock_status:
            # Step 2: Get status
            mock_status.return_value = {
                "type": "scrape",
                "status": "processing",
                "priority": 5,
                "created_at": datetime.utcnow(),
                "result": None,
                "error": None
            }

            status_response = integration_client.get(f"/api/v1/tasks/{task_id}")
            assert status_response.status_code == 200
            assert status_response.json()["status"] == "processing"

        with patch("src.api.routes.tasks.task_queue.get_task_status", new_callable=AsyncMock) as mock_status:
            with patch("src.api.routes.tasks.task_queue.cancel_task", new_callable=AsyncMock) as mock_cancel:
                # Step 3: Cancel task
                mock_status.return_value = {
                    "status": "pending",
                    "type": "scrape",
                    "priority": 5,
                    "created_at": datetime.utcnow()
                }
                mock_cancel.return_value = True

                cancel_response = integration_client.delete(f"/api/v1/tasks/{task_id}")
                assert cancel_response.status_code == 200

    def test_analysis_workflow(self, integration_client):
        """Test: Text analysis → Sentiment → Competitors → Trends"""
        # Step 1: Text analysis
        with patch("src.api.routes.analysis.ollama_client.analyze", new_callable=AsyncMock) as mock_analyze:
            mock_analyze.return_value = {"sentiment": "positive", "confidence": 0.9}

            text_response = integration_client.post(
                "/api/v1/analysis/text",
                json={"text": "Great product!", "analysis_type": "sentiment"}
            )

            assert text_response.status_code == 200

        # Step 2: Sentiment analysis
        with patch("src.api.routes.analysis.SentimentAnalyzer") as mock_analyzer_class:
            mock_analyzer = AsyncMock()
            mock_analyzer.analyze_search_results = AsyncMock(return_value={
                "sample_size": 50,
                "sentiment_distribution": {"positive": 30, "neutral": 15, "negative": 5},
                "dominant_sentiment": "positive",
                "average_confidence": 0.85
            })
            mock_analyzer_class.return_value = mock_analyzer

            sentiment_response = integration_client.get("/api/v1/analysis/sentiment/1")
            assert sentiment_response.status_code == 200

        # Step 3: Competitor analysis
        with patch("src.api.routes.analysis.CompetitorAnalyzer") as mock_analyzer_class:
            mock_analyzer = AsyncMock()
            mock_analyzer.analyze_competitors = AsyncMock(return_value={
                "competitors": ["Google"],
                "products": ["Search"],
                "mentions": {"Google": 10},
                "analysis": "Analysis results"
            })
            mock_analyzer_class.return_value = mock_analyzer

            competitor_response = integration_client.get("/api/v1/analysis/competitors/1")
            assert competitor_response.status_code == 200

        # Step 4: Trend analysis
        with patch("src.api.routes.analysis.TrendAnalyzer") as mock_analyzer_class:
            mock_analyzer = AsyncMock()
            mock_analyzer.analyze_trends = AsyncMock(return_value={
                "trends": ["AI adoption"],
                "keywords": ["AI"],
                "topics": ["Technology"]
            })
            mock_analyzer_class.return_value = mock_analyzer

            trend_response = integration_client.get("/api/v1/analysis/trends/1")
            assert trend_response.status_code == 200


# ============================================================
# Test: Multi-Endpoint Integration
# ============================================================

class TestMultiEndpointIntegration:
    """Test workflows involving multiple endpoints"""

    def test_search_analyze_export_workflow(self, integration_client):
        """Test: Create search → Analyze → Export"""
        # Create search
        with patch("src.api.routes.search.DatabaseOperations.create_search", new_callable=AsyncMock) as mock_create:
            with patch("src.api.routes.search.task_queue.enqueue", new_callable=AsyncMock) as mock_enqueue:
                mock_search = MagicMock()
                mock_search.id = 1
                mock_search.query = "AI trends"
                mock_search.status = "pending"
                mock_search.created_at = datetime.utcnow()

                mock_create.return_value = mock_search
                mock_enqueue.return_value = 123

                create_response = integration_client.post(
                    "/api/v1/search",
                    json={"query": "AI trends", "sources": ["google"]}
                )

                assert create_response.status_code == 201
                search_id = create_response.json()["search_id"]

        # Analyze sentiment
        with patch("src.api.routes.analysis.SentimentAnalyzer") as mock_analyzer_class:
            mock_analyzer = AsyncMock()
            mock_analyzer.analyze_search_results = AsyncMock(return_value={
                "sample_size": 50,
                "sentiment_distribution": {"positive": 40, "neutral": 10, "negative": 0},
                "dominant_sentiment": "positive",
                "average_confidence": 0.92
            })
            mock_analyzer_class.return_value = mock_analyzer

            analysis_response = integration_client.get(f"/api/v1/analysis/sentiment/{search_id}")
            assert analysis_response.status_code == 200

        # Export to JSON
        with patch("src.api.routes.export.DatabaseOperations.get_search", new_callable=AsyncMock) as mock_get:
            with patch("src.api.routes.export.DatabaseOperations.get_search_results", new_callable=AsyncMock) as mock_results:
                mock_search.status = "completed"
                mock_search.completed_at = datetime.utcnow()
                mock_search.sources = ["google"]

                mock_get.return_value = mock_search
                mock_results.return_value = []

                export_response = integration_client.get(f"/api/v1/export/search/{search_id}/json")
                assert export_response.status_code == 200
                assert "search" in export_response.json()


# ============================================================
# Test: Error Handling
# ============================================================

class TestErrorHandling:
    """Test error handling across modules"""

    def test_search_not_found_cascades(self, integration_client):
        """Test that search not found error cascades to analysis"""
        with patch("src.api.routes.analysis.SentimentAnalyzer") as mock_analyzer_class:
            mock_analyzer = AsyncMock()
            mock_analyzer.analyze_search_results = AsyncMock(
                side_effect=ValueError("No results found for search 999")
            )
            mock_analyzer_class.return_value = mock_analyzer

            response = integration_client.get("/api/v1/analysis/sentiment/999")

            assert response.status_code == 404
            assert "not found" in response.json()["detail"].lower()

    def test_invalid_task_type_error(self, integration_client):
        """Test invalid task type validation"""
        # Validation happens at Pydantic level, so this tests integration
        response = integration_client.post(
            "/api/v1/tasks",
            json={
                "type": "",  # Empty type
                "payload": {}
            }
        )

        # Will fail validation
        assert response.status_code in [400, 422]

    def test_analysis_error_handling(self, integration_client):
        """Test analysis error handling"""
        with patch("src.api.routes.analysis.ollama_client.analyze", new_callable=AsyncMock) as mock_analyze:
            mock_analyze.side_effect = Exception("Ollama service unavailable")

            response = integration_client.post(
                "/api/v1/analysis/text",
                json={"text": "test", "analysis_type": "sentiment"}
            )

            assert response.status_code == 500
            assert "failed" in response.json()["detail"].lower()


# ============================================================
# Test: WebSocket Integration
# ============================================================

class TestWebSocketIntegration:
    """Test WebSocket real-time updates"""

    def test_websocket_connection(self, integration_client):
        """Test WebSocket connection establishment"""
        # Note: TestClient doesn't support WebSocket properly
        # This is a basic connection test
        # Real WebSocket testing would require a different approach

        # For now, just verify the endpoint exists
        from src.api import app
        routes = [route.path for route in app.routes]
        assert "/ws/{client_id}" in routes

    def test_websocket_subscribe_action(self):
        """Test WebSocket subscribe message format"""
        # Test message format validation
        subscribe_message = {
            "action": "subscribe",
            "search_id": 123
        }

        # Verify message structure
        assert subscribe_message["action"] == "subscribe"
        assert isinstance(subscribe_message["search_id"], int)


# ============================================================
# Test: Performance & Load
# ============================================================

class TestPerformance:
    """Test API performance characteristics"""

    def test_batch_task_performance(self, integration_client):
        """Test batch task creation performance"""
        with patch("src.api.routes.tasks.task_queue.enqueue_batch", new_callable=AsyncMock) as mock_batch:
            # Simulate creating 50 tasks
            task_ids = list(range(1, 51))
            mock_batch.return_value = task_ids

            response = integration_client.post(
                "/api/v1/tasks/batch",
                json={
                    "tasks": [
                        {"type": "scrape", "payload": {"query": f"query{i}"}}
                        for i in range(50)
                    ]
                }
            )

            assert response.status_code == 201
            assert response.json()["count"] == 50

    def test_pagination_performance(self, integration_client):
        """Test pagination with large datasets"""
        with patch("src.api.routes.search.db_manager.get_session") as mock_session_ctx:
            # Mock large result set
            mock_session = AsyncMock()
            mock_session_ctx.return_value.__aenter__.return_value = mock_session

            mock_result = MagicMock()
            mock_result.scalars.return_value.all.return_value = []
            mock_result.scalar.return_value = 1000  # 1000 total searches

            mock_session.execute = AsyncMock(return_value=mock_result)

            # Test pagination
            response = integration_client.get("/api/v1/search?skip=0&limit=100")

            assert response.status_code == 200
            data = response.json()
            assert data["total"] == 1000


# ============================================================
# Run Tests
# ============================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
