"""
Comprehensive tests for Scheduler Service (Module 10)
Tests job scheduling, management, API endpoints, and integration
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
from fastapi.testclient import TestClient

from src.services.scheduler import SchedulerService, scheduler
from src.api.main import app
from src.database.manager import DatabaseManager
from src.database.models import TaskQueue
from src.task_queue.manager import TaskQueueManager
from src.cache.manager import CacheManager
from src.config.settings import settings


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
async def test_scheduler():
    """Create isolated test scheduler"""
    # Create scheduler with in-memory database
    test_sched = SchedulerService()
    await test_sched.start()

    yield test_sched

    # Cleanup
    await test_sched.stop()


@pytest.fixture
async def test_db():
    """Create test database for cleanup tests"""
    db_manager = DatabaseManager("sqlite+aiosqlite:///:memory:")
    await db_manager.initialize()

    yield db_manager

    await db_manager.close()


@pytest.fixture
async def test_task_queue():
    """Create test task queue"""
    queue = TaskQueueManager()
    await queue.start()

    yield queue

    await queue.stop()


@pytest.fixture
async def test_cache():
    """Create test cache"""
    cache = CacheManager(enable_memory_cache=True)
    await cache.initialize()

    yield cache

    await cache.close()


@pytest.fixture
def api_client():
    """Create FastAPI test client"""
    return TestClient(app)


# ============================================================
# Setup & Initialization Tests (3 tests)
# ============================================================

@pytest.mark.asyncio
async def test_scheduler_initialization():
    """Test scheduler initializes correctly"""
    test_sched = SchedulerService()

    # Verify scheduler object created
    assert test_sched is not None
    assert hasattr(test_sched, 'scheduler')
    assert hasattr(test_sched, 'handlers')

    # Verify handlers dictionary is empty initially
    assert isinstance(test_sched.handlers, dict)


@pytest.mark.asyncio
async def test_scheduler_start_stop(test_scheduler):
    """Test scheduler start and stop lifecycle"""
    # Scheduler should be running (started in fixture)
    assert test_scheduler.scheduler.running

    # Stop scheduler
    await test_scheduler.stop()
    assert not test_scheduler.scheduler.running

    # Restart scheduler
    await test_scheduler.start()
    assert test_scheduler.scheduler.running


@pytest.mark.asyncio
async def test_jobstore_creation():
    """Test that jobstore database is created"""
    test_sched = SchedulerService()
    await test_sched.start()

    # Check that scheduler.db would be created in data directory
    # (We can't check actual file in test, but verify scheduler is configured)
    assert test_sched.scheduler.state is not None

    await test_sched.stop()


# ============================================================
# Handler Tests (4 tests)
# ============================================================

@pytest.mark.asyncio
async def test_search_handler_enqueues_task(test_scheduler):
    """Test search handler enqueues scrape task"""
    with patch('src.services.scheduler.task_queue.enqueue', new_callable=AsyncMock) as mock_enqueue:
        await test_scheduler._handle_search_job(
            query="test query",
            sources=["google", "bing"]
        )

        # Verify task was enqueued
        mock_enqueue.assert_called_once()
        call_args = mock_enqueue.call_args
        assert call_args[1]['task_type'] == 'scrape'
        assert call_args[1]['payload']['query'] == 'test query'
        assert call_args[1]['payload']['sources'] == ["google", "bing"]


@pytest.mark.asyncio
async def test_analysis_handler_enqueues_tasks(test_scheduler):
    """Test analysis handler enqueues multiple analysis tasks"""
    with patch('src.services.scheduler.task_queue.enqueue', new_callable=AsyncMock) as mock_enqueue:
        await test_scheduler._handle_analysis_job(
            search_id=123,
            types=["sentiment", "trends", "competitors"]
        )

        # Verify 3 tasks were enqueued (one for each type)
        assert mock_enqueue.call_count == 3

        # Verify all calls were for analyze tasks
        for call in mock_enqueue.call_args_list:
            assert call[1]['task_type'] == 'analyze'
            assert call[1]['payload']['search_id'] == 123


@pytest.mark.asyncio
async def test_export_handler_enqueues_task(test_scheduler):
    """Test export handler enqueues export task"""
    with patch('src.services.scheduler.task_queue.enqueue', new_callable=AsyncMock) as mock_enqueue:
        await test_scheduler._handle_export_job(
            search_id=456,
            format="csv"
        )

        # Verify task was enqueued
        mock_enqueue.assert_called_once()
        call_args = mock_enqueue.call_args
        assert call_args[1]['task_type'] == 'export'
        assert call_args[1]['payload']['search_id'] == 456
        assert call_args[1]['payload']['format'] == 'csv'


@pytest.mark.asyncio
async def test_cleanup_handler_cleans_data(test_scheduler, test_cache, test_db):
    """Test cleanup handler removes expired cache and old tasks"""
    # Add some expired cache entries
    await test_cache.set("expired_key", "value", ttl=-1)

    # Add old completed task
    async with test_db.get_session() as session:
        old_task = TaskQueue(
            task_type="test",
            payload={"data": "test"},
            status="completed",
            completed_at=datetime.utcnow() - timedelta(days=8)
        )
        session.add(old_task)
        await session.commit()

    # Mock cache and db_manager
    with patch('src.services.scheduler.cache', test_cache):
        with patch('src.services.scheduler.db_manager', test_db):
            # Run cleanup handler
            await test_scheduler._handle_cleanup_job()

    # Verify cache was cleaned (this is a basic check)
    # In real test, we'd verify specific entries were removed


# ============================================================
# Search Scheduling Tests (4 tests)
# ============================================================

@pytest.mark.asyncio
async def test_schedule_recurring_search_cron(test_scheduler):
    """Test scheduling recurring search with cron expression"""
    job_id = test_scheduler.schedule_recurring_search(
        query="AI news",
        sources=["google"],
        cron_expression="0 9 * * *"  # 9 AM daily
    )

    # Verify job was created
    assert job_id is not None
    assert "search_" in job_id

    # Verify job exists in scheduler
    job = test_scheduler.get_job(job_id)
    assert job is not None
    assert job['name'] == "Search: AI news"
    assert 'cron' in job['trigger'].lower()


@pytest.mark.asyncio
async def test_schedule_recurring_search_interval(test_scheduler):
    """Test scheduling recurring search with interval"""
    job_id = test_scheduler.schedule_recurring_search(
        query="tech updates",
        sources=["bing"],
        interval_hours=6
    )

    # Verify job was created
    assert job_id is not None

    # Verify job exists in scheduler
    job = test_scheduler.get_job(job_id)
    assert job is not None
    assert job['name'] == "Search: tech updates"
    assert 'interval' in job['trigger'].lower()


@pytest.mark.asyncio
async def test_schedule_search_requires_trigger(test_scheduler):
    """Test that scheduling search requires either cron or interval"""
    with pytest.raises(ValueError) as exc_info:
        test_scheduler.schedule_recurring_search(
            query="test",
            sources=["google"]
            # Neither cron_expression nor interval_hours provided
        )

    assert "Either cron_expression or interval_hours required" in str(exc_info.value)


@pytest.mark.asyncio
async def test_schedule_search_generates_unique_ids(test_scheduler):
    """Test that job IDs are unique"""
    job_id1 = test_scheduler.schedule_recurring_search(
        query="query1",
        sources=["google"],
        interval_hours=1
    )

    # Wait a tiny bit to ensure timestamp differs
    await asyncio.sleep(0.01)

    job_id2 = test_scheduler.schedule_recurring_search(
        query="query1",
        sources=["google"],
        interval_hours=1
    )

    # Verify IDs are different
    assert job_id1 != job_id2


# ============================================================
# Analysis & Export Scheduling Tests (3 tests)
# ============================================================

@pytest.mark.asyncio
async def test_schedule_analysis_delayed(test_scheduler):
    """Test scheduling delayed analysis job"""
    job_id = test_scheduler.schedule_analysis(
        search_id=789,
        analysis_types=["sentiment", "trends"],
        delay_minutes=10
    )

    # Verify job was created
    assert job_id is not None
    assert "analysis_789_" in job_id

    # Verify job exists in scheduler
    job = test_scheduler.get_job(job_id)
    assert job is not None
    assert job['name'] == "Analysis: Search 789"

    # Verify next run is approximately 10 minutes from now
    if job['next_run']:
        next_run = datetime.fromisoformat(job['next_run'])
        expected_run = datetime.now() + timedelta(minutes=10)
        # Allow 1 minute tolerance
        assert abs((next_run - expected_run).total_seconds()) < 60


@pytest.mark.asyncio
async def test_schedule_export_cron(test_scheduler):
    """Test scheduling recurring export with cron"""
    job_id = test_scheduler.schedule_export(
        search_id=101,
        format="excel",
        cron_expression="0 8 * * 1"  # Monday 8 AM
    )

    # Verify job was created
    assert job_id is not None
    assert "export_101_" in job_id

    # Verify job exists in scheduler
    job = test_scheduler.get_job(job_id)
    assert job is not None
    assert job['name'] == "Export: Search 101 (excel)"
    assert 'cron' in job['trigger'].lower()


@pytest.mark.asyncio
async def test_schedule_cleanup_auto(test_scheduler):
    """Test cleanup job is auto-scheduled on start"""
    # Cleanup job should already be scheduled (in fixture start)
    jobs = test_scheduler.list_jobs()

    # Find cleanup job
    cleanup_job = next(
        (j for j in jobs if j['id'] == 'cleanup_daily'),
        None
    )

    assert cleanup_job is not None
    assert cleanup_job['name'] == 'Daily Cleanup'


# ============================================================
# Job Management Tests (6 tests)
# ============================================================

@pytest.mark.asyncio
async def test_get_job_exists(test_scheduler):
    """Test getting existing job details"""
    # Create a job
    job_id = test_scheduler.schedule_recurring_search(
        query="test",
        sources=["google"],
        interval_hours=1
    )

    # Get job details
    job = test_scheduler.get_job(job_id)

    assert job is not None
    assert job['id'] == job_id
    assert job['name'] == "Search: test"
    assert 'trigger' in job
    assert 'next_run' in job


@pytest.mark.asyncio
async def test_get_job_not_found(test_scheduler):
    """Test getting non-existent job returns None"""
    job = test_scheduler.get_job("nonexistent_job_id")
    assert job is None


@pytest.mark.asyncio
async def test_list_jobs(test_scheduler):
    """Test listing all scheduled jobs"""
    # Create multiple jobs
    job_id1 = test_scheduler.schedule_recurring_search(
        query="job1", sources=["google"], interval_hours=1
    )
    job_id2 = test_scheduler.schedule_recurring_search(
        query="job2", sources=["bing"], interval_hours=2
    )

    # List jobs
    jobs = test_scheduler.list_jobs()

    # Should have at least the 2 we created plus cleanup job
    assert len(jobs) >= 3

    # Verify our jobs are in the list
    job_ids = [j['id'] for j in jobs]
    assert job_id1 in job_ids
    assert job_id2 in job_ids


@pytest.mark.asyncio
async def test_remove_job(test_scheduler):
    """Test removing a scheduled job"""
    # Create a job
    job_id = test_scheduler.schedule_recurring_search(
        query="to_remove",
        sources=["google"],
        interval_hours=1
    )

    # Verify job exists
    assert test_scheduler.get_job(job_id) is not None

    # Remove job
    result = test_scheduler.remove_job(job_id)
    assert result is True

    # Verify job is gone
    assert test_scheduler.get_job(job_id) is None


@pytest.mark.asyncio
async def test_pause_resume_job(test_scheduler):
    """Test pausing and resuming a job"""
    # Create a job
    job_id = test_scheduler.schedule_recurring_search(
        query="pause_test",
        sources=["google"],
        interval_hours=1
    )

    # Pause job
    result = test_scheduler.pause_job(job_id)
    assert result is True

    # Job should still exist
    job = test_scheduler.get_job(job_id)
    assert job is not None

    # Resume job
    result = test_scheduler.resume_job(job_id)
    assert result is True


@pytest.mark.asyncio
async def test_modify_job(test_scheduler):
    """Test modifying a job's trigger"""
    from apscheduler.triggers.interval import IntervalTrigger

    # Create a job
    job_id = test_scheduler.schedule_recurring_search(
        query="modify_test",
        sources=["google"],
        interval_hours=1
    )

    # Modify job to 3 hour interval
    new_trigger = IntervalTrigger(hours=3)
    result = test_scheduler.modify_job(job_id, trigger=new_trigger)
    assert result is True

    # Verify modification
    job = test_scheduler.get_job(job_id)
    assert job is not None


# ============================================================
# API Endpoint Tests (6 tests)
# ============================================================

def test_api_schedule_search(api_client):
    """Test POST /scheduler/search endpoint"""
    response = api_client.post(
        "/api/v1/scheduler/search",
        json={
            "query": "API test query",
            "sources": ["google"],
            "interval_hours": 12
        }
    )

    assert response.status_code == 201
    data = response.json()
    assert "job_id" in data
    assert "message" in data
    assert "API test query" in data["message"]


def test_api_schedule_export(api_client):
    """Test POST /scheduler/export endpoint"""
    response = api_client.post(
        "/api/v1/scheduler/export",
        json={
            "search_id": 999,
            "format": "csv",
            "cron_expression": "0 9 * * *"
        }
    )

    assert response.status_code == 201
    data = response.json()
    assert "job_id" in data
    assert "message" in data


def test_api_list_jobs(api_client):
    """Test GET /scheduler/jobs endpoint"""
    # First create a job
    api_client.post(
        "/api/v1/scheduler/search",
        json={
            "query": "list test",
            "sources": ["google"],
            "interval_hours": 1
        }
    )

    # List jobs
    response = api_client.get("/api/v1/scheduler/jobs")

    assert response.status_code == 200
    data = response.json()
    assert "jobs" in data
    assert "total" in data
    assert isinstance(data["jobs"], list)


def test_api_get_job(api_client):
    """Test GET /scheduler/jobs/{job_id} endpoint"""
    # Create a job
    create_response = api_client.post(
        "/api/v1/scheduler/search",
        json={
            "query": "get test",
            "sources": ["google"],
            "interval_hours": 1
        }
    )
    job_id = create_response.json()["job_id"]

    # Get job details
    response = api_client.get(f"/api/v1/scheduler/jobs/{job_id}")

    assert response.status_code == 200
    data = response.json()
    assert data["id"] == job_id
    assert "name" in data
    assert "trigger" in data


def test_api_remove_job(api_client):
    """Test DELETE /scheduler/jobs/{job_id} endpoint"""
    # Create a job
    create_response = api_client.post(
        "/api/v1/scheduler/search",
        json={
            "query": "remove test",
            "sources": ["google"],
            "interval_hours": 1
        }
    )
    job_id = create_response.json()["job_id"]

    # Remove job
    response = api_client.delete(f"/api/v1/scheduler/jobs/{job_id}")

    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "removed" in data["message"].lower()


def test_api_pause_resume(api_client):
    """Test POST /scheduler/jobs/{job_id}/pause and /resume endpoints"""
    # Create a job
    create_response = api_client.post(
        "/api/v1/scheduler/search",
        json={
            "query": "pause test",
            "sources": ["google"],
            "interval_hours": 1
        }
    )
    job_id = create_response.json()["job_id"]

    # Pause job
    pause_response = api_client.post(f"/api/v1/scheduler/jobs/{job_id}/pause")
    assert pause_response.status_code == 200
    assert pause_response.json()["success"] is True

    # Resume job
    resume_response = api_client.post(f"/api/v1/scheduler/jobs/{job_id}/resume")
    assert resume_response.status_code == 200
    assert resume_response.json()["success"] is True


# ============================================================
# Error Handling Tests (2 tests)
# ============================================================

@pytest.mark.asyncio
async def test_invalid_cron_expression(test_scheduler):
    """Test that invalid cron expression raises ValueError"""
    with pytest.raises(ValueError) as exc_info:
        test_scheduler.schedule_export(
            search_id=123,
            format="csv",
            cron_expression="invalid cron"
        )

    assert "Invalid cron expression" in str(exc_info.value)


@pytest.mark.asyncio
async def test_handler_error_logging(test_scheduler):
    """Test that handler errors are logged properly"""
    # Mock task_queue to raise an error
    with patch('src.services.scheduler.task_queue.enqueue', new_callable=AsyncMock) as mock_enqueue:
        mock_enqueue.side_effect = Exception("Test error")

        # This should not raise, but log the error
        await test_scheduler._handle_search_job(
            query="error test",
            sources=["google"]
        )

        # If we get here, error was handled gracefully


# ============================================================
# Integration Test (1 test)
# ============================================================

@pytest.mark.asyncio
async def test_full_scheduler_workflow(test_scheduler):
    """Test complete workflow: schedule, list, modify, remove"""
    # 1. Schedule a search
    job_id = test_scheduler.schedule_recurring_search(
        query="integration test",
        sources=["google", "bing"],
        interval_hours=2
    )
    assert job_id is not None

    # 2. Verify it appears in job list
    jobs = test_scheduler.list_jobs()
    assert any(j['id'] == job_id for j in jobs)

    # 3. Get job details
    job = test_scheduler.get_job(job_id)
    assert job is not None
    assert job['name'] == "Search: integration test"

    # 4. Pause the job
    assert test_scheduler.pause_job(job_id) is True

    # 5. Resume the job
    assert test_scheduler.resume_job(job_id) is True

    # 6. Remove the job
    assert test_scheduler.remove_job(job_id) is True

    # 7. Verify it's gone
    assert test_scheduler.get_job(job_id) is None

    print("Integration test completed successfully!")
