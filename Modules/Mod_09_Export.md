# Module 9: Export System

## Overview
**Purpose:** Export data in various formats for Power BI and other BI tools integration.

**Responsibilities:**
- CSV/Excel export
- JSON export
- PDF report generation
- Power BI compatible formats
- Custom report templates
- Scheduled exports

**Development Time:** 2 days (Week 10, Day 39-42)

---

## Implementation Guide

### Export Service (`src/services/export_service.py`)

```python
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
import json
import logging

from openpyxl import Workbook
from openpyxl.styles import PatternFill, Font, Alignment
from openpyxl.utils.dataframe import dataframe_to_rows

from src.core.config import settings
from src.core.database import db_manager
from src.core.models import Search, SearchResult

logger = logging.getLogger(__name__)


class ExportService:
    """Service for exporting data in various formats"""

    def __init__(self):
        self.export_dir = Path(settings.EXPORT_DIR)
        self.export_dir.mkdir(parents=True, exist_ok=True)

    async def export_search(
        self,
        search_id: int,
        format: str = "csv",
        include_analysis: bool = True
    ) -> str:
        """
        Export search results

        Args:
            search_id: Search ID
            format: Export format (csv, excel, json)
            include_analysis: Include analysis results

        Returns:
            Path to exported file
        """
        # Get search data
        data = await self._get_search_data(search_id)

        if not data:
            raise ValueError(f"Search {search_id} not found")

        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"search_{search_id}_{timestamp}.{format}"
        filepath = self.export_dir / filename

        # Export based on format
        if format == "csv":
            await self._export_csv(data, filepath)
        elif format in ["excel", "xlsx"]:
            await self._export_excel(data, filepath, include_analysis)
        elif format == "json":
            await self._export_json(data, filepath)
        elif format == "powerbi":
            await self._export_powerbi(data, filepath)
        else:
            raise ValueError(f"Unsupported format: {format}")

        logger.info(f"Exported search {search_id} to {filepath}")
        return str(filepath)

    async def _get_search_data(
        self,
        search_id: int
    ) -> Optional[Dict[str, Any]]:
        """Get all data for a search"""
        async with db_manager.get_session() as session:
            # Get search
            search = await session.get(Search, search_id)
            if not search:
                return None

            # Get results
            results = await session.execute(
                select(SearchResult)
                .where(SearchResult.search_id == search_id)
                .order_by(SearchResult.source, SearchResult.position)
            )
            search_results = results.scalars().all()

            # Get analysis if available
            from src.core.cache import cache
            sentiment = await cache.get(f"analysis:{search_id}:sentiment")
            competitors = await cache.get(f"analysis:{search_id}:competitors")
            trends = await cache.get(f"analysis:{search_id}:trends")

            return {
                "search": {
                    "id": search.id,
                    "query": search.query,
                    "status": search.status,
                    "sources": search.sources,
                    "created_at": search.created_at,
                    "completed_at": search.completed_at
                },
                "results": [
                    {
                        "source": r.source,
                        "position": r.position,
                        "title": r.title,
                        "snippet": r.snippet,
                        "url": r.url,
                        "scraped_at": r.scraped_at,
                        "sentiment": r.sentiment_score
                    }
                    for r in search_results
                ],
                "analysis": {
                    "sentiment": sentiment,
                    "competitors": competitors,
                    "trends": trends
                }
            }

    async def _export_csv(
        self,
        data: Dict[str, Any],
        filepath: Path
    ):
        """Export to CSV"""
        df = pd.DataFrame(data["results"])

        # Add search metadata
        df["search_id"] = data["search"]["id"]
        df["query"] = data["search"]["query"]

        # Save to CSV
        df.to_csv(filepath, index=False)

    async def _export_excel(
        self,
        data: Dict[str, Any],
        filepath: Path,
        include_analysis: bool
    ):
        """Export to Excel with multiple sheets"""
        with pd.ExcelWriter(filepath, engine="openpyxl") as writer:
            # Search metadata sheet
            search_df = pd.DataFrame([data["search"]])
            search_df.to_excel(writer, sheet_name="Search Info", index=False)

            # Results sheet
            results_df = pd.DataFrame(data["results"])
            results_df.to_excel(writer, sheet_name="Results", index=False)

            # Analysis sheets
            if include_analysis and data["analysis"]:
                if data["analysis"]["sentiment"]:
                    sentiment_df = pd.DataFrame([data["analysis"]["sentiment"]])
                    sentiment_df.to_excel(
                        writer,
                        sheet_name="Sentiment",
                        index=False
                    )

                if data["analysis"]["trends"]:
                    trends_df = pd.DataFrame(
                        data["analysis"]["trends"].get("top_trends", []),
                        columns=["Trend"]
                    )
                    trends_df.to_excel(
                        writer,
                        sheet_name="Trends",
                        index=False
                    )

            # Format the workbook
            workbook = writer.book
            for sheet in workbook.worksheets:
                # Auto-adjust column widths
                for column in sheet.columns:
                    max_length = 0
                    column_letter = column[0].column_letter

                    for cell in column:
                        try:
                            if len(str(cell.value)) > max_length:
                                max_length = len(str(cell.value))
                        except:
                            pass

                    adjusted_width = min(max_length + 2, 50)
                    sheet.column_dimensions[column_letter].width = adjusted_width

                # Add header formatting
                for cell in sheet[1]:
                    cell.font = Font(bold=True)
                    cell.fill = PatternFill(
                        start_color="366092",
                        end_color="366092",
                        fill_type="solid"
                    )
                    cell.font = Font(color="FFFFFF", bold=True)

    async def _export_json(
        self,
        data: Dict[str, Any],
        filepath: Path
    ):
        """Export to JSON"""
        # Convert datetime objects to strings
        def serialize(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            raise TypeError(f"Type {type(obj)} not serializable")

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2, default=serialize)

    async def _export_powerbi(
        self,
        data: Dict[str, Any],
        filepath: Path
    ):
        """Export in Power BI optimized format"""
        # Create denormalized dataset for Power BI
        powerbi_data = []

        for result in data["results"]:
            row = {
                "SearchID": data["search"]["id"],
                "Query": data["search"]["query"],
                "SearchDate": data["search"]["created_at"].isoformat(),
                **result
            }

            # Add analysis data if available
            if data["analysis"]["sentiment"]:
                row["OverallSentiment"] = data["analysis"]["sentiment"].get(
                    "dominant_sentiment"
                )

            powerbi_data.append(row)

        # Save as CSV (Power BI friendly)
        df = pd.DataFrame(powerbi_data)

        # Clean column names for Power BI
        df.columns = [col.replace("_", " ").title() for col in df.columns]

        # Save with specific settings for Power BI
        df.to_csv(
            filepath.with_suffix(".csv"),
            index=False,
            encoding="utf-8-sig"  # BOM for Excel/Power BI
        )

    async def generate_report(
        self,
        search_id: int,
        template: str = "default"
    ) -> str:
        """
        Generate formatted report

        Args:
            search_id: Search ID
            template: Report template name

        Returns:
            Path to report file
        """
        data = await self._get_search_data(search_id)

        # Generate HTML report
        html_content = self._render_report_template(data, template)

        # Save report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"report_{search_id}_{timestamp}.html"
        filepath = self.export_dir / filename

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(html_content)

        return str(filepath)

    def _render_report_template(
        self,
        data: Dict[str, Any],
        template: str
    ) -> str:
        """Render report HTML template"""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>CIAP Report - {data['search']['query']}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: #333; }}
                .metadata {{ background: #f5f5f5; padding: 10px; }}
                .results {{ margin-top: 20px; }}
                .result {{ border: 1px solid #ddd; padding: 10px; margin: 10px 0; }}
                .sentiment-positive {{ color: green; }}
                .sentiment-negative {{ color: red; }}
                .sentiment-neutral {{ color: gray; }}
            </style>
        </head>
        <body>
            <h1>Competitive Intelligence Report</h1>

            <div class="metadata">
                <h2>Search Information</h2>
                <p><strong>Query:</strong> {data['search']['query']}</p>
                <p><strong>Date:</strong> {data['search']['created_at']}</p>
                <p><strong>Sources:</strong> {', '.join(data['search']['sources'])}</p>
                <p><strong>Total Results:</strong> {len(data['results'])}</p>
            </div>
        """

        # Add sentiment analysis if available
        if data["analysis"]["sentiment"]:
            sentiment = data["analysis"]["sentiment"]
            html += f"""
            <div class="analysis">
                <h2>Sentiment Analysis</h2>
                <p><strong>Overall Sentiment:</strong>
                   <span class="sentiment-{sentiment.get('dominant_sentiment', 'neutral')}">
                   {sentiment.get('dominant_sentiment', 'Unknown')}
                   </span>
                </p>
                <p><strong>Confidence:</strong> {sentiment.get('average_confidence', 0):.2%}</p>
            </div>
            """

        # Add top results
        html += """
            <div class="results">
                <h2>Top Results</h2>
        """

        for result in data["results"][:20]:
            html += f"""
                <div class="result">
                    <h3>{result['title']}</h3>
                    <p>{result['snippet']}</p>
                    <a href="{result['url']}" target="_blank">{result['url']}</a>
                    <p><small>Source: {result['source']} | Position: {result['position']}</small></p>
                </div>
            """

        html += """
            </div>
        </body>
        </html>
        """

        return html

    async def schedule_export(
        self,
        search_id: int,
        format: str,
        schedule: str
    ) -> int:
        """
        Schedule recurring export

        Args:
            search_id: Search ID
            format: Export format
            schedule: Cron expression

        Returns:
            Schedule ID
        """
        from src.core.queue import task_queue

        # Create export task
        task_id = await task_queue.enqueue(
            task_type="export",
            payload={
                "search_id": search_id,
                "format": format,
                "scheduled": True
            }
        )

        return task_id


# Global export service instance
export_service = ExportService()
```

### Export API Routes (`src/api/routes/export.py`)

```python
from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse

router = APIRouter()


@router.post("/{search_id}")
async def export_search(
    search_id: int,
    format: str = "csv",
    background_tasks: BackgroundTasks = None
):
    """Export search results"""
    try:
        filepath = await export_service.export_search(
            search_id,
            format
        )

        return FileResponse(
            filepath,
            media_type="application/octet-stream",
            filename=Path(filepath).name
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{search_id}/report")
async def generate_report(search_id: int):
    """Generate HTML report"""
    try:
        filepath = await export_service.generate_report(search_id)

        return {
            "report_path": filepath,
            "download_url": f"/api/v1/export/download/{Path(filepath).name}"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/download/{filename}")
async def download_export(filename: str):
    """Download exported file"""
    filepath = Path(settings.EXPORT_DIR) / filename

    if not filepath.exists():
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(filepath, filename=filename)
```

---

## Testing

```python
@pytest.mark.asyncio
async def test_csv_export():
    service = ExportService()

    # Mock data
    data = {
        "search": {"id": 1, "query": "test"},
        "results": [
            {"title": "Result 1", "url": "http://example.com"}
        ]
    }

    filepath = await service._export_csv(data, Path("test.csv"))
    assert Path(filepath).exists()

    # Read and verify
    df = pd.read_csv(filepath)
    assert len(df) == 1
    assert df.iloc[0]["title"] == "Result 1"
```

---

## Module Checklist

- [ ] CSV export working
- [ ] Excel export with formatting
- [ ] JSON export functional
- [ ] Power BI format optimized
- [ ] HTML report generation
- [ ] Export scheduling
- [ ] File download endpoints
- [ ] Unit tests passing

---

## Next Steps
- Module 10: Scheduler - Job scheduling system