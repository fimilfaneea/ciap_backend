"""
Export Service for CIAP
Handles data export in various formats (CSV, Excel, JSON, Power BI, HTML)
"""

import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
import json
import logging
import re

from openpyxl import Workbook
from openpyxl.styles import PatternFill, Font, Alignment
from openpyxl.utils.dataframe import dataframe_to_rows

from ..config.settings import settings
from ..database.manager import db_manager
from ..database.operations import DatabaseOperations

logger = logging.getLogger(__name__)


class ExportService:
    """Service for exporting data in various formats"""

    SUPPORTED_FORMATS = ["csv", "excel", "xlsx", "json", "powerbi", "html"]

    def __init__(self):
        """Initialize export service with directory setup"""
        self.export_dir = Path(settings.EXPORT_DIR)
        self.max_rows = settings.EXPORT_MAX_ROWS

        # Create export directory if it doesn't exist
        self.export_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"ExportService initialized. Export directory: {self.export_dir}")

    async def export_search(
        self,
        search_id: int,
        format: str = "csv",
        include_analysis: bool = True
    ) -> str:
        """
        Export search results in specified format

        Args:
            search_id: Search ID to export
            format: Export format (csv, excel, json, powerbi, html)
            include_analysis: Include analysis results in export

        Returns:
            Path to exported file

        Raises:
            ValueError: If search not found or format not supported
            Exception: If export fails
        """
        # Validate format
        format = format.lower()
        if not self._validate_format(format):
            raise ValueError(
                f"Unsupported format: {format}. "
                f"Supported formats: {', '.join(self.SUPPORTED_FORMATS)}"
            )

        # Get search data
        data = await self._get_search_data(search_id, include_analysis)

        if not data:
            raise ValueError(f"Search {search_id} not found")

        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        query_slug = self._sanitize_filename(data["search"]["query"])
        filename = f"search_{search_id}_{query_slug}_{timestamp}"

        # Export based on format
        try:
            if format == "csv":
                filepath = self.export_dir / f"{filename}.csv"
                await self._export_csv(data, filepath)

            elif format in ["excel", "xlsx"]:
                filepath = self.export_dir / f"{filename}.xlsx"
                await self._export_excel(data, filepath, include_analysis)

            elif format == "json":
                filepath = self.export_dir / f"{filename}.json"
                await self._export_json(data, filepath)

            elif format == "powerbi":
                filepath = self.export_dir / f"{filename}_powerbi.csv"
                await self._export_powerbi(data, filepath)

            elif format == "html":
                filepath = self.export_dir / f"{filename}.html"
                await self._export_html(data, filepath)

            else:
                raise ValueError(f"Format handler not implemented: {format}")

            logger.info(
                f"Successfully exported search {search_id} to {filepath} "
                f"(format: {format}, size: {filepath.stat().st_size} bytes)"
            )

            return str(filepath)

        except Exception as e:
            logger.error(f"Export failed for search {search_id} (format: {format}): {e}")
            raise

    async def _get_search_data(
        self,
        search_id: int,
        include_analysis: bool = True
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve all data for a search

        Args:
            search_id: Search ID
            include_analysis: Include cached analysis data

        Returns:
            Dictionary containing search, results, and analysis data
        """
        try:
            async with db_manager.get_session() as session:
                # Get search
                search = await DatabaseOperations.get_search(session, search_id)

                if not search:
                    return None

                # Get results
                results = await DatabaseOperations.get_search_results(
                    session,
                    search_id,
                    limit=self.max_rows
                )

                # Prepare search metadata
                search_data = {
                    "id": search.id,
                    "query": search.query,
                    "status": search.status,
                    "sources": search.sources,
                    "created_at": search.created_at,
                    "completed_at": search.completed_at
                }

                # Prepare results data
                results_data = [
                    {
                        "source": r.source,
                        "position": r.position,
                        "title": r.title,
                        "snippet": r.snippet,
                        "url": r.url,
                        "scraped_at": r.scraped_at,
                        "sentiment_score": r.sentiment_score
                    }
                    for r in results
                ]

                # Get analysis data from cache if requested
                analysis_data = None
                if include_analysis:
                    from ..cache.manager import cache

                    sentiment = await cache.get(f"analysis:{search_id}:sentiment")
                    competitors = await cache.get(f"analysis:{search_id}:competitors")
                    trends = await cache.get(f"analysis:{search_id}:trends")
                    insights = await cache.get(f"analysis:{search_id}:insights")

                    analysis_data = {
                        "sentiment": sentiment,
                        "competitors": competitors,
                        "trends": trends,
                        "insights": insights
                    }

                return {
                    "search": search_data,
                    "results": results_data,
                    "analysis": analysis_data
                }

        except Exception as e:
            logger.error(f"Failed to retrieve search data for {search_id}: {e}")
            raise

    async def _export_csv(
        self,
        data: Dict[str, Any],
        filepath: Path
    ):
        """
        Export to CSV format

        Args:
            data: Search data dictionary
            filepath: Output file path
        """
        # Convert results to DataFrame
        if not data["results"]:
            # Create empty DataFrame with expected columns
            df = pd.DataFrame(columns=[
                "search_id", "query", "source", "position", "title",
                "snippet", "url", "scraped_at", "sentiment_score"
            ])
        else:
            df = pd.DataFrame(data["results"])

            # Add search metadata columns
            df.insert(0, "search_id", data["search"]["id"])
            df.insert(1, "query", data["search"]["query"])

        # Apply max rows limit
        if len(df) > self.max_rows:
            df = df.head(self.max_rows)
            logger.warning(
                f"CSV export limited to {self.max_rows} rows "
                f"(original: {len(data['results'])} rows)"
            )

        # Save to CSV with UTF-8 BOM for Excel compatibility
        df.to_csv(filepath, index=False, encoding="utf-8-sig")

        logger.info(f"CSV export completed: {len(df)} rows written to {filepath}")

    async def _export_excel(
        self,
        data: Dict[str, Any],
        filepath: Path,
        include_analysis: bool
    ):
        """
        Export to Excel format with multiple sheets and formatting

        Args:
            data: Search data dictionary
            filepath: Output file path
            include_analysis: Include analysis sheets
        """
        with pd.ExcelWriter(filepath, engine="openpyxl") as writer:
            # Sheet 1: Search Info
            search_info = pd.DataFrame([{
                "Search ID": data["search"]["id"],
                "Query": data["search"]["query"],
                "Status": data["search"]["status"],
                "Sources": ", ".join(data["search"]["sources"]) if data["search"]["sources"] else "",
                "Created At": self._serialize_datetime(data["search"]["created_at"]),
                "Completed At": self._serialize_datetime(data["search"]["completed_at"]),
                "Total Results": len(data["results"]),
                "Exported At": datetime.now().isoformat(),
                "Max Rows Limit": self.max_rows
            }])
            search_info.to_excel(writer, sheet_name="Search Info", index=False)

            # Sheet 2: Results
            if data["results"]:
                results_df = pd.DataFrame(data["results"])

                # Apply max rows limit
                if len(results_df) > self.max_rows:
                    results_df = results_df.head(self.max_rows)
                    logger.warning(
                        f"Excel export limited to {self.max_rows} rows "
                        f"(original: {len(data['results'])} rows)"
                    )

                # Convert datetime columns
                if "scraped_at" in results_df.columns:
                    results_df["scraped_at"] = results_df["scraped_at"].apply(
                        self._serialize_datetime
                    )

                results_df.to_excel(writer, sheet_name="Results", index=False)
            else:
                # Empty results sheet
                pd.DataFrame(columns=[
                    "source", "position", "title", "snippet",
                    "url", "scraped_at", "sentiment_score"
                ]).to_excel(writer, sheet_name="Results", index=False)

            # Sheet 3-5: Analysis sheets (if available and requested)
            if include_analysis and data["analysis"]:
                # Sentiment sheet
                if data["analysis"]["sentiment"]:
                    sentiment_data = data["analysis"]["sentiment"]
                    sentiment_df = pd.DataFrame([{
                        "Dominant Sentiment": sentiment_data.get("dominant_sentiment", "N/A"),
                        "Average Confidence": sentiment_data.get("average_confidence", 0),
                        "Positive Count": sentiment_data.get("positive_count", 0),
                        "Negative Count": sentiment_data.get("negative_count", 0),
                        "Neutral Count": sentiment_data.get("neutral_count", 0),
                        "Total Analyzed": sentiment_data.get("total_analyzed", 0)
                    }])
                    sentiment_df.to_excel(writer, sheet_name="Sentiment", index=False)

                # Trends sheet
                if data["analysis"]["trends"]:
                    trends_data = data["analysis"]["trends"]
                    top_trends = trends_data.get("top_trends", [])

                    if top_trends:
                        trends_df = pd.DataFrame(top_trends)
                        trends_df.to_excel(writer, sheet_name="Trends", index=False)

                # Competitors sheet
                if data["analysis"]["competitors"]:
                    competitors_data = data["analysis"]["competitors"]
                    top_competitors = competitors_data.get("top_competitors", [])

                    if top_competitors:
                        competitors_df = pd.DataFrame(top_competitors)
                        competitors_df.to_excel(writer, sheet_name="Competitors", index=False)

            # Apply formatting
            workbook = writer.book
            self._format_excel_workbook(workbook)

        logger.info(f"Excel export completed: {filepath}")

    def _format_excel_workbook(self, workbook: Workbook):
        """
        Apply formatting to Excel workbook

        Args:
            workbook: openpyxl Workbook object
        """
        # Header formatting
        header_fill = PatternFill(
            start_color="366092",
            end_color="366092",
            fill_type="solid"
        )
        header_font = Font(color="FFFFFF", bold=True)
        header_alignment = Alignment(horizontal="center", vertical="center")

        # Format each sheet
        for sheet in workbook.worksheets:
            # Auto-adjust column widths
            for column in sheet.columns:
                max_length = 0
                column_letter = column[0].column_letter

                for cell in column:
                    try:
                        if cell.value:
                            cell_length = len(str(cell.value))
                            if cell_length > max_length:
                                max_length = cell_length
                    except:
                        pass

                # Set width with max limit of 50
                adjusted_width = min(max_length + 2, 50)
                sheet.column_dimensions[column_letter].width = adjusted_width

            # Apply header formatting to first row
            for cell in sheet[1]:
                cell.fill = header_fill
                cell.font = header_font
                cell.alignment = header_alignment

    async def _export_json(
        self,
        data: Dict[str, Any],
        filepath: Path
    ):
        """
        Export to JSON format with full data structure

        Args:
            data: Search data dictionary
            filepath: Output file path
        """
        # Prepare export data with metadata
        export_data = {
            "metadata": {
                "exported_at": datetime.now().isoformat(),
                "version": "0.9.0",
                "format": "json",
                "export_service": "CIAP Export Service"
            },
            "search": {
                "id": data["search"]["id"],
                "query": data["search"]["query"],
                "status": data["search"]["status"],
                "sources": data["search"]["sources"],
                "created_at": self._serialize_datetime(data["search"]["created_at"]),
                "completed_at": self._serialize_datetime(data["search"]["completed_at"])
            },
            "results": [
                {
                    "source": r["source"],
                    "position": r["position"],
                    "title": r["title"],
                    "snippet": r["snippet"],
                    "url": r["url"],
                    "scraped_at": self._serialize_datetime(r["scraped_at"]),
                    "sentiment_score": r["sentiment_score"]
                }
                for r in data["results"][:self.max_rows]
            ],
            "analysis": data["analysis"] if data["analysis"] else None,
            "statistics": {
                "total_results": len(data["results"]),
                "exported_results": min(len(data["results"]), self.max_rows),
                "results_truncated": len(data["results"]) > self.max_rows
            }
        }

        # Write to file with pretty printing
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)

        logger.info(
            f"JSON export completed: {len(export_data['results'])} results "
            f"written to {filepath}"
        )

    async def _export_powerbi(
        self,
        data: Dict[str, Any],
        filepath: Path
    ):
        """
        Export in Power BI optimized format (denormalized CSV)

        Args:
            data: Search data dictionary
            filepath: Output file path
        """
        # Create denormalized flat structure for Power BI
        powerbi_data = []

        # Extract analysis summary fields
        overall_sentiment = None
        dominant_trend = None

        if data["analysis"]:
            if data["analysis"]["sentiment"]:
                overall_sentiment = data["analysis"]["sentiment"].get(
                    "dominant_sentiment"
                )

            if data["analysis"]["trends"]:
                top_trends = data["analysis"]["trends"].get("top_trends", [])
                if top_trends and len(top_trends) > 0:
                    # Get first trend if it's a dict with 'trend' key
                    if isinstance(top_trends[0], dict):
                        dominant_trend = top_trends[0].get("trend") or top_trends[0].get("keyword")
                    else:
                        dominant_trend = str(top_trends[0])

        # Create denormalized rows (join search metadata with each result)
        for result in data["results"][:self.max_rows]:
            row = {
                "SearchID": data["search"]["id"],
                "Query": data["search"]["query"],
                "SearchDate": self._serialize_datetime(data["search"]["created_at"]),
                "SearchStatus": data["search"]["status"],
                "Source": result["source"],
                "Position": result["position"],
                "Title": result["title"],
                "Snippet": result["snippet"],
                "URL": result["url"],
                "ScrapedAt": self._serialize_datetime(result["scraped_at"]),
                "SentimentScore": result["sentiment_score"] if result["sentiment_score"] is not None else "",
                "OverallSentiment": overall_sentiment or "",
                "DominantTrend": dominant_trend or ""
            }
            powerbi_data.append(row)

        # Convert to DataFrame
        if powerbi_data:
            df = pd.DataFrame(powerbi_data)
        else:
            # Create empty DataFrame with expected columns
            df = pd.DataFrame(columns=[
                "SearchID", "Query", "SearchDate", "SearchStatus",
                "Source", "Position", "Title", "Snippet", "URL",
                "ScrapedAt", "SentimentScore", "OverallSentiment", "DominantTrend"
            ])

        # Clean column names for Power BI (already in Title Case with spaces)
        # No changes needed as we defined them correctly above

        # Save as CSV with UTF-8 BOM for Power BI/Excel compatibility
        df.to_csv(filepath, index=False, encoding="utf-8-sig")

        logger.info(
            f"Power BI export completed: {len(df)} denormalized rows "
            f"written to {filepath}"
        )

    async def _export_html(
        self,
        data: Dict[str, Any],
        filepath: Path
    ):
        """
        Export as HTML report with professional formatting

        Args:
            data: Search data dictionary
            filepath: Output file path
        """
        # Render HTML template
        html_content = self._render_report_template(data, "default")

        # Write to file
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(html_content)

        logger.info(f"HTML report generated: {filepath}")

    async def generate_report(
        self,
        search_id: int,
        template: str = "default"
    ) -> str:
        """
        Generate formatted HTML report

        Args:
            search_id: Search ID to generate report for
            template: Report template name (default, executive, detailed)

        Returns:
            Path to generated report file

        Raises:
            ValueError: If search not found
        """
        # Get search data
        data = await self._get_search_data(search_id, include_analysis=True)

        if not data:
            raise ValueError(f"Search {search_id} not found")

        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        query_slug = self._sanitize_filename(data["search"]["query"])
        filename = f"report_{search_id}_{query_slug}_{timestamp}.html"
        filepath = self.export_dir / filename

        # Export as HTML
        await self._export_html(data, filepath)

        return str(filepath)

    def _render_report_template(
        self,
        data: Dict[str, Any],
        template: str
    ) -> str:
        """
        Render HTML report template with data

        Args:
            data: Search data dictionary
            template: Template name

        Returns:
            Rendered HTML content
        """
        # Get sentiment color and analysis
        sentiment_class = "neutral"
        sentiment_text = "Unknown"
        sentiment_confidence = 0.0

        if data["analysis"] and data["analysis"]["sentiment"]:
            sentiment_data = data["analysis"]["sentiment"]
            sentiment_text = sentiment_data.get("dominant_sentiment", "Unknown").title()
            sentiment_confidence = sentiment_data.get("average_confidence", 0.0)

            # Map sentiment to CSS class
            sentiment_lower = sentiment_text.lower()
            if sentiment_lower == "positive":
                sentiment_class = "positive"
            elif sentiment_lower == "negative":
                sentiment_class = "negative"
            else:
                sentiment_class = "neutral"

        # Build HTML content
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CIAP Report - {data['search']['query']}</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
            color: #333;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #366092;
            margin-top: 0;
            border-bottom: 3px solid #366092;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #366092;
            margin-top: 30px;
            border-bottom: 1px solid #ddd;
            padding-bottom: 5px;
        }}
        .metadata {{
            background: #f5f5f5;
            padding: 15px;
            border-radius: 5px;
            margin: 20px 0;
        }}
        .metadata p {{
            margin: 5px 0;
        }}
        .metadata strong {{
            display: inline-block;
            width: 150px;
        }}
        .analysis {{
            background: #e8f4f8;
            padding: 15px;
            border-left: 4px solid #366092;
            margin: 20px 0;
        }}
        .sentiment-positive {{
            color: #28a745;
            font-weight: bold;
        }}
        .sentiment-negative {{
            color: #dc3545;
            font-weight: bold;
        }}
        .sentiment-neutral {{
            color: #6c757d;
            font-weight: bold;
        }}
        .results {{
            margin-top: 30px;
        }}
        .result {{
            border: 1px solid #ddd;
            padding: 15px;
            margin: 15px 0;
            border-radius: 5px;
            background-color: #fafafa;
        }}
        .result h3 {{
            margin: 0 0 10px 0;
            color: #366092;
            font-size: 18px;
        }}
        .result p {{
            margin: 8px 0;
            line-height: 1.5;
        }}
        .result a {{
            color: #366092;
            text-decoration: none;
            word-break: break-all;
        }}
        .result a:hover {{
            text-decoration: underline;
        }}
        .result-meta {{
            font-size: 12px;
            color: #6c757d;
            margin-top: 10px;
        }}
        .trends {{
            background: #fff3cd;
            padding: 15px;
            border-left: 4px solid #ffc107;
            margin: 20px 0;
        }}
        .trend-list {{
            list-style: none;
            padding: 0;
        }}
        .trend-list li {{
            padding: 5px 0;
            border-bottom: 1px solid #e0e0e0;
        }}
        .trend-list li:last-child {{
            border-bottom: none;
        }}
        .footer {{
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
            text-align: center;
            color: #6c757d;
            font-size: 14px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🔍 Competitive Intelligence Report</h1>

        <div class="metadata">
            <h2>Search Information</h2>
            <p><strong>Query:</strong> {data['search']['query']}</p>
            <p><strong>Search ID:</strong> {data['search']['id']}</p>
            <p><strong>Status:</strong> {data['search']['status'].title()}</p>
            <p><strong>Sources:</strong> {', '.join(data['search']['sources']) if data['search']['sources'] else 'N/A'}</p>
            <p><strong>Created:</strong> {self._serialize_datetime(data['search']['created_at'])}</p>
            <p><strong>Completed:</strong> {self._serialize_datetime(data['search']['completed_at']) if data['search']['completed_at'] else 'In Progress'}</p>
            <p><strong>Total Results:</strong> {len(data['results'])}</p>
            <p><strong>Report Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
"""

        # Add sentiment analysis section
        if data["analysis"] and data["analysis"]["sentiment"]:
            sentiment_data = data["analysis"]["sentiment"]
            html += f"""
        <div class="analysis">
            <h2>📊 Sentiment Analysis</h2>
            <p>
                <strong>Overall Sentiment:</strong>
                <span class="sentiment-{sentiment_class}">{sentiment_text}</span>
            </p>
            <p><strong>Confidence:</strong> {sentiment_confidence:.1%}</p>
            <p><strong>Breakdown:</strong></p>
            <ul>
                <li>Positive: {sentiment_data.get('positive_count', 0)} results</li>
                <li>Negative: {sentiment_data.get('negative_count', 0)} results</li>
                <li>Neutral: {sentiment_data.get('neutral_count', 0)} results</li>
            </ul>
        </div>
"""

        # Add trends section
        if data["analysis"] and data["analysis"]["trends"]:
            trends_data = data["analysis"]["trends"]
            top_trends = trends_data.get("top_trends", [])

            if top_trends:
                html += """
        <div class="trends">
            <h2>📈 Top Trends</h2>
            <ul class="trend-list">
"""
                for trend in top_trends[:10]:  # Top 10 trends
                    if isinstance(trend, dict):
                        trend_text = trend.get("trend") or trend.get("keyword") or str(trend)
                    else:
                        trend_text = str(trend)
                    html += f"                <li>{trend_text}</li>\n"

                html += """            </ul>
        </div>
"""

        # Add top results section
        html += """
        <div class="results">
            <h2>🔗 Top Results</h2>
"""

        # Show top 20 results
        for i, result in enumerate(data["results"][:20], 1):
            html += f"""
            <div class="result">
                <h3>{i}. {result['title']}</h3>
                <p>{result['snippet']}</p>
                <p><a href="{result['url']}" target="_blank">{result['url']}</a></p>
                <div class="result-meta">
                    <small>
                        Source: {result['source'].title()} |
                        Position: #{result['position']} |
                        Scraped: {self._serialize_datetime(result['scraped_at'])}
                    </small>
                </div>
            </div>
"""

        # Add footer
        html += """
        </div>

        <div class="footer">
            <p>Generated by CIAP - Competitive Intelligence Automation Platform v0.9.0</p>
            <p>© 2025 CIAP. All rights reserved.</p>
        </div>
    </div>
</body>
</html>
"""

        return html

    def _validate_format(self, format: str) -> bool:
        """
        Validate export format

        Args:
            format: Format string to validate

        Returns:
            True if format is supported, False otherwise
        """
        return format.lower() in self.SUPPORTED_FORMATS

    def _sanitize_filename(self, filename: str, max_length: int = 50) -> str:
        """
        Sanitize filename by removing invalid characters

        Args:
            filename: Original filename
            max_length: Maximum filename length

        Returns:
            Sanitized filename
        """
        # Remove or replace invalid characters
        sanitized = re.sub(r'[<>:"/\\|?*]', '_', filename)

        # Replace spaces with underscores
        sanitized = sanitized.replace(' ', '_')

        # Remove multiple consecutive underscores
        sanitized = re.sub(r'_+', '_', sanitized)

        # Truncate to max length
        if len(sanitized) > max_length:
            sanitized = sanitized[:max_length]

        # Remove leading/trailing underscores
        sanitized = sanitized.strip('_')

        return sanitized.lower()

    def _serialize_datetime(self, dt: Optional[datetime]) -> str:
        """
        Serialize datetime to ISO format string

        Args:
            dt: Datetime object or None

        Returns:
            ISO format string or empty string
        """
        if dt is None:
            return ""

        if isinstance(dt, datetime):
            return dt.isoformat()

        # If it's already a string, return as-is
        return str(dt)


# Global export service instance (singleton)
export_service = ExportService()
