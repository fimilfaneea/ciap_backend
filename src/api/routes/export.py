"""
Export Routes for CIAP API
Full export functionality implementation (Module 9)
"""

from fastapi import APIRouter, HTTPException, Query, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional
from pathlib import Path
from datetime import datetime
import mimetypes
import logging
import os

from ...services.export_service import export_service
from ...config.settings import settings

logger = logging.getLogger(__name__)

router = APIRouter()


# ============================================================
# Pydantic Models
# ============================================================

class ExportRequest(BaseModel):
    """Request model for export endpoint"""
    format: str = Field(
        ...,
        description="Export format",
        pattern="^(csv|excel|xlsx|json|powerbi|html)$"
    )
    include_analysis: bool = Field(
        default=True,
        description="Include analysis data in export"
    )


class ExportResponse(BaseModel):
    """Response model for export endpoint"""
    filename: str = Field(..., description="Name of exported file")
    filepath: str = Field(..., description="Absolute path to exported file")
    format: str = Field(..., description="Export format used")
    size_bytes: int = Field(..., description="File size in bytes")
    download_url: str = Field(..., description="URL to download the file")
    created_at: datetime = Field(..., description="Export creation timestamp")


class ReportResponse(BaseModel):
    """Response model for report generation endpoint"""
    filename: str = Field(..., description="Name of report file")
    filepath: str = Field(..., description="Absolute path to report file")
    size_bytes: int = Field(..., description="File size in bytes")
    download_url: str = Field(..., description="URL to download the report")
    created_at: datetime = Field(..., description="Report creation timestamp")


class DeleteResponse(BaseModel):
    """Response model for delete endpoint"""
    filename: str = Field(..., description="Name of deleted file")
    success: bool = Field(..., description="Whether deletion was successful")
    message: str = Field(..., description="Status message")


# ============================================================
# Export Endpoints
# ============================================================

@router.post("/{search_id}", response_model=ExportResponse, status_code=200)
async def export_search(
    search_id: int,
    format: str = Query(
        "csv",
        description="Export format (csv, excel, json, powerbi, html)",
        pattern="^(csv|excel|xlsx|json|powerbi|html)$"
    ),
    include_analysis: bool = Query(
        True,
        description="Include analysis data in export"
    ),
    background_tasks: BackgroundTasks = None
):
    """
    Export search results in specified format

    Supports multiple export formats:
    - **csv**: Comma-separated values
    - **excel/xlsx**: Multi-sheet Excel workbook with formatting
    - **json**: JSON format with full data structure
    - **powerbi**: Power BI optimized denormalized CSV
    - **html**: Styled HTML report

    Args:
        search_id: Search ID to export
        format: Export format (csv/excel/json/powerbi/html)
        include_analysis: Include cached analysis data
        background_tasks: Background tasks for large exports

    Returns:
        Export metadata and download URL

    Raises:
        HTTPException 404: Search not found
        HTTPException 400: Invalid format
        HTTPException 500: Export failed
    """
    logger.info(
        f"Export requested: search_id={search_id}, format={format}, "
        f"include_analysis={include_analysis}"
    )

    try:
        # Perform export
        filepath = await export_service.export_search(
            search_id=search_id,
            format=format,
            include_analysis=include_analysis
        )

        # Get file info
        file_path = Path(filepath)
        if not file_path.exists():
            raise HTTPException(
                status_code=500,
                detail="Export file was created but cannot be found"
            )

        file_size = file_path.stat().st_size
        filename = file_path.name

        # Build download URL
        download_url = f"/api/v1/export/download/{filename}"

        logger.info(
            f"Export completed: {filename} ({file_size} bytes)"
        )

        return ExportResponse(
            filename=filename,
            filepath=str(filepath),
            format=format,
            size_bytes=file_size,
            download_url=download_url,
            created_at=datetime.now()
        )

    except ValueError as e:
        # Search not found or invalid format
        logger.error(f"Export validation error: {e}")
        raise HTTPException(status_code=404, detail=str(e))

    except Exception as e:
        logger.error(f"Export failed for search {search_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Export failed: {str(e)}"
        )


@router.post("/{search_id}/report", response_model=ReportResponse, status_code=200)
async def generate_report(
    search_id: int,
    template: str = Query(
        "default",
        description="Report template (default, executive, detailed)"
    )
):
    """
    Generate comprehensive HTML report

    Creates a professionally formatted HTML report with:
    - Search metadata and statistics
    - Sentiment analysis (if available)
    - Trend analysis (if available)
    - Top 20 search results with formatting
    - Professional CSS styling

    Args:
        search_id: Search ID to generate report for
        template: Report template name (currently only 'default' supported)

    Returns:
        Report metadata and download URL

    Raises:
        HTTPException 404: Search not found
        HTTPException 500: Report generation failed
    """
    logger.info(
        f"Report generation requested: search_id={search_id}, template={template}"
    )

    try:
        # Generate report
        filepath = await export_service.generate_report(
            search_id=search_id,
            template=template
        )

        # Get file info
        file_path = Path(filepath)
        if not file_path.exists():
            raise HTTPException(
                status_code=500,
                detail="Report was created but cannot be found"
            )

        file_size = file_path.stat().st_size
        filename = file_path.name

        # Build download URL
        download_url = f"/api/v1/export/download/{filename}"

        logger.info(
            f"Report generated: {filename} ({file_size} bytes)"
        )

        return ReportResponse(
            filename=filename,
            filepath=str(filepath),
            size_bytes=file_size,
            download_url=download_url,
            created_at=datetime.now()
        )

    except ValueError as e:
        logger.error(f"Report generation validation error: {e}")
        raise HTTPException(status_code=404, detail=str(e))

    except Exception as e:
        logger.error(
            f"Report generation failed for search {search_id}: {e}",
            exc_info=True
        )
        raise HTTPException(
            status_code=500,
            detail=f"Report generation failed: {str(e)}"
        )


@router.get("/download/{filename}")
async def download_export(filename: str):
    """
    Download exported file

    Serves exported files from the export directory with proper MIME types.
    Prevents path traversal attacks by validating filename.

    Args:
        filename: Name of file to download

    Returns:
        FileResponse with exported file

    Raises:
        HTTPException 400: Invalid filename (path traversal attempt)
        HTTPException 404: File not found
    """
    logger.info(f"Download requested: {filename}")

    # Security: Prevent path traversal
    if ".." in filename or "/" in filename or "\\" in filename:
        logger.warning(f"Path traversal attempt detected: {filename}")
        raise HTTPException(
            status_code=400,
            detail="Invalid filename"
        )

    # Build file path
    filepath = Path(settings.EXPORT_DIR) / filename

    # Check if file exists
    if not filepath.exists():
        logger.error(f"File not found: {filepath}")
        raise HTTPException(
            status_code=404,
            detail=f"File not found: {filename}"
        )

    # Detect MIME type
    mime_type, _ = mimetypes.guess_type(str(filepath))
    if not mime_type:
        mime_type = "application/octet-stream"

    logger.info(
        f"Serving file: {filename} ({filepath.stat().st_size} bytes, "
        f"mime_type={mime_type})"
    )

    # Return file response
    return FileResponse(
        path=str(filepath),
        media_type=mime_type,
        filename=filename
    )


@router.get("/formats")
async def get_export_formats():
    """
    Get available export formats

    Returns a list of all supported export formats with metadata including
    file extension, MIME type, status, and description.

    Returns:
        Dictionary containing list of format metadata
    """
    return {
        "formats": [
            {
                "name": "CSV",
                "format_code": "csv",
                "extension": ".csv",
                "mime_type": "text/csv",
                "status": "available",
                "description": "Comma-separated values for spreadsheet applications",
                "features": [
                    "Flat structure",
                    "Search metadata columns",
                    "UTF-8 with BOM encoding",
                    "Excel compatible"
                ]
            },
            {
                "name": "Excel",
                "format_code": "excel",
                "extension": ".xlsx",
                "mime_type": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                "status": "available",
                "description": "Microsoft Excel workbook with multiple sheets and formatting",
                "features": [
                    "Multi-sheet workbook",
                    "Professional formatting",
                    "Auto-adjusted columns",
                    "Separate analysis sheets"
                ]
            },
            {
                "name": "JSON",
                "format_code": "json",
                "extension": ".json",
                "mime_type": "application/json",
                "status": "available",
                "description": "JavaScript Object Notation for data interchange",
                "features": [
                    "Full data structure",
                    "Nested objects",
                    "Analysis data included",
                    "Pretty-printed"
                ]
            },
            {
                "name": "Power BI",
                "format_code": "powerbi",
                "extension": ".csv",
                "mime_type": "text/csv",
                "status": "available",
                "description": "Power BI optimized denormalized CSV format",
                "features": [
                    "Denormalized structure",
                    "Clean column names",
                    "Analysis summary fields",
                    "Optimized for BI tools"
                ]
            },
            {
                "name": "HTML Report",
                "format_code": "html",
                "extension": ".html",
                "mime_type": "text/html",
                "status": "available",
                "description": "Professional HTML report with CSS styling",
                "features": [
                    "Styled report layout",
                    "Sentiment color coding",
                    "Top 20 results",
                    "Print-friendly"
                ]
            }
        ],
        "module": "Module 9",
        "version": "0.9.0",
        "note": "All export formats are now fully implemented and available"
    }


@router.delete("/{filename}", response_model=DeleteResponse)
async def delete_export(filename: str):
    """
    Delete exported file

    Removes an exported file from the export directory.
    Includes security checks to prevent path traversal.

    Args:
        filename: Name of file to delete

    Returns:
        Deletion status and message

    Raises:
        HTTPException 400: Invalid filename (path traversal attempt)
        HTTPException 404: File not found
        HTTPException 500: Deletion failed
    """
    logger.info(f"Delete requested: {filename}")

    # Security: Prevent path traversal
    if ".." in filename or "/" in filename or "\\" in filename:
        logger.warning(f"Path traversal attempt detected in delete: {filename}")
        raise HTTPException(
            status_code=400,
            detail="Invalid filename"
        )

    # Build file path
    filepath = Path(settings.EXPORT_DIR) / filename

    # Check if file exists
    if not filepath.exists():
        logger.error(f"File not found for deletion: {filepath}")
        raise HTTPException(
            status_code=404,
            detail=f"File not found: {filename}"
        )

    try:
        # Delete file
        filepath.unlink()

        logger.info(f"File deleted successfully: {filename}")

        return DeleteResponse(
            filename=filename,
            success=True,
            message=f"File {filename} deleted successfully"
        )

    except Exception as e:
        logger.error(f"Failed to delete file {filename}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete file: {str(e)}"
        )


# ============================================================
# Legacy Endpoints (Backwards Compatibility)
# ============================================================

@router.get("/search/{search_id}/csv")
async def export_search_csv_legacy(search_id: int):
    """
    Legacy CSV export endpoint (redirects to main export endpoint)

    Deprecated: Use POST /export/{search_id}?format=csv instead

    Args:
        search_id: Search ID to export

    Returns:
        Redirect information
    """
    logger.warning(
        f"Legacy CSV endpoint called for search {search_id}. "
        "Recommend using POST /export/{search_id}?format=csv"
    )

    return JSONResponse(
        status_code=200,
        content={
            "message": "This endpoint is deprecated",
            "recommended_endpoint": f"POST /api/v1/export/{search_id}?format=csv",
            "note": "Please use the main export endpoint for better functionality"
        }
    )


@router.get("/search/{search_id}/json")
async def export_search_json_legacy(search_id: int):
    """
    Legacy JSON export endpoint (redirects to main export endpoint)

    Deprecated: Use POST /export/{search_id}?format=json instead

    Args:
        search_id: Search ID to export

    Returns:
        Redirect information
    """
    logger.warning(
        f"Legacy JSON endpoint called for search {search_id}. "
        "Recommend using POST /export/{search_id}?format=json"
    )

    return JSONResponse(
        status_code=200,
        content={
            "message": "This endpoint is deprecated",
            "recommended_endpoint": f"POST /api/v1/export/{search_id}?format=json",
            "note": "Please use the main export endpoint for better functionality"
        }
    )


@router.get("/analysis/{search_id}/report")
async def export_analysis_report_legacy(search_id: int):
    """
    Legacy report endpoint (redirects to main report endpoint)

    Deprecated: Use POST /export/{search_id}/report instead

    Args:
        search_id: Search ID to generate report for

    Returns:
        Redirect information
    """
    logger.warning(
        f"Legacy report endpoint called for search {search_id}. "
        "Recommend using POST /export/{search_id}/report"
    )

    return JSONResponse(
        status_code=200,
        content={
            "message": "This endpoint is deprecated",
            "recommended_endpoint": f"POST /api/v1/export/{search_id}/report",
            "note": "Please use the main report endpoint for better functionality"
        }
    )
