"""
Configuration utilities for CIAP
Additional configuration management and validation functions
"""

from pathlib import Path
from typing import Dict, Any, List
import json
import logging

logger = logging.getLogger(__name__)


class ConfigManager:
    """Additional configuration management utilities"""

    @staticmethod
    def load_prompts() -> Dict[str, str]:
        """
        Load all prompt templates from prompts directory

        Returns:
            Dictionary mapping prompt names to their content
        """
        from .config import settings

        prompts = {}
        prompts_dir = Path(settings.PROMPTS_DIR)

        if prompts_dir.exists():
            for prompt_file in prompts_dir.glob("*.txt"):
                prompt_name = prompt_file.stem
                try:
                    prompts[prompt_name] = prompt_file.read_text(encoding="utf-8")
                    logger.debug(f"Loaded prompt template: {prompt_name}")
                except Exception as e:
                    logger.error(f"Failed to load prompt {prompt_name}: {e}")
        else:
            logger.warning(f"Prompts directory not found: {prompts_dir}")

        return prompts

    @staticmethod
    def load_user_agents(file_path: str = "config/user_agents.txt") -> List[str]:
        """
        Load user agents from file

        Args:
            file_path: Path to user agents file

        Returns:
            List of user agent strings, or default list if file not found
        """
        from .config import settings

        path = Path(file_path)
        if path.exists():
            try:
                agents = [
                    line.strip()
                    for line in path.read_text(encoding="utf-8").splitlines()
                    if line.strip() and not line.startswith("#")
                ]
                logger.info(f"Loaded {len(agents)} user agents from {file_path}")
                return agents
            except Exception as e:
                logger.error(f"Failed to load user agents from {file_path}: {e}")

        logger.debug("Using default user agents from settings")
        return settings.SCRAPER_USER_AGENTS

    @staticmethod
    def validate_environment() -> List[str]:
        """
        Validate environment setup and connectivity

        Returns:
            List of issues found (empty list if all checks pass)
        """
        from .config import settings
        import requests

        issues = []

        # Check directories
        checks = {
            "database_dir": Path(settings.DATABASE_URL.replace("sqlite:///", "")).parent,
            "log_dir": Path(settings.LOG_FILE).parent,
            "export_dir": Path(settings.EXPORT_DIR),
            "data_dir": Path(settings.DATA_DIR),
            "prompts_dir": Path(settings.PROMPTS_DIR)
        }

        for name, path in checks.items():
            if not path.exists():
                try:
                    path.mkdir(parents=True, exist_ok=True)
                    logger.info(f"Created {name}: {path}")
                except Exception as e:
                    issue = f"Failed to create {name} at {path}: {e}"
                    issues.append(issue)
                    logger.error(issue)
            else:
                logger.debug(f"Verified {name} exists: {path}")

        # Check Ollama connection
        try:
            response = requests.get(
                f"{settings.OLLAMA_URL}/api/tags",
                timeout=5
            )
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [m.get("name", "") for m in models]

                if settings.OLLAMA_MODEL in model_names:
                    logger.info(f"Ollama connected with model {settings.OLLAMA_MODEL}")
                else:
                    issue = f"Ollama model {settings.OLLAMA_MODEL} not found. Available: {model_names}"
                    issues.append(issue)
                    logger.warning(issue)
            else:
                issue = f"Ollama returned status {response.status_code}"
                issues.append(issue)
                logger.warning(issue)
        except requests.exceptions.ConnectionError:
            issue = "Cannot connect to Ollama - is it running?"
            issues.append(issue)
            logger.warning(issue)
        except Exception as e:
            issue = f"Ollama connection check failed: {e}"
            issues.append(issue)
            logger.error(issue)

        # Validate database file
        db_path = Path(settings.DATABASE_URL.replace("sqlite:///", ""))
        if db_path.exists():
            logger.info(f"Database file exists: {db_path} ({db_path.stat().st_size / 1024:.2f} KB)")
        else:
            logger.info(f"Database file will be created at: {db_path}")

        return issues

    @staticmethod
    def export_config(output_file: str = "config_export.json") -> Path:
        """
        Export current configuration to file

        Args:
            output_file: Output file path (JSON or YAML)

        Returns:
            Path to exported config file
        """
        from .config import settings

        config_dict = settings.to_dict(exclude_secrets=True)
        output_path = Path(output_file)

        try:
            if output_file.endswith(".yaml") or output_file.endswith(".yml"):
                # Try to import yaml
                try:
                    import yaml
                    output_path.write_text(
                        yaml.dump(config_dict, default_flow_style=False, sort_keys=False),
                        encoding="utf-8"
                    )
                    logger.info(f"Exported config to YAML: {output_path}")
                except ImportError:
                    logger.warning("PyYAML not installed, falling back to JSON")
                    output_path = Path(output_file.replace(".yaml", ".json").replace(".yml", ".json"))
                    output_path.write_text(json.dumps(config_dict, indent=2), encoding="utf-8")
            else:
                output_path.write_text(json.dumps(config_dict, indent=2), encoding="utf-8")
                logger.info(f"Exported config to JSON: {output_path}")
        except Exception as e:
            logger.error(f"Failed to export config: {e}")
            raise

        return output_path

    @staticmethod
    def get_runtime_info() -> Dict[str, Any]:
        """
        Get runtime configuration and system information

        Returns:
            Dictionary with runtime information
        """
        from .config import settings
        import platform
        import sys

        info = {
            "python_version": sys.version,
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "platform_system": platform.system(),
            "platform_release": platform.release(),
            "environment": settings.ENVIRONMENT.value,
            "database": settings.DATABASE_URL,
            "database_async": settings.get_database_url_async(),
            "ollama_url": settings.OLLAMA_URL,
            "ollama_model": settings.OLLAMA_MODEL,
            "api_host": settings.API_HOST,
            "api_port": settings.API_PORT,
            "api_prefix": settings.API_PREFIX,
            "log_level": settings.LOG_LEVEL.value,
            "log_file": settings.LOG_FILE,
            "workers": settings.TASK_QUEUE_MAX_WORKERS,
            "cache_ttl": settings.CACHE_TTL_SECONDS,
            "data_dir": settings.DATA_DIR,
            "export_dir": settings.EXPORT_DIR,
            "prompts_dir": settings.PROMPTS_DIR,
        }

        return info

    @staticmethod
    def print_config_summary():
        """Print a formatted summary of current configuration"""
        from .config import settings

        print("\n" + "="*60)
        print("CIAP Configuration Summary")
        print("="*60)

        print(f"\n🔧 Environment: {settings.ENVIRONMENT.value}")
        print(f"📊 Database: {settings.DATABASE_URL}")
        print(f"🤖 Ollama: {settings.OLLAMA_URL} ({settings.OLLAMA_MODEL})")
        print(f"🌐 API: {settings.API_HOST}:{settings.API_PORT}{settings.API_PREFIX}")
        print(f"📝 Log Level: {settings.LOG_LEVEL.value}")
        print(f"👷 Workers: {settings.TASK_QUEUE_MAX_WORKERS}")
        print(f"⏱️  Cache TTL: {settings.CACHE_TTL_SECONDS}s")

        print(f"\n📁 Directories:")
        print(f"  - Data: {settings.DATA_DIR}")
        print(f"  - Logs: {Path(settings.LOG_FILE).parent}")
        print(f"  - Exports: {settings.EXPORT_DIR}")
        print(f"  - Prompts: {settings.PROMPTS_DIR}")

        print(f"\n🔍 Scraper Settings:")
        print(f"  - Timeout: {settings.SCRAPER_TIMEOUT}s")
        print(f"  - Retries: {settings.SCRAPER_RETRY_COUNT}")
        print(f"  - Rate Limit Delay: {settings.SCRAPER_RATE_LIMIT_DELAY}s")
        print(f"  - Google Max Results: {settings.GOOGLE_MAX_RESULTS}")
        print(f"  - Bing Max Results: {settings.BING_MAX_RESULTS}")

        print("\n" + "="*60 + "\n")


def validate_and_report() -> bool:
    """
    Validate environment and print report

    Returns:
        True if validation passed, False otherwise
    """
    print("\n🔍 Validating environment...")

    issues = ConfigManager.validate_environment()

    if not issues:
        print("✅ All environment checks passed!")
        return True
    else:
        print(f"\n⚠️  Found {len(issues)} issue(s):")
        for issue in issues:
            print(f"  - {issue}")
        return False


if __name__ == "__main__":
    # Standalone testing
    ConfigManager.print_config_summary()
    validate_and_report()

    # Test runtime info
    print("\n📋 Runtime Information:")
    info = ConfigManager.get_runtime_info()
    for key, value in info.items():
        print(f"  {key}: {value}")
