"""Configuration management for the AI Stock Discovery Engine."""

import os
from typing import Dict, Any
import logging

class Config:
    """Centralized configuration management for API toggles and settings."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._load_config()
    
    def _load_config(self):
        """Load configuration from environment variables."""
        self.ENABLE_TWITTER_API = self._get_bool_env("ENABLE_TWITTER_API", True)
        self.ENABLE_POLYMARKET_API = self._get_bool_env("ENABLE_POLYMARKET_API", True)
        self.ENABLE_NEWS_API = self._get_bool_env("ENABLE_NEWS_API", True)
        self.ENABLE_POLYGON_API = self._get_bool_env("ENABLE_POLYGON_API", True)
        self.ENABLE_YFINANCE_API = self._get_bool_env("ENABLE_YFINANCE_API", True)
        self.ENABLE_OPENAI_API = self._get_bool_env("ENABLE_OPENAI_API", True)
        self.PRIORITIZE_POLYGON = self._get_bool_env("PRIORITIZE_POLYGON", True)
        
        self.logger.info(f"Configuration loaded - Twitter: {self.ENABLE_TWITTER_API}, "
                        f"Polymarket: {self.ENABLE_POLYMARKET_API}, "
                        f"News: {self.ENABLE_NEWS_API}, "
                        f"Polygon: {self.ENABLE_POLYGON_API}, "
                        f"YFinance: {self.ENABLE_YFINANCE_API}, "
                        f"OpenAI: {self.ENABLE_OPENAI_API}, "
                        f"Prioritize Polygon: {self.PRIORITIZE_POLYGON}")
    
    def _get_bool_env(self, key: str, default: bool = False) -> bool:
        """Get boolean value from environment variable."""
        value = os.getenv(key, str(default)).lower()
        return value in ('true', '1', 'yes', 'on')
    
    def is_enabled(self, service: str) -> bool:
        """Check if a service is enabled."""
        service_map = {
            'twitter': self.ENABLE_TWITTER_API,
            'polymarket': self.ENABLE_POLYMARKET_API,
            'news': self.ENABLE_NEWS_API,
            'polygon': self.ENABLE_POLYGON_API,
            'yfinance': self.ENABLE_YFINANCE_API,
            'openai': self.ENABLE_OPENAI_API
        }
        return service_map.get(service.lower(), False)
    
    def get_enabled_services(self) -> Dict[str, bool]:
        """Get dictionary of all enabled services."""
        return {
            'twitter': self.ENABLE_TWITTER_API,
            'polymarket': self.ENABLE_POLYMARKET_API,
            'news': self.ENABLE_NEWS_API,
            'polygon': self.ENABLE_POLYGON_API,
            'yfinance': self.ENABLE_YFINANCE_API,
            'openai': self.ENABLE_OPENAI_API
        }
    
    def validate_configuration(self) -> bool:
        """Validate that at least one data source is enabled."""
        data_sources = [
            self.ENABLE_TWITTER_API,
            self.ENABLE_POLYMARKET_API,
            self.ENABLE_NEWS_API,
            (self.ENABLE_POLYGON_API or self.ENABLE_YFINANCE_API)
        ]
        
        if not any(data_sources):
            self.logger.error("No data sources are enabled! At least one must be enabled.")
            return False
        
        return True

config = Config()
