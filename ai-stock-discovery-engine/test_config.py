#!/usr/bin/env python3
"""Test script to verify the configuration system works correctly."""

import os
import sys
from src.utils.config import config

def test_configuration():
    """Test the configuration system."""
    print("Testing AI Stock Discovery Engine Configuration System")
    print("=" * 60)
    
    print(f"Configuration loaded successfully: {config is not None}")
    
    enabled_services = config.get_enabled_services()
    print(f"\nEnabled services: {enabled_services}")
    
    services = ['twitter', 'polymarket', 'news', 'polygon', 'yfinance', 'openai']
    for service in services:
        enabled = config.is_enabled(service)
        print(f"{service.capitalize()} API: {'ENABLED' if enabled else 'DISABLED'}")
    
    is_valid = config.validate_configuration()
    print(f"\nConfiguration is valid: {is_valid}")
    
    print(f"Prioritize Polygon.io: {config.PRIORITIZE_POLYGON}")
    
    print("\n" + "=" * 60)
    print("Configuration test completed successfully!")

if __name__ == "__main__":
    try:
        test_configuration()
    except Exception as e:
        print(f"Configuration test failed: {str(e)}")
        sys.exit(1)
