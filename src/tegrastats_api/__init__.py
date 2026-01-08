"""
Tegrastats API - HTTP API and WebSocket server for NVIDIA Jetson tegrastats monitoring.

This package provides a Flask-based HTTP API and WebSocket server for accessing
tegrastats data from NVIDIA Jetson devices. Designed for integration with
embedded devices like ESP32S3.

Example:
    Basic usage as a library:
    
    >>> from tegrastats_api import TegrastatsServer
    >>> server = TegrastatsServer(host='0.0.0.0', port=58090)
    >>> server.run()
    
    Command line usage:
    
    $ tegrastats-api --host 0.0.0.0 --port 58090
"""

__version__ = "1.0.0"
__author__ = "Tegrastats API Team"
__email__ = "contact@example.com"
__license__ = "MIT"

from .server import TegrastatsServer, ConnectionLimiter
from .parser import TegrastatsParser
from .config import Config
from .model_monitor import ModelMonitor
from .cli import main as cli_main

__all__ = [
    "TegrastatsServer",
    "TegrastatsParser", 
    "Config",
    "ConnectionLimiter",
    "ModelMonitor",
    "cli_main",
    "__version__",
]