"""
Model Monitor module for tracking vLLM model startup and runtime logs.

This module monitors systemd services for vLLM models (llm, embedding, reranker)
and tracks their startup progress and runtime logs.
"""

import logging
import subprocess
import threading
import time
import re
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime
from collections import deque


logger = logging.getLogger(__name__)


class ModelServiceMonitor:
    """Monitor for a single model service."""
    
    # Startup checkpoints for progress tracking
    CHECKPOINTS = [
        (r"Initializing a V1 LLM engine", 10),
        (r"Loading safetensors checkpoint shards:.*100%.*Completed", 25),
        (r"Available KV cache memory", 40),
        (r"Capturing CUDA graphs", 50),
        (r"Graph capturing finished", 75),
        (r"Application startup complete", 100),
    ]
    
    # Patterns to extract model information
    MODEL_NAME_PATTERN = r"'served_model_name':\s*\['([^']+)'\]"
    API_PORT_PATTERN = r"Starting vLLM API server \d+ on http://[\d.:]+:(\d+)"
    
    def __init__(self, service_name: str, model_type: str, max_log_lines: int = 500):
        """
        Initialize model service monitor.
        
        Args:
            service_name: Systemd service name (e.g., 'dev-llm.service')
            model_type: Model type identifier ('llm', 'embedding', 'reranker')
            max_log_lines: Maximum number of log lines to keep in memory
        """
        self.service_name = service_name
        self.model_type = model_type
        self.max_log_lines = max_log_lines
        
        # State
        self._process: Optional[subprocess.Popen] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._lock = threading.Lock()
        
        # Data
        self._progress = 0
        self._status_text = "0%"
        self._model_name = ""
        self._api_port = ""
        self._is_enabled = False
        self._startup_complete = False
        self._log_lines = deque(maxlen=max_log_lines)
        self._last_update_time = 0
        
        # Timer for delayed model info extraction
        self._info_timer: Optional[threading.Timer] = None
        
        # Callback
        self._update_callback: Optional[Callable] = None
    
    def start(self, update_callback: Optional[Callable] = None) -> None:
        """
        Start monitoring the service.
        
        Args:
            update_callback: Callback function called when status updates
        """
        if self._running:
            logger.warning(f"Monitor for {self.service_name} already running")
            return
        
        self._update_callback = update_callback
        self._running = True
        self._thread = threading.Thread(target=self._monitor_logs, daemon=True)
        self._thread.start()
        
        logger.info(f"Started monitoring {self.service_name}")
    
    def stop(self) -> None:
        """Stop monitoring the service."""
        if not self._running:
            return
        
        self._running = False
        
        # Cancel any pending timer
        if self._info_timer:
            self._info_timer.cancel()
            self._info_timer = None
        
        if self._process:
            try:
                self._process.terminate()
                self._process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self._process.kill()
            except Exception as e:
                logger.error(f"Error stopping process for {self.service_name}: {e}")
            finally:
                self._process = None
        
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2)
        
        logger.info(f"Stopped monitoring {self.service_name}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status."""
        with self._lock:
            return {
                "service": self.service_name,
                "model_type": self.model_type,
                "progress": self._progress,
                "status_text": self._status_text,
                "model_name": self._model_name,
                "api_port": self._api_port,
                "is_enabled": self._is_enabled,
                "startup_complete": self._startup_complete,
                "last_update": self._last_update_time
            }
    
    def get_logs(self, lines: int = 100) -> List[str]:
        """Get recent log lines."""
        with self._lock:
            return list(self._log_lines)[-lines:]
    
    def _monitor_logs(self) -> None:
        """Background thread to monitor journalctl logs."""
        import select
        import os
        
        # Set timeout based on model type
        timeout = 10 if self.model_type == "llm" else 5
        max_retries = 3
        retry_count = 0
        
        while self._running and retry_count < max_retries:
            # Wait a bit before checking service
            time.sleep(1 + retry_count * 2)  # Progressive delay
            
            # Check if service is active
            try:
                result = subprocess.run(
                    ["systemctl", "is-active", self.service_name],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode != 0:
                    retry_count += 1
                    logger.info(f"Service {self.service_name} is not active (attempt {retry_count}/{max_retries})")
                    if retry_count >= max_retries:
                        self._update_status(0, f"未启用{self._get_model_display_name()}", False)
                        return
                    continue
            except Exception as e:
                logger.error(f"Failed to check service status for {self.service_name}: {e}")
                retry_count += 1
                if retry_count >= max_retries:
                    self._update_status(0, f"未启用{self._get_model_display_name()}", False)
                    return
                continue
            
            # Service is active, break retry loop
            break
        
        if not self._running:
            return
        
        # Service is active, start following logs
        try:
            # Get more historical lines to ensure we capture startup logs
            # LLM startup can be verbose, so we need more lines
            history_lines = 300 if self.model_type == "llm" else 200
            # Use stdbuf to force line-buffered output from journalctl
            # This prevents log lines from getting stuck in journalctl's buffer
            cmd = ["stdbuf", "-oL", "journalctl", "-u", self.service_name, "-f", "-n", str(history_lines)]
            self._process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            # Make stdout non-blocking for timeout detection
            fd = self._process.stdout.fileno()
            import fcntl
            flags = fcntl.fcntl(fd, fcntl.F_GETFL)
            fcntl.fcntl(fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)
            
            self._update_status(0, "0%", True)
            logger.info(f"Started following logs for {self.service_name}")
            
            last_log_time = time.time()
            has_received_any_log = False
            
            while self._running:
                # Use select to wait for data with timeout
                try:
                    readable, _, _ = select.select([self._process.stdout], [], [], 1.0)
                except (ValueError, OSError):
                    # Process might have been terminated
                    break
                
                if readable:
                    try:
                        line = self._process.stdout.readline()
                        if line:
                            line = line.strip()
                            if line:
                                has_received_any_log = True
                                last_log_time = time.time()
                                
                                # Add to log buffer
                                with self._lock:
                                    timestamp = datetime.now().strftime("%H:%M:%S")
                                    self._log_lines.append(f"[{timestamp}] {line}")
                                
                                # Process line for startup progress
                                if not self._startup_complete:
                                    self._process_startup_line(line)
                        # Note: In non-blocking mode, empty string doesn't mean EOF
                        # We check process status separately below
                    except (IOError, OSError) as e:
                        # No data available (non-blocking read) or read error
                        if hasattr(e, 'errno') and e.errno == 11:  # EAGAIN
                            pass  # Normal for non-blocking, no data available
                        else:
                            logger.debug(f"Read error for {self.service_name}: {e}")
                else:
                    # Timeout on select - check if we've exceeded our timeout threshold
                    if not has_received_any_log:
                        elapsed = time.time() - last_log_time
                        if elapsed > timeout:
                            logger.info(f"No logs received for {self.service_name} within {timeout}s")
                            self._update_status(0, f"未启用{self._get_model_display_name()}", False)
                            return
                
                # Check if process is still running
                if self._process.poll() is not None:
                    logger.warning(f"journalctl process for {self.service_name} terminated")
                    break
        
        except Exception as e:
            logger.error(f"Error monitoring logs for {self.service_name}: {e}")
            self._update_status(0, f"未启用{self._get_model_display_name()}", False)
        finally:
            self._running = False
    
    def _process_startup_line(self, line: str) -> None:
        """Process a log line for startup progress."""
        # Check against checkpoints
        for pattern, progress in self.CHECKPOINTS:
            if re.search(pattern, line, re.IGNORECASE):
                logger.debug(f"{self.service_name}: Checkpoint reached - {progress}%")
                
                # If 100%, mark startup complete BEFORE updating status
                if progress == 100:
                    # Use lock to ensure thread-safe update of startup_complete
                    with self._lock:
                        self._startup_complete = True
                    # Cancel any existing timer
                    if self._info_timer:
                        self._info_timer.cancel()
                    # Wait 5 seconds then extract model info
                    self._info_timer = threading.Timer(5.0, self._extract_model_info)
                    self._info_timer.start()
                
                # Update status (this will trigger callback with correct startup_complete flag)
                self._update_status(progress, f"{progress}%", True)
                break
        
        # Try to extract model name and port immediately when seen
        if not self._model_name:
            name_match = re.search(self.MODEL_NAME_PATTERN, line)
            if name_match:
                with self._lock:
                    self._model_name = name_match.group(1)
                logger.info(f"{self.service_name}: Found model name: {self._model_name}")
        
        if not self._api_port:
            port_match = re.search(self.API_PORT_PATTERN, line)
            if port_match:
                with self._lock:
                    self._api_port = port_match.group(1)
                logger.info(f"{self.service_name}: Found API port: {self._api_port}")
    
    def _extract_model_info(self) -> None:
        """Extract model information from logs after startup complete."""
        # Check if we're still running
        if not self._running:
            return
        
        if self._model_name and self._api_port:
            status_text = f"模型：{self._model_name} | 端口：{self._api_port}"
            self._update_status(100, status_text, True)
            logger.info(f"{self.service_name}: Startup complete - {status_text}")
        else:
            # Keep showing 100% if we couldn't extract info, but still update to trigger callback
            status_text = f"已启动 | 100%"
            self._update_status(100, status_text, True)
            logger.warning(f"{self.service_name}: Could not extract model info, showing generic completion")
    
    def _update_status(self, progress: int, status_text: str, is_enabled: bool) -> None:
        """Update internal status and call callback."""
        with self._lock:
            self._progress = progress
            self._status_text = status_text
            self._is_enabled = is_enabled
            self._last_update_time = time.time()
        
        if self._update_callback:
            try:
                self._update_callback(self.model_type, self.get_status())
            except Exception as e:
                logger.error(f"Error in update callback for {self.service_name}: {e}")
    
    def _get_model_display_name(self) -> str:
        """Get display name for the model type."""
        names = {
            "llm": "主模型",
            "embedding": "嵌入模型",
            "reranker": "重排模型"
        }
        return names.get(self.model_type, "模型")


class ModelMonitor:
    """Main model monitor managing all three model services."""
    
    def __init__(self, max_log_lines: int = 500):
        """
        Initialize model monitor.
        
        Args:
            max_log_lines: Maximum log lines to keep per model
        """
        self.max_log_lines = max_log_lines
        
        # Create monitors for each service
        self.monitors = {
            "llm": ModelServiceMonitor("dev-llm.service", "llm", max_log_lines),
            "embedding": ModelServiceMonitor("dev-embedding.service", "embedding", max_log_lines),
            "reranker": ModelServiceMonitor("dev-reranker.service", "reranker", max_log_lines)
        }
        
        # Callback for status updates
        self._update_callback: Optional[Callable] = None
        self._lock = threading.Lock()
    
    def start(self, update_callback: Optional[Callable] = None) -> None:
        """
        Start monitoring all model services.
        
        Args:
            update_callback: Callback function called when any model status updates
        """
        self._update_callback = update_callback
        
        for monitor in self.monitors.values():
            monitor.start(self._on_model_update)
        
        logger.info("Model monitor started for all services")
    
    def stop(self) -> None:
        """Stop monitoring all model services."""
        for monitor in self.monitors.values():
            monitor.stop()
        
        logger.info("Model monitor stopped for all services")
    
    def get_all_status(self) -> Dict[str, Any]:
        """Get status of all models."""
        with self._lock:
            return {
                "llm": self.monitors["llm"].get_status(),
                "embedding": self.monitors["embedding"].get_status(),
                "reranker": self.monitors["reranker"].get_status(),
                "timestamp": datetime.utcnow().isoformat() + 'Z'
            }
    
    def get_logs(self, model_type: str, lines: int = 100) -> List[str]:
        """Get logs for a specific model."""
        if model_type in self.monitors:
            return self.monitors[model_type].get_logs(lines)
        return []
    
    def _on_model_update(self, model_type: str, status: Dict[str, Any]) -> None:
        """Called when a model status updates."""
        logger.debug(f"Model update: {model_type} - {status['status_text']}")
        
        if self._update_callback:
            try:
                self._update_callback(self.get_all_status())
            except Exception as e:
                logger.error(f"Error in model monitor callback: {e}")
    
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()

