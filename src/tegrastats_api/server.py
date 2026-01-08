"""
Main server module for Tegrastats API.
"""

import logging
import threading
import time
from datetime import datetime
from typing import Dict, Any, Optional

from flask import Flask, jsonify, request
from flask_socketio import SocketIO, emit
from flask_cors import CORS

from .config import Config
from .parser import TegrastatsParser
from .model_monitor import ModelMonitor


logger = logging.getLogger(__name__)


class ConnectionLimiter:
    """Connection limiter for WebSocket connections."""
    
    def __init__(self, max_connections: int = 10):
        self.max_connections = max_connections
        self.current_connections = 0
        self.lock = threading.Lock()
    
    def can_accept(self) -> bool:
        """Check if new connection can be accepted."""
        with self.lock:
            return self.current_connections < self.max_connections
    
    def add_connection(self) -> bool:
        """Add a new connection."""
        with self.lock:
            if self.current_connections < self.max_connections:
                self.current_connections += 1
                return True
            return False
    
    def remove_connection(self) -> None:
        """Remove a connection."""
        with self.lock:
            if self.current_connections > 0:
                self.current_connections -= 1
    
    def get_count(self) -> int:
        """Get current connection count."""
        with self.lock:
            return self.current_connections


class TegrastatsServer:
    """Main Tegrastats API server."""
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize server.
        
        Args:
            config: Server configuration
        """
        self.config = config or Config()
        self.app = Flask(__name__)
        self.app.config['SECRET_KEY'] = 'tegrastats-api-secret'
        
        # Setup CORS
        CORS(self.app, origins=self.config.cors_origins)
        
        # Setup SocketIO
        self.socketio = SocketIO(
            self.app,
            cors_allowed_origins=self.config.cors_origins,
            async_mode='threading',
            allow_unsafe_werkzeug=self.config.allow_unsafe_werkzeug,
            logger=False,
            engineio_logger=False
        )
        
        # Initialize components
        self.parser = TegrastatsParser(interval=self.config.tegrastats_interval)
        self.limiter = ConnectionLimiter(max_connections=self.config.max_connections)
        self.model_monitor = ModelMonitor(max_log_lines=500)
        
        # Setup routes and events
        self._setup_routes()
        self._setup_socketio_events()
        
        # Data update thread
        self._update_thread: Optional[threading.Thread] = None
        self._running = False
        
        logger.info("Tegrastats API服务器已初始化")
    
    def _setup_routes(self) -> None:
        """Setup Flask routes."""
        
        @self.app.route('/api/health', methods=['GET'])
        def health():
            """Health check endpoint."""
            return jsonify({
                'status': 'healthy',
                'service': 'tegrastats-api',
                'timestamp': datetime.utcnow().isoformat() + 'Z',
                'connected_clients': self.limiter.get_count()
            })
        
        @self.app.route('/api/status', methods=['GET'])
        def status():
            """Get complete system status."""
            data = self.parser.get_current_status()
            if not data:
                return jsonify({'error': 'No data available'}), 503
            
            # Add timestamp in ISO format
            data['timestamp'] = datetime.utcnow().isoformat() + 'Z'
            return jsonify(data)
        
        @self.app.route('/api/cpu', methods=['GET'])
        def cpu():
            """Get CPU information."""
            data = self.parser.get_current_status()
            if not data or 'cpu' not in data:
                return jsonify({'error': 'CPU data not available'}), 503
            
            return jsonify({
                'cpu': data['cpu'],
                'timestamp': datetime.utcnow().isoformat() + 'Z'
            })
        
        @self.app.route('/api/memory', methods=['GET'])
        def memory():
            """Get memory information."""
            data = self.parser.get_current_status()
            if not data or 'memory' not in data:
                return jsonify({'error': 'Memory data not available'}), 503
            
            return jsonify({
                'memory': data['memory'],
                'timestamp': datetime.utcnow().isoformat() + 'Z'
            })
        
        @self.app.route('/api/temperature', methods=['GET'])
        def temperature():
            """Get temperature information."""
            data = self.parser.get_current_status()
            if not data or 'temperature' not in data:
                return jsonify({'error': 'Temperature data not available'}), 503
            
            return jsonify({
                'temperature': data['temperature'],
                'timestamp': datetime.utcnow().isoformat() + 'Z'
            })
        
        @self.app.route('/api/power', methods=['GET'])
        def power():
            """Get power information."""
            data = self.parser.get_current_status()
            if not data or 'power' not in data:
                return jsonify({'error': 'Power data not available'}), 503
            
            return jsonify({
                'power': data['power'],
                'timestamp': datetime.utcnow().isoformat() + 'Z'
            })
        
        @self.app.route('/api/model_status', methods=['GET'])
        def model_status():
            """Get model status information."""
            status = self.model_monitor.get_all_status()
            return jsonify(status)
        
        @self.app.route('/api/model_logs/<model_type>', methods=['GET'])
        def model_logs(model_type):
            """Get model logs for a specific model."""
            lines = request.args.get('lines', default=100, type=int)
            lines = min(lines, 1000)  # Cap at 1000 lines
            
            if model_type not in ['llm', 'embedding', 'reranker']:
                return jsonify({'error': 'Invalid model type'}), 400
            
            logs = self.model_monitor.get_logs(model_type, lines)
            return jsonify({
                'model_type': model_type,
                'logs': logs,
                'count': len(logs),
                'timestamp': datetime.utcnow().isoformat() + 'Z'
            })
    
    def _setup_socketio_events(self) -> None:
        """Setup SocketIO event handlers."""
        
        @self.socketio.on('connect')
        def handle_connect():
            """Handle client connection."""
            client_ip = request.environ.get('REMOTE_ADDR', 'unknown')
            
            if not self.limiter.can_accept():
                logger.warning(f"拒绝连接 {client_ip}: 达到最大连接数限制")
                return False
            
            if self.limiter.add_connection():
                logger.info(f"WebSocket客户端连接: {client_ip}, SID: {request.sid}, "
                           f"当前连接数: {self.limiter.get_count()}")
                return True
            else:
                logger.warning(f"无法添加连接: {client_ip}")
                return False
        
        @self.socketio.on('disconnect')
        def handle_disconnect():
            """Handle client disconnection."""
            client_ip = request.environ.get('REMOTE_ADDR', 'unknown')
            self.limiter.remove_connection()
            logger.info(f"WebSocket客户端断开: {client_ip}, SID: {request.sid}, "
                       f"当前连接数: {self.limiter.get_count()}")
    
    def _update_data_thread(self) -> None:
        """Background thread for updating data."""
        logger.info("数据更新线程启动")
        
        while self._running:
            try:
                if self.limiter.get_count() > 0:
                    # Send tegrastats data
                    data = self.parser.get_current_status()
                    if data:
                        # Add timestamp
                        data['timestamp'] = datetime.utcnow().isoformat() + 'Z'
                        
                        # Emit to all connected clients
                        self.socketio.emit('tegrastats_update', data)
                        logger.debug(f"向 {self.limiter.get_count()} 个客户端发送tegrastats数据更新")
                    
                    # Send model status data
                    model_status = self.model_monitor.get_all_status()
                    self.socketio.emit('model_status_update', model_status)
                    logger.debug(f"向 {self.limiter.get_count()} 个客户端发送模型状态更新")
                
                time.sleep(self.config.update_interval)
                
            except Exception as e:
                logger.error(f"数据更新线程错误: {e}")
                time.sleep(1)
        
        logger.info("数据更新线程停止")
    
    def start(self) -> None:
        """Start the server components."""
        try:
            # Start tegrastats parser
            self.parser.start()
            
            # Start model monitor
            self.model_monitor.start()
            
            # Start data update thread
            self._running = True
            self._update_thread = threading.Thread(target=self._update_data_thread, daemon=True)
            self._update_thread.start()
            
            logger.info("服务器组件已启动")
            
        except Exception as e:
            logger.error(f"启动服务器组件失败: {e}")
            self.stop()
            raise
    
    def stop(self) -> None:
        """Stop the server components."""
        logger.info("正在停止服务器...")
        
        # Stop data update thread
        self._running = False
        if self._update_thread and self._update_thread.is_alive():
            self._update_thread.join(timeout=2)
        
        # Stop parser
        self.parser.stop()
        
        # Stop model monitor
        self.model_monitor.stop()
        
        logger.info("服务器已关闭")
    
    def run(self, **kwargs) -> None:
        """
        Run the server.
        
        Args:
            **kwargs: Additional arguments for SocketIO.run()
        """
        # Start components
        self.start()
        
        try:
            # Configure logging
            logging.basicConfig(
                level=getattr(logging, self.config.log_level),
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            
            logger.info(f"启动Tegrastats API服务器")
            logger.info(f"监听地址: {self.config.host}:{self.config.port}")
            logger.info(f"最大连接数: {self.config.max_connections}")
            logger.info(f"数据更新频率: {self.config.update_interval}秒")
            
            # Run server
            run_kwargs = {
                'host': self.config.host,
                'port': self.config.port,
                'debug': self.config.debug,
                **kwargs
            }
            
            # Add allow_unsafe_werkzeug if enabled in config
            if self.config.allow_unsafe_werkzeug:
                run_kwargs['allow_unsafe_werkzeug'] = True
                logger.warning("使用不安全的Werkzeug模式 (仅用于开发环境)")
            
            self.socketio.run(self.app, **run_kwargs)
            
        except KeyboardInterrupt:
            logger.info("收到中断信号")
        except Exception as e:
            logger.error(f"服务器运行错误: {e}")
            raise
        finally:
            self.stop()
    
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()