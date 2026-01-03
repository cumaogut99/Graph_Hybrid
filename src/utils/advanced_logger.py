#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gelişmiş Loglama ve Monitoring Sistemi
=====================================

Bu modül, Time Graph uygulaması için kapsamlı loglama,
monitoring ve debugging sistemi sağlar.
"""

import os
import sys
import logging
import json
import time
import traceback
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from functools import wraps
import threading
from pathlib import Path
import gzip
import shutil

# Configure logging - Production mode: sadece WARNING ve üstü
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LogLevel:
    """Log seviyeleri."""
    TRACE = 5
    DEBUG = 10
    INFO = 20
    WARNING = 30
    ERROR = 40
    CRITICAL = 50

class PerformanceMonitor:
    """Performans monitoring sınıfı."""
    
    def __init__(self):
        self.metrics = {}
        self.start_times = {}
        
    def start_timer(self, operation: str):
        """Timer başlat."""
        self.start_times[operation] = time.time()
    
    def end_timer(self, operation: str) -> float:
        """Timer bitir ve süreyi döndür."""
        if operation in self.start_times:
            duration = time.time() - self.start_times[operation]
            
            if operation not in self.metrics:
                self.metrics[operation] = []
            
            self.metrics[operation].append(duration)
            del self.start_times[operation]
            return duration
        return 0.0
    
    def get_stats(self, operation: str) -> Dict[str, float]:
        """Operasyon istatistiklerini al."""
        if operation not in self.metrics:
            return {}
        
        times = self.metrics[operation]
        return {
            'count': len(times),
            'total': sum(times),
            'average': sum(times) / len(times),
            'min': min(times),
            'max': max(times),
            'last': times[-1] if times else 0
        }

class AdvancedLogger:
    """
    A comprehensive logger with structured logging, performance tracking, 
    and data operation logging capabilities.
    """
    
    def __init__(self, name: str, log_dir: str = "logs"):
        self.name = name
        self.log_dir = log_dir
        self._setup_logger()

    def _setup_logger(self):
        """Set up file and stream handlers for different log levels."""
        # This is a simplified setup. In a real application, you would
        # use rotating file handlers, different formatters, etc.
        pass

    def log(self, level: int, message: str, category: str, context: Optional[Dict[str, Any]] = None):
        """
        Log a message with structured context.
        
        Args:
            level: Logging level (e.g., logging.INFO)
            message: Main log message
            category: High-level category (e.g., 'startup', 'data_processing')
            context: Additional structured data
        """
        log_record = {
            'message': message,
            'category': category,
            'context': context or {}
        }
        # In a real implementation, you would format this into a JSON string
        # or another structured format.
        logger.log(level, f"[{category.upper()}] {message} - Context: {context}")

    def info(self, message: str, category: str, context: Optional[Dict[str, Any]] = None):
        self.log(logging.INFO, message, category, context)

    def warning(self, message: str, category: str, context: Optional[Dict[str, Any]] = None):
        self.log(logging.WARNING, message, category, context)

    def error(self, message: str, category: str, context: Optional[Dict[str, Any]] = None, exc_info=False):
        # The actual logger will handle exc_info correctly.
        # This is just for the method signature.
        self.log(logging.ERROR, message, category, context)

    def log_data_operation(self, operation: str, details: Dict[str, Any]):
        """
        Log a specific data operation.
        
        Args:
            operation: Name of the operation (e.g., 'file_load', 'filter_apply')
            details: Details about the operation
        """
        # Example of checking for DataFrame without importing polars
        if 'dataframe' in details:
            df = details['dataframe']
            df_details = {
                'type': str(type(df)),
                'shape': getattr(df, 'shape', 'N/A'),
                'columns': getattr(df, 'columns', [])
            }
            details['dataframe_details'] = df_details
            del details['dataframe']  # Avoid logging the whole DataFrame
            
        self.info(f"Data operation: {operation}", "data_operations", details)

    def log_session_start(self):
        """Session başlangıç logu."""
        session_info = {
            'session_id': self.session_id,
            'start_time': datetime.now().isoformat(),
            'python_version': sys.version,
            'platform': sys.platform,
            'working_directory': os.getcwd(),
            'command_line': ' '.join(sys.argv)
        }
        
        self.app_logger.info(f"=== SESSION START ===")
        self.app_logger.info(f"Session Info: {json.dumps(session_info, indent=2)}")
    
    def log_session_end(self):
        """Session bitiş logu."""
        session_summary = {
            'session_id': self.session_id,
            'end_time': datetime.now().isoformat(),
            'performance_summary': self._get_performance_summary()
        }
        
        self.app_logger.info(f"Session Summary: {json.dumps(session_summary, indent=2)}")
        self.app_logger.info(f"=== SESSION END ===")
    
    def _get_performance_summary(self) -> Dict[str, Any]:
        """Performans özetini al."""
        summary = {}
        for operation, stats in self.performance_monitor.metrics.items():
            if stats:
                summary[operation] = {
                    'total_calls': len(stats),
                    'total_time': sum(stats),
                    'avg_time': sum(stats) / len(stats),
                    'max_time': max(stats)
                }
        return summary
    
    def log_error(self, error: Exception, context: Dict[str, Any] = None, 
                  category: str = "general"):
        """Hata logla."""
        error_info = {
            'error_type': type(error).__name__,
            'error_message': str(error),
            'category': category,
            'context': context or {},
            'traceback': traceback.format_exc(),
            'timestamp': datetime.now().isoformat()
        }
        
        self.error_logger.error(f"ERROR [{category}]: {json.dumps(error_info, indent=2)}")
    
    def log_performance(self, operation: str, duration: float, details: Dict[str, Any] = None):
        """Performans metriği logla."""
        perf_entry = {
            'operation': operation,
            'duration': duration,
            'timestamp': datetime.now().isoformat(),
            'details': details or {}
        }
        
        self.perf_logger.info(f"PERF: {json.dumps(perf_entry)}")
    
    def debug(self, message: str, context: Dict[str, Any] = None):
        """Debug mesajı logla."""
        if context:
            message = f"{message} | Context: {json.dumps(context)}"
        self.debug_logger.debug(message)
    
    def get_recent_logs(self, logger_name: str = "app", lines: int = 100) -> List[str]:
        """Son log satırlarını al."""
        log_file = self.log_dir / f"{logger_name}.log" if logger_name == "app" else self.log_dir / f"{logger_name}.log"
        
        if not log_file.exists():
            return []
        
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                return f.readlines()[-lines:]
        except Exception as e:
            self.error_logger.error(f"Failed to read recent logs: {e}")
            return []
    
    def search_logs(self, pattern: str, logger_name: str = "app", 
                   days_back: int = 7) -> List[str]:
        """Log'larda arama yap."""
        results = []
        log_file = self.log_dir / f"{logger_name}.log"
        
        if not log_file.exists():
            return results
        
        try:
            cutoff_date = datetime.now() - timedelta(days=days_back)
            
            with open(log_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if pattern.lower() in line.lower():
                        # Tarih kontrolü (basit)
                        try:
                            line_date_str = line.split(' - ')[0]
                            line_date = datetime.strptime(line_date_str, '%Y-%m-%d %H:%M:%S,%f')
                            if line_date >= cutoff_date:
                                results.append(line.strip())
                        except:
                            # Tarih parse edilemezse yine de ekle
                            results.append(line.strip())
            
        except Exception as e:
            self.error_logger.error(f"Log search failed: {e}")
        
        return results
    
    def generate_debug_report(self) -> Dict[str, Any]:
        """Debug raporu oluştur."""
        report = {
            'session_id': self.session_id,
            'report_time': datetime.now().isoformat(),
            'log_files': {},
            'performance_metrics': self._get_performance_summary(),
            'recent_errors': self.get_recent_logs('error', 50),
            'system_info': {
                'python_version': sys.version,
                'platform': sys.platform,
                'working_directory': os.getcwd()
            }
        }
        
        # Log dosyası bilgileri
        for log_file in self.log_dir.glob("*.log"):
            try:
                stat = log_file.stat()
                report['log_files'][log_file.name] = {
                    'size_bytes': stat.st_size,
                    'modified': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    'lines': sum(1 for _ in open(log_file, 'r', encoding='utf-8'))
                }
            except Exception as e:
                report['log_files'][log_file.name] = {'error': str(e)}
        
        return report
    
    def export_debug_report(self, file_path: str = None) -> str:
        """Debug raporunu dosyaya aktar."""
        if not file_path:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            file_path = self.log_dir / f"debug_report_{timestamp}.json"
        
        report = self.generate_debug_report()
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            
            self.info(f"Debug report exported to: {file_path}")
            return str(file_path)
            
        except Exception as e:
            self.error_logger.error(f"Failed to export debug report: {e}")
            return ""

# Decorator fonksiyonları
def log_performance(logger: AdvancedLogger, operation_name: str = None):
    """Performans loglama decorator'ı."""
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            op_name = operation_name or f"{func.__module__}.{func.__name__}"
            
            logger.performance_monitor.start_timer(op_name)
            
            try:
                result = func(*args, **kwargs)
                duration = logger.performance_monitor.end_timer(op_name)
                
                logger.log_performance(op_name, duration, {
                    'success': True,
                    'args_count': len(args),
                    'kwargs_count': len(kwargs)
                })
                
                return result
                
            except Exception as e:
                duration = logger.performance_monitor.end_timer(op_name)
                
                logger.log_performance(op_name, duration, {
                    'success': False,
                    'error': str(e),
                    'args_count': len(args),
                    'kwargs_count': len(kwargs)
                })
                
                raise
        
        return wrapper
    return decorator

def log_function_calls(logger: AdvancedLogger, level: str = "debug"):
    """Fonksiyon çağrılarını loglayan decorator."""
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            func_name = f"{func.__module__}.{func.__name__}"
            
            # Giriş logu
            if level == "debug":
                logger.debug(f"ENTER: {func_name}", {
                    'args_count': len(args),
                    'kwargs': list(kwargs.keys())
                })
            
            try:
                result = func(*args, **kwargs)
                
                # Çıkış logu
                if level == "debug":
                    logger.debug(f"EXIT: {func_name}", {
                        'success': True,
                        'result_type': type(result).__name__
                    })
                
                return result
                
            except Exception as e:
                # Hata logu
                logger.log_error(e, {
                    'function': func_name,
                    'args_count': len(args),
                    'kwargs': list(kwargs.keys())
                }, category="function_call")
                
                raise
        
        return wrapper
    return decorator

def log_data_operations(logger: AdvancedLogger):
    """Veri işleme operasyonlarını loglayan decorator."""
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            func_name = f"{func.__module__}.{func.__name__}"
            
            # DataFrame bilgilerini topla
            data_info = {}
            for i, arg in enumerate(args):
                if hasattr(arg, 'shape'):  # pandas DataFrame/Series
                    data_info[f'arg_{i}_shape'] = arg.shape
                    data_info[f'arg_{i}_type'] = type(arg).__name__
            
            logger.log_data_operation(f"START_{func_name}", data_info)
            
            try:
                result = func(*args, **kwargs)
                
                # Sonuç bilgilerini topla
                result_info = {}
                if hasattr(result, 'shape'):
                    result_info['result_shape'] = result.shape
                    result_info['result_type'] = type(result).__name__
                
                logger.log_data_operation(f"END_{func_name}", {
                    **data_info,
                    **result_info,
                    'success': True
                })
                
                return result
                
            except Exception as e:
                logger.log_data_operation(f"ERROR_{func_name}", {
                    **data_info,
                    'error': str(e),
                    'success': False
                })
                
                raise
        
        return wrapper
    return decorator

# Global logger instance
_global_logger = None

def get_logger() -> AdvancedLogger:
    """Global logger instance'ını al."""
    global _global_logger
    if _global_logger is None:
        _global_logger = AdvancedLogger()
    return _global_logger

def setup_global_logger(app_name: str = "TimeGraph", log_dir: str = "logs"):
    """Global logger'ı kur."""
    global _global_logger
    _global_logger = AdvancedLogger(app_name, log_dir)
    return _global_logger
