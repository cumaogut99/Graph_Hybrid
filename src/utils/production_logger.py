#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Production Logger - Ticari Kullanım İçin Profesyonel Log Sistemi
================================================================

Bu modül, production ortamı için optimize edilmiş, performanslı bir log sistemi sağlar.
Debug logları otomatik olarak devre dışı bırakılır ve sadece önemli olaylar loglanır.
"""

import logging
import os
import sys
from datetime import datetime
from typing import Optional
from logging.handlers import RotatingFileHandler


class ProductionLogger:
    """
    Ticari kullanım için optimize edilmiş logger.
    
    Özellikler:
    - Production modunda debug logları devre dışı
    - Performans optimizasyonu (gereksiz log çağrıları atlanır)
    - Otomatik log rotasyonu
    - Kullanıcı dostu hata mesajları
    """
    
    def __init__(self, name: str, log_dir: str = "logs", production_mode: bool = True):
        """
        Args:
            name: Logger adı
            log_dir: Log dosyalarının saklanacağı dizin
            production_mode: True ise DEBUG logları devre dışı
        """
        self.name = name
        self.production_mode = production_mode
        self.logger = logging.getLogger(name)
        
        # Production modda sadece WARNING ve üstü (INFO logları devre dışı)
        # Development modda DEBUG ve üstü
        log_level = logging.WARNING if production_mode else logging.DEBUG
        self.logger.setLevel(log_level)
        
        # Mevcut handler'ları temizle
        self.logger.handlers.clear()
        
        # Log dizinini oluştur
        os.makedirs(log_dir, exist_ok=True)
        
        # 1. Rotating File Handler - Otomatik log rotasyonu
        log_file = os.path.join(log_dir, f"{name}.log")
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=5 * 1024 * 1024,  # 5 MB
            backupCount=3,  # Son 3 log dosyasını sakla
            encoding='utf-8'
        )
        file_handler.setLevel(log_level)
        
        # Production-friendly format - Daha temiz
        if production_mode:
            file_formatter = logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
        else:
            # Development modda daha detaylı
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
        
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)
        
        # 2. Console Handler - Sadece WARNING ve üstü
        if not production_mode:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(logging.DEBUG)
            console_formatter = logging.Formatter(
                '%(levelname)s - %(message)s'
            )
            console_handler.setFormatter(console_formatter)
            self.logger.addHandler(console_handler)
    
    def debug(self, message: str, **kwargs):
        """Debug seviyesinde log (sadece development modda)"""
        if not self.production_mode:
            self.logger.debug(message, **kwargs)
    
    def info(self, message: str, **kwargs):
        """Bilgilendirme logu"""
        self.logger.info(message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Uyarı logu"""
        self.logger.warning(message, **kwargs)
    
    def error(self, message: str, exc_info: bool = False, **kwargs):
        """Hata logu"""
        self.logger.error(message, exc_info=exc_info, **kwargs)
    
    def critical(self, message: str, exc_info: bool = True, **kwargs):
        """Kritik hata logu"""
        self.logger.critical(message, exc_info=exc_info, **kwargs)
    
    def log_operation_start(self, operation_name: str, details: Optional[str] = None):
        """İşlem başlangıcını logla"""
        msg = f"[{operation_name}] Başlatıldı"
        if details:
            msg += f" - {details}"
        self.info(msg)
    
    def log_operation_end(self, operation_name: str, success: bool = True, duration_ms: Optional[float] = None):
        """İşlem sonucunu logla"""
        status = "Tamamlandı" if success else "Başarısız"
        msg = f"[{operation_name}] {status}"
        if duration_ms is not None:
            msg += f" ({duration_ms:.1f}ms)"
        
        if success:
            self.info(msg)
        else:
            self.error(msg)
    
    def log_performance_metric(self, metric_name: str, value: float, unit: str = "ms"):
        """Performans metriğini logla"""
        self.info(f"[PERFORMANCE] {metric_name}: {value:.2f}{unit}")
    
    def set_production_mode(self, enabled: bool):
        """Production modunu aktif/inaktif et"""
        self.production_mode = enabled
        log_level = logging.WARNING if enabled else logging.DEBUG
        self.logger.setLevel(log_level)
        for handler in self.logger.handlers:
            handler.setLevel(log_level)


# Global logger instance - Tüm uygulama tarafından kullanılabilir
_global_logger: Optional[ProductionLogger] = None


def get_logger(name: str = "time_graph_app", production_mode: bool = True) -> ProductionLogger:
    """
    Global logger instance'ını al veya oluştur.
    
    Args:
        name: Logger adı
        production_mode: Production modu aktif mi?
    
    Returns:
        ProductionLogger instance
    """
    global _global_logger
    
    if _global_logger is None:
        _global_logger = ProductionLogger(name, production_mode=production_mode)
    
    return _global_logger


def configure_production_logging(production_mode: bool = True):
    """
    Uygulama genelinde production logging'i yapılandır.
    
    Args:
        production_mode: True ise DEBUG logları kapalı
    """
    logger = get_logger(production_mode=production_mode)
    
    # Diğer tüm logger'ları da production mode'a al
    logging.root.setLevel(logging.WARNING if production_mode else logging.DEBUG)
    
    # PyQtGraph gibi kütüphanelerin verbose loglarını sustur
    if production_mode:
        logging.getLogger('PyQt5').setLevel(logging.WARNING)
        logging.getLogger('matplotlib').setLevel(logging.WARNING)
        logging.getLogger('polars').setLevel(logging.WARNING)
    
    return logger


# Convenience functions
def log_info(message: str):
    """Hızlı info log"""
    get_logger().info(message)


def log_error(message: str, exc_info: bool = False):
    """Hızlı error log"""
    get_logger().error(message, exc_info=exc_info)


def log_warning(message: str):
    """Hızlı warning log"""
    get_logger().warning(message)

