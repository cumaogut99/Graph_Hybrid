#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gelişmiş Hata Yönetim Sistemi
============================

Bu modül, Time Graph uygulaması için kapsamlı hata yakalama,
işleme ve raporlama sistemi sağlar.
"""

import sys
import os
import logging
import traceback
import json
from datetime import datetime
from typing import Dict, Any, Optional, Callable
from enum import Enum
import polars as pl
import numpy as np
from PyQt5.QtWidgets import QMessageBox, QWidget
from PyQt5.QtCore import QObject, pyqtSignal

class ErrorSeverity(Enum):
    """Hata şiddet seviyeleri."""
    LOW = "low"           # Uyarı seviyesi, uygulama çalışmaya devam eder
    MEDIUM = "medium"     # Orta seviye, bazı özellikler etkilenebilir
    HIGH = "high"         # Yüksek seviye, önemli işlevler etkilenir
    CRITICAL = "critical" # Kritik, uygulama durabilir

class ErrorCategory(Enum):
    """Hata kategorileri."""
    DATA_FORMAT = "data_format"
    FILE_IO = "file_io"
    MEMORY = "memory"
    VALIDATION = "validation"
    IMPORT = "import"
    NETWORK = "network"
    UI = "ui"
    UNKNOWN = "unknown"

class ErrorHandler(QObject):
    """Merkezi hata yönetim sistemi."""
    
    # Sinyaller
    error_occurred = pyqtSignal(str, str, str)  # category, message, details
    recovery_attempted = pyqtSignal(str, bool)  # error_id, success
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_widget = parent
        self.error_log = []
        self.recovery_strategies = {}
        self.error_stats = {}
        
        # Hata recovery stratejilerini kaydet
        self._register_recovery_strategies()
        
        # Detaylı logger kurulumu
        self._setup_detailed_logging()
    
    def _setup_detailed_logging(self):
        """Detaylı logging sistemi kurulumu."""
        # Ana logger
        self.logger = logging.getLogger('ErrorHandler')
        self.logger.setLevel(logging.DEBUG)
        
        # Hata dosyası handler
        error_handler = logging.FileHandler('errors.log', encoding='utf-8')
        error_handler.setLevel(logging.ERROR)
        
        # Debug dosyası handler
        debug_handler = logging.FileHandler('debug.log', encoding='utf-8')
        debug_handler.setLevel(logging.DEBUG)
        
        # Formatter
        detailed_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        error_handler.setFormatter(detailed_formatter)
        debug_handler.setFormatter(detailed_formatter)
        
        self.logger.addHandler(error_handler)
        self.logger.addHandler(debug_handler)
    
    def handle_error(self, 
                    error: Exception, 
                    category: ErrorCategory = ErrorCategory.UNKNOWN,
                    severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                    context: Dict[str, Any] = None,
                    user_message: str = None,
                    auto_recover: bool = True) -> bool:
        """
        Merkezi hata işleme fonksiyonu.
        
        Args:
            error: Yakalanan hata
            category: Hata kategorisi
            severity: Hata şiddeti
            context: Hata bağlamı (dosya yolu, veri bilgisi vb.)
            user_message: Kullanıcıya gösterilecek özel mesaj
            auto_recover: Otomatik recovery denenmeli mi
            
        Returns:
            bool: Recovery başarılı oldu mu
        """
        error_id = self._generate_error_id()
        
        # Hata bilgilerini topla
        error_info = {
            'id': error_id,
            'timestamp': datetime.now().isoformat(),
            'type': type(error).__name__,
            'message': str(error),
            'category': category.value,
            'severity': severity.value,
            'context': context or {},
            'traceback': traceback.format_exc(),
            'system_info': self._get_system_info()
        }
        
        # Hata loguna ekle
        self.error_log.append(error_info)
        
        # İstatistikleri güncelle
        self._update_error_stats(category, severity)
        
        # Detaylı loglama
        self._log_error_details(error_info)
        
        # Kullanıcıya bildirim
        self._notify_user(error_info, user_message)
        
        # Recovery denemesi
        recovery_success = False
        if auto_recover and category in self.recovery_strategies:
            recovery_success = self._attempt_recovery(error_info)
        
        # Sinyal gönder
        self.error_occurred.emit(category.value, str(error), error_info['traceback'])
        if auto_recover:
            self.recovery_attempted.emit(error_id, recovery_success)
        
        return recovery_success
    
    def _generate_error_id(self) -> str:
        """Benzersiz hata ID'si oluştur."""
        return f"ERR_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(self.error_log):04d}"
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Sistem bilgilerini topla."""
        return {
            'python_version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            'platform': sys.platform,
            'memory_usage': self._get_memory_usage(),
            'polars_version': pl.__version__,
            'numpy_version': np.__version__
        }
    
    def _get_memory_usage(self) -> Dict[str, float]:
        """Bellek kullanım bilgisi."""
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            return {
                'rss_mb': memory_info.rss / 1024 / 1024,
                'vms_mb': memory_info.vms / 1024 / 1024,
                'percent': process.memory_percent()
            }
        except ImportError:
            return {'error': 'psutil not available'}
    
    def _update_error_stats(self, category: ErrorCategory, severity: ErrorSeverity):
        """Hata istatistiklerini güncelle."""
        if category.value not in self.error_stats:
            self.error_stats[category.value] = {}
        
        if severity.value not in self.error_stats[category.value]:
            self.error_stats[category.value][severity.value] = 0
        
        self.error_stats[category.value][severity.value] += 1
    
    def _log_error_details(self, error_info: Dict[str, Any]):
        """Detaylı hata loglaması."""
        severity = error_info['severity']
        
        if severity == ErrorSeverity.CRITICAL.value:
            self.logger.critical(f"CRITICAL ERROR [{error_info['id']}]: {error_info['message']}")
        elif severity == ErrorSeverity.HIGH.value:
            self.logger.error(f"HIGH ERROR [{error_info['id']}]: {error_info['message']}")
        elif severity == ErrorSeverity.MEDIUM.value:
            self.logger.warning(f"MEDIUM ERROR [{error_info['id']}]: {error_info['message']}")
        else:
            self.logger.info(f"LOW ERROR [{error_info['id']}]: {error_info['message']}")
        
        # Context bilgilerini logla
        if error_info['context']:
            self.logger.debug(f"Context for [{error_info['id']}]: {json.dumps(error_info['context'], indent=2)}")
        
        # Traceback'i logla
        self.logger.debug(f"Traceback for [{error_info['id']}]:\n{error_info['traceback']}")
    
    def _notify_user(self, error_info: Dict[str, Any], custom_message: str = None):
        """Kullanıcıya hata bildirimi."""
        if not self.parent_widget:
            return
        
        severity = error_info['severity']
        category = error_info['category']
        
        # Kullanıcı dostu mesaj oluştur
        if custom_message:
            user_message = custom_message
        else:
            user_message = self._generate_user_friendly_message(error_info)
        
        # Mesaj kutusu türünü belirle
        if severity == ErrorSeverity.CRITICAL.value:
            icon = QMessageBox.Critical
            title = "Kritik Hata"
        elif severity == ErrorSeverity.HIGH.value:
            icon = QMessageBox.Warning
            title = "Önemli Hata"
        elif severity == ErrorSeverity.MEDIUM.value:
            icon = QMessageBox.Warning
            title = "Uyarı"
        else:
            icon = QMessageBox.Information
            title = "Bilgi"
        
        # Detaylı bilgi
        detailed_info = f"""
Hata ID: {error_info['id']}
Kategori: {category}
Zaman: {error_info['timestamp']}

Teknik Detaylar:
{error_info['type']}: {error_info['message']}
        """
        
        # Mesaj kutusunu göster
        msg_box = QMessageBox(self.parent_widget)
        msg_box.setIcon(icon)
        msg_box.setWindowTitle(title)
        msg_box.setText(user_message)
        msg_box.setDetailedText(detailed_info)
        msg_box.exec_()
    
    def _generate_user_friendly_message(self, error_info: Dict[str, Any]) -> str:
        """Kullanıcı dostu hata mesajı oluştur."""
        category = error_info['category']
        error_type = error_info['type']
        
        messages = {
            ErrorCategory.DATA_FORMAT.value: {
                'default': "Veri formatında sorun tespit edildi. Lütfen dosyanızın formatını kontrol edin.",
                'UnicodeDecodeError': "Dosya encoding sorunu. Farklı encoding (UTF-8, Windows-1254) deneyin.",
                'ValueError': "Veri değerlerinde sorun var. Tarih/saat formatlarını kontrol edin.",
                'ParserError': "Dosya yapısında sorun var. Delimiter ve header ayarlarını kontrol edin."
            },
            ErrorCategory.FILE_IO.value: {
                'default': "Dosya okuma/yazma hatası oluştu.",
                'FileNotFoundError': "Dosya bulunamadı. Dosya yolunu kontrol edin.",
                'PermissionError': "Dosya erişim izni yok. Dosya başka bir program tarafından kullanılıyor olabilir.",
                'OSError': "Sistem seviyesi dosya hatası. Disk alanını ve izinleri kontrol edin."
            },
            ErrorCategory.MEMORY.value: {
                'default': "Bellek yetersizliği. Daha küçük dosya deneyin veya bilgisayarı yeniden başlatın.",
                'MemoryError': "Yetersiz RAM. Dosya çok büyük veya sistem belleği dolu."
            },
            ErrorCategory.VALIDATION.value: {
                'default': "Veri doğrulama hatası. Veri formatını kontrol edin."
            }
        }
        
        category_messages = messages.get(category, {'default': 'Beklenmeyen bir hata oluştu.'})
        return category_messages.get(error_type, category_messages['default'])
    
    def _register_recovery_strategies(self):
        """Recovery stratejilerini kaydet."""
        self.recovery_strategies = {
            ErrorCategory.DATA_FORMAT: self._recover_data_format_error,
            ErrorCategory.FILE_IO: self._recover_file_io_error,
            ErrorCategory.MEMORY: self._recover_memory_error,
            ErrorCategory.VALIDATION: self._recover_validation_error
        }
    
    def _attempt_recovery(self, error_info: Dict[str, Any]) -> bool:
        """Hata recovery denemesi."""
        category = ErrorCategory(error_info['category'])
        
        if category in self.recovery_strategies:
            try:
                self.logger.info(f"Attempting recovery for error [{error_info['id']}]")
                success = self.recovery_strategies[category](error_info)
                
                if success:
                    self.logger.info(f"Recovery successful for [{error_info['id']}]")
                else:
                    self.logger.warning(f"Recovery failed for [{error_info['id']}]")
                
                return success
            except Exception as recovery_error:
                self.logger.error(f"Recovery attempt failed for [{error_info['id']}]: {recovery_error}")
                return False
        
        return False
    
    def _recover_data_format_error(self, error_info: Dict[str, Any]) -> bool:
        """Veri format hatası recovery."""
        context = error_info.get('context', {})
        
        # Farklı encoding denemeleri
        if 'file_path' in context:
            encodings = ['utf-8', 'windows-1254', 'iso-8859-9', 'cp1252']
            for encoding in encodings:
                try:
                    # Test okuma
                    pl.read_csv(context['file_path'], encoding=encoding, n_rows=5)
                    context['suggested_encoding'] = encoding
                    return True
                except:
                    continue
        
        return False
    
    def _recover_file_io_error(self, error_info: Dict[str, Any]) -> bool:
        """Dosya I/O hatası recovery."""
        context = error_info.get('context', {})
        
        if 'file_path' in context:
            file_path = context['file_path']
            
            # Dosya varlığını kontrol et
            if not os.path.exists(file_path):
                context['suggestion'] = 'Dosya bulunamadı. Dosya yolunu kontrol edin.'
                return False
            
            # İzinleri kontrol et
            if not os.access(file_path, os.R_OK):
                context['suggestion'] = 'Dosya okuma izni yok.'
                return False
            
            # Dosya boyutunu kontrol et
            file_size = os.path.getsize(file_path)
            if file_size == 0:
                context['suggestion'] = 'Dosya boş.'
                return False
            
            context['file_size'] = file_size
            return True
        
        return False
    
    def _recover_memory_error(self, error_info: Dict[str, Any]) -> bool:
        """Bellek hatası recovery."""
        context = error_info.get('context', {})
        
        # Chunk reading öner
        if 'file_path' in context:
            context['suggestion'] = 'Dosya çok büyük. Chunk reading kullanın.'
            context['recommended_chunk_size'] = 10000
            return True
        
        return False
    
    def _recover_validation_error(self, error_info: Dict[str, Any]) -> bool:
        """Veri doğrulama hatası recovery."""
        context = error_info.get('context', {})
        
        # Veri temizleme önerileri
        context['suggestions'] = [
            'Eksik değerleri kontrol edin',
            'Veri tiplerini kontrol edin',
            'Outlier değerleri kontrol edin'
        ]
        
        return True
    
    def get_error_report(self) -> Dict[str, Any]:
        """Hata raporu oluştur."""
        return {
            'total_errors': len(self.error_log),
            'error_stats': self.error_stats,
            'recent_errors': self.error_log[-10:] if self.error_log else [],
            'most_common_category': self._get_most_common_category(),
            'suggestions': self._get_general_suggestions()
        }
    
    def _get_most_common_category(self) -> str:
        """En yaygın hata kategorisini bul."""
        if not self.error_stats:
            return "none"
        
        category_totals = {}
        for category, severities in self.error_stats.items():
            category_totals[category] = sum(severities.values())
        
        return max(category_totals, key=category_totals.get)
    
    def _get_general_suggestions(self) -> list:
        """Genel öneriler."""
        suggestions = []
        
        if self.error_stats:
            most_common = self._get_most_common_category()
            
            if most_common == ErrorCategory.DATA_FORMAT.value:
                suggestions.extend([
                    "Dosya encoding'ini kontrol edin (UTF-8 önerilen)",
                    "Tarih/saat formatlarını standartlaştırın",
                    "CSV delimiter'ını doğru ayarlayın"
                ])
            elif most_common == ErrorCategory.FILE_IO.value:
                suggestions.extend([
                    "Dosya izinlerini kontrol edin",
                    "Dosya başka program tarafından kullanılıyor olabilir",
                    "Disk alanını kontrol edin"
                ])
            elif most_common == ErrorCategory.MEMORY.value:
                suggestions.extend([
                    "Daha küçük dosyalar kullanın",
                    "Bilgisayarı yeniden başlatın",
                    "Diğer programları kapatın"
                ])
        
        return suggestions
    
    def export_error_log(self, file_path: str):
        """Hata logunu dosyaya aktar."""
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'export_time': datetime.now().isoformat(),
                    'error_log': self.error_log,
                    'error_stats': self.error_stats,
                    'report': self.get_error_report()
                }, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Error log exported to: {file_path}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to export error log: {e}")
            return False

# Decorator fonksiyonları
def handle_data_errors(error_handler: ErrorHandler, 
                      category: ErrorCategory = ErrorCategory.DATA_FORMAT,
                      severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                      auto_recover: bool = True):
    """Veri işleme hatalarını yakalayan decorator."""
    def decorator(func: Callable):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                context = {
                    'function': func.__name__,
                    'args': str(args)[:200],  # İlk 200 karakter
                    'kwargs': str(kwargs)[:200]
                }
                
                error_handler.handle_error(
                    error=e,
                    category=category,
                    severity=severity,
                    context=context,
                    auto_recover=auto_recover
                )
                
                # Hata durumunda None döndür veya varsayılan değer
                return None
        return wrapper
    return decorator

def safe_file_operation(error_handler: ErrorHandler):
    """Dosya işlemlerini güvenli hale getiren decorator."""
    def decorator(func: Callable):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except (FileNotFoundError, PermissionError, OSError) as e:
                context = {
                    'function': func.__name__,
                    'file_path': kwargs.get('file_path') or (args[0] if args else 'unknown')
                }
                
                error_handler.handle_error(
                    error=e,
                    category=ErrorCategory.FILE_IO,
                    severity=ErrorSeverity.HIGH,
                    context=context
                )
                return None
        return wrapper
    return decorator
