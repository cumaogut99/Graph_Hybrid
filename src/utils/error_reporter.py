#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Error Reporter - Profesyonel Hata Raporlama Sistemi
===================================================

Kullanıcı dostu hata mesajları ve otomatik hata raporlama.
"""

import sys
import traceback
import platform
from datetime import datetime
from typing import Optional, Callable
from pathlib import Path


class ErrorReporter:
    """
    Profesyonel hata raporlama sistemi.
    
    Özellikler:
    - Kullanıcı dostu hata mesajları
    - Detaylı hata logları (development)
    - Otomatik crash report oluşturma
    - Sistem bilgilerini toplama
    """
    
    def __init__(self, app_name: str = "TimeGraphApp", app_version: str = "1.0.0"):
        """
        Args:
            app_name: Uygulama adı
            app_version: Uygulama versiyonu
        """
        self.app_name = app_name
        self.app_version = app_version
        self.crash_dir = Path("crash_reports")
        self.crash_dir.mkdir(exist_ok=True)
    
    def get_user_friendly_message(self, exception: Exception) -> str:
        """
        Kullanıcı dostu hata mesajı üret.
        
        Args:
            exception: Hata nesnesi
        
        Returns:
            Kullanıcıya gösterilecek mesaj
        """
        error_type = type(exception).__name__
        
        # Yaygın hatalar için özel mesajlar
        friendly_messages = {
            'FileNotFoundError': "📁 Dosya bulunamadı. Lütfen dosya yolunu kontrol edin.",
            'PermissionError': "🔒 Dosyaya erişim izni yok. Lütfen yönetici olarak çalıştırın.",
            'MemoryError': "💾 Bellek yetersiz. Lütfen büyük dosyaları parçalayın veya RAM artırın.",
            'ValueError': "⚠️ Geçersiz veri formatı. Lütfen veri formatını kontrol edin.",
            'KeyError': "🔑 Beklenen veri bulunamadı. Dosya formatı hatalı olabilir.",
            'ImportError': "📦 Gerekli kütüphane bulunamadı. Lütfen requirements.txt'i yükleyin.",
            'ConnectionError': "🌐 Bağlantı hatası. İnternet bağlantınızı kontrol edin.",
            'TimeoutError': "⏱️ İşlem zaman aşımına uğradı. Lütfen tekrar deneyin.",
        }
        
        base_message = friendly_messages.get(
            error_type,
            f"❌ Beklenmeyen bir hata oluştu: {error_type}"
        )
        
        # Hatanın kısa açıklamasını ekle (çok detaylı olmasın)
        error_detail = str(exception)
        if error_detail and len(error_detail) < 100:
            base_message += f"\n\n<b>Detay:</b> {error_detail}"
        
        return base_message
    
    def get_system_info(self) -> dict:
        """Sistem bilgilerini topla"""
        try:
            import psutil
            memory = psutil.virtual_memory()
            memory_info = f"{memory.total / (1024**3):.1f} GB (Kullanılabilir: {memory.available / (1024**3):.1f} GB)"
        except:
            memory_info = "N/A"
        
        return {
            'app_name': self.app_name,
            'app_version': self.app_version,
            'platform': platform.platform(),
            'python_version': platform.python_version(),
            'architecture': platform.machine(),
            'processor': platform.processor(),
            'memory': memory_info,
            'timestamp': datetime.now().isoformat()
        }
    
    def create_crash_report(self, exception: Exception, extra_info: Optional[dict] = None) -> Path:
        """
        Crash report oluştur.
        
        Args:
            exception: Hata nesnesi
            extra_info: Ek bilgiler
        
        Returns:
            Crash report dosya yolu
        """
        # Dosya adı
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"crash_{timestamp}.txt"
        filepath = self.crash_dir / filename
        
        # Rapor içeriği
        report_lines = [
            "=" * 70,
            f"{self.app_name} - CRASH REPORT",
            "=" * 70,
            "",
            "SYSTEM INFORMATION:",
            "-" * 70,
        ]
        
        # Sistem bilgileri
        sys_info = self.get_system_info()
        for key, value in sys_info.items():
            report_lines.append(f"{key}: {value}")
        
        report_lines.extend([
            "",
            "ERROR INFORMATION:",
            "-" * 70,
            f"Exception Type: {type(exception).__name__}",
            f"Exception Message: {str(exception)}",
            "",
            "TRACEBACK:",
            "-" * 70,
        ])
        
        # Stack trace
        tb_lines = traceback.format_exception(type(exception), exception, exception.__traceback__)
        report_lines.extend(tb_lines)
        
        # Ek bilgiler
        if extra_info:
            report_lines.extend([
                "",
                "ADDITIONAL INFORMATION:",
                "-" * 70,
            ])
            for key, value in extra_info.items():
                report_lines.append(f"{key}: {value}")
        
        report_lines.extend([
            "",
            "=" * 70,
            "END OF REPORT",
            "=" * 70,
        ])
        
        # Dosyaya yaz
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write('\n'.join(report_lines))
            
            return filepath
        except Exception as e:
            print(f"Crash report oluşturulamadı: {e}")
            return None
    
    def handle_exception(
        self, 
        exception: Exception, 
        show_dialog: bool = True,
        extra_info: Optional[dict] = None,
        dialog_callback: Optional[Callable] = None
    ) -> Optional[Path]:
        """
        Hatayı handle et.
        
        Args:
            exception: Hata nesnesi
            show_dialog: Dialog gösterilsin mi?
            extra_info: Ek bilgiler
            dialog_callback: Dialog gösterme fonksiyonu
        
        Returns:
            Crash report dosya yolu (varsa)
        """
        # Crash report oluştur
        crash_report = self.create_crash_report(exception, extra_info)
        
        # Kullanıcıya göster
        if show_dialog and dialog_callback:
            user_message = self.get_user_friendly_message(exception)
            
            if crash_report:
                user_message += f"\n\n<i>Hata raporu kaydedildi: {crash_report.name}</i>"
            
            dialog_callback(user_message)
        
        return crash_report
    
    def install_global_exception_handler(self, dialog_callback: Optional[Callable] = None):
        """
        Global exception handler kur.
        
        Yakalanmayan tüm hataları yakalar ve rapor oluşturur.
        
        Args:
            dialog_callback: Dialog gösterme fonksiyonu
        """
        def exception_hook(exc_type, exc_value, exc_traceback):
            """Global exception handler"""
            # KeyboardInterrupt'ı normal şekilde handle et
            if issubclass(exc_type, KeyboardInterrupt):
                sys.__excepthook__(exc_type, exc_value, exc_traceback)
                return
            
            # Hata raporu oluştur
            exception = exc_value if exc_value else exc_type()
            
            print("\n" + "=" * 70)
            print("UNHANDLED EXCEPTION CAUGHT")
            print("=" * 70)
            traceback.print_exception(exc_type, exc_value, exc_traceback)
            
            crash_report = self.create_crash_report(exception, {
                'exc_type': str(exc_type),
                'unhandled': True
            })
            
            # Kullanıcıya bildir
            if dialog_callback:
                user_message = self.get_user_friendly_message(exception)
                user_message += "\n\n<b>Uygulama beklenmedik bir hatayla karşılaştı.</b>"
                
                if crash_report:
                    user_message += f"\n\nHata raporu: {crash_report.name}"
                
                dialog_callback(user_message)
        
        # Exception handler'ı kur
        sys.excepthook = exception_hook


# Global error reporter instance
_error_reporter_instance: Optional[ErrorReporter] = None


def get_error_reporter(app_name: str = "TimeGraphApp", app_version: str = "1.0.0") -> ErrorReporter:
    """Global ErrorReporter instance'ını al"""
    global _error_reporter_instance
    
    if _error_reporter_instance is None:
        _error_reporter_instance = ErrorReporter(app_name, app_version)
    
    return _error_reporter_instance


def report_error(exception: Exception, show_dialog: bool = True, extra_info: Optional[dict] = None):
    """Hızlı hata raporlama"""
    reporter = get_error_reporter()
    return reporter.handle_exception(exception, show_dialog, extra_info)

