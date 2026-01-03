#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
License Manager - Ticari Lisans Yönetim Sistemi
===============================================

Bu modül, uygulamanın ticari lisans kontrolünü sağlar.
"""

import os
import json
import hashlib
import platform
import uuid
from datetime import datetime, timedelta
from typing import Optional, Tuple
from pathlib import Path


class LicenseManager:
    """
    Ticari lisans yönetim sistemi.
    
    Özellikler:
    - Hardware-based license key
    - Süre sınırlı lisanslar
    - Trial mode desteği
    - Lisans doğrulama
    """
    
    LICENSE_FILE = "license.dat"
    TRIAL_DAYS = 30
    
    def __init__(self, app_name: str = "TimeGraphApp"):
        """
        Args:
            app_name: Uygulama adı
        """
        self.app_name = app_name
        self.license_path = self._get_license_path()
        self._license_data = None
    
    def _get_license_path(self) -> Path:
        """Lisans dosyasının yolunu al"""
        # Windows: %APPDATA%\TimeGraphApp\license.dat
        # Linux/Mac: ~/.timegraphapp/license.dat
        
        if platform.system() == "Windows":
            base_dir = Path(os.environ.get('APPDATA', os.path.expanduser('~')))
        else:
            base_dir = Path.home()
        
        app_dir = base_dir / f".{self.app_name.lower()}"
        app_dir.mkdir(parents=True, exist_ok=True)
        
        return app_dir / self.LICENSE_FILE
    
    def _get_machine_id(self) -> str:
        """Makine benzersiz kimliğini al"""
        # Makine bilgilerinden hash üret
        machine_info = f"{platform.node()}-{platform.machine()}-{platform.processor()}"
        
        # MAC adresi ekle (daha güvenilir)
        try:
            mac = ':'.join(['{:02x}'.format((uuid.getnode() >> elements) & 0xff)
                           for elements in range(0, 2*6, 2)][::-1])
            machine_info += f"-{mac}"
        except:
            pass
        
        # SHA256 hash
        return hashlib.sha256(machine_info.encode()).hexdigest()[:32]
    
    def check_license(self) -> Tuple[bool, str, Optional[dict]]:
        """
        Lisansı kontrol et.
        
        Returns:
            (is_valid, message, license_info)
            - is_valid: Lisans geçerli mi?
            - message: Kullanıcıya gösterilecek mesaj
            - license_info: Lisans bilgileri (varsa)
        """
        # 1. Lisans dosyası var mı?
        if not self.license_path.exists():
            return self._check_trial_mode()
        
        # 2. Lisans dosyasını oku ve doğrula
        try:
            with open(self.license_path, 'r', encoding='utf-8') as f:
                license_data = json.load(f)
            
            # Lisans tipini kontrol et
            license_type = license_data.get('type', 'unknown')
            
            if license_type == 'trial':
                return self._validate_trial_license(license_data)
            elif license_type == 'full':
                return self._validate_full_license(license_data)
            elif license_type == 'subscription':
                return self._validate_subscription_license(license_data)
            else:
                return False, "Geçersiz lisans tipi", None
                
        except Exception as e:
            return False, f"Lisans dosyası okunamadı: {str(e)}", None
    
    def _check_trial_mode(self) -> Tuple[bool, str, Optional[dict]]:
        """Trial modunu kontrol et ve başlat"""
        # İlk çalıştırma - trial başlat
        trial_end = datetime.now() + timedelta(days=self.TRIAL_DAYS)
        
        license_data = {
            'type': 'trial',
            'start_date': datetime.now().isoformat(),
            'end_date': trial_end.isoformat(),
            'machine_id': self._get_machine_id()
        }
        
        try:
            with open(self.license_path, 'w', encoding='utf-8') as f:
                json.dump(license_data, f, indent=2)
            
            return (
                True, 
                f"Trial sürümü başlatıldı. {self.TRIAL_DAYS} gün boyunca tüm özellikleri kullanabilirsiniz.",
                license_data
            )
        except Exception as e:
            return False, f"Trial başlatılamadı: {str(e)}", None
    
    def _validate_trial_license(self, license_data: dict) -> Tuple[bool, str, Optional[dict]]:
        """Trial lisansını doğrula"""
        try:
            end_date = datetime.fromisoformat(license_data['end_date'])
            now = datetime.now()
            
            if now > end_date:
                return False, "Trial süresi doldu. Tam lisans satın almalısınız.", None
            
            remaining_days = (end_date - now).days
            
            message = f"Trial modu aktif. Kalan süre: {remaining_days} gün"
            
            return True, message, license_data
            
        except Exception as e:
            return False, f"Trial lisansı doğrulanamadı: {str(e)}", None
    
    def _validate_full_license(self, license_data: dict) -> Tuple[bool, str, Optional[dict]]:
        """Tam lisansı doğrula"""
        try:
            # Machine ID kontrolü
            stored_machine_id = license_data.get('machine_id')
            current_machine_id = self._get_machine_id()
            
            if stored_machine_id != current_machine_id:
                return False, "Lisans bu bilgisayar için geçerli değil.", None
            
            # License key doğrulama
            license_key = license_data.get('license_key')
            if not self._verify_license_key(license_key, license_data):
                return False, "Lisans anahtarı geçersiz.", None
            
            owner = license_data.get('owner', 'Kayıtlı Kullanıcı')
            return True, f"Tam lisans aktif. Lisans sahibi: {owner}", license_data
            
        except Exception as e:
            return False, f"Lisans doğrulanamadı: {str(e)}", None
    
    def _validate_subscription_license(self, license_data: dict) -> Tuple[bool, str, Optional[dict]]:
        """Abonelik lisansını doğrula"""
        try:
            # Machine ID kontrolü
            stored_machine_id = license_data.get('machine_id')
            current_machine_id = self._get_machine_id()
            
            if stored_machine_id != current_machine_id:
                return False, "Lisans bu bilgisayar için geçerli değil.", None
            
            # Subscription bitiş tarihi kontrolü
            end_date = datetime.fromisoformat(license_data['subscription_end'])
            now = datetime.now()
            
            if now > end_date:
                return False, "Abonelik süresi doldu. Lütfen yenileyin.", None
            
            remaining_days = (end_date - now).days
            owner = license_data.get('owner', 'Kayıtlı Kullanıcı')
            
            message = f"Abonelik aktif. Kalan süre: {remaining_days} gün. Lisans sahibi: {owner}"
            
            return True, message, license_data
            
        except Exception as e:
            return False, f"Abonelik doğrulanamadı: {str(e)}", None
    
    def _verify_license_key(self, license_key: str, license_data: dict) -> bool:
        """Lisans anahtarını doğrula"""
        try:
            # Basit doğrulama - Production'da daha güvenli bir yöntem kullanılmalı
            machine_id = license_data.get('machine_id')
            owner = license_data.get('owner', '')
            
            # Expected key format: HASH(machine_id + owner + secret)
            secret = "YOUR_SECRET_KEY_HERE"  # Bu değeri değiştirin!
            
            expected_key = hashlib.sha256(
                f"{machine_id}{owner}{secret}".encode()
            ).hexdigest()[:16].upper()
            
            return license_key == expected_key
            
        except:
            return False
    
    def activate_license(self, license_key: str, owner: str = "", license_type: str = "full") -> Tuple[bool, str]:
        """
        Lisansı aktive et.
        
        Args:
            license_key: Lisans anahtarı
            owner: Lisans sahibi
            license_type: Lisans tipi ('full' veya 'subscription')
        
        Returns:
            (success, message)
        """
        try:
            machine_id = self._get_machine_id()
            
            license_data = {
                'type': license_type,
                'license_key': license_key,
                'owner': owner,
                'machine_id': machine_id,
                'activation_date': datetime.now().isoformat()
            }
            
            # Subscription ise bitiş tarihi ekle
            if license_type == 'subscription':
                # 1 yıllık subscription
                end_date = datetime.now() + timedelta(days=365)
                license_data['subscription_end'] = end_date.isoformat()
            
            # Lisansı doğrula
            if not self._verify_license_key(license_key, license_data):
                return False, "Geçersiz lisans anahtarı"
            
            # Lisansı kaydet
            with open(self.license_path, 'w', encoding='utf-8') as f:
                json.dump(license_data, f, indent=2)
            
            return True, "Lisans başarıyla aktive edildi!"
            
        except Exception as e:
            return False, f"Lisans aktivasyonu başarısız: {str(e)}"
    
    def get_machine_id_for_activation(self) -> str:
        """Aktivasyon için makine kimliğini al (kullanıcıya gösterilecek)"""
        return self._get_machine_id()
    
    def get_license_info(self) -> Optional[dict]:
        """Mevcut lisans bilgilerini al"""
        is_valid, message, license_info = self.check_license()
        
        if license_info:
            return {
                'is_valid': is_valid,
                'message': message,
                'type': license_info.get('type'),
                'owner': license_info.get('owner', 'N/A'),
                'machine_id': self._get_machine_id()
            }
        
        return None


# Singleton instance
_license_manager_instance: Optional[LicenseManager] = None


def get_license_manager() -> LicenseManager:
    """Global LicenseManager instance'ını al"""
    global _license_manager_instance
    
    if _license_manager_instance is None:
        _license_manager_instance = LicenseManager()
    
    return _license_manager_instance

