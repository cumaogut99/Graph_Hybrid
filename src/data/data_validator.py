#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Akıllı Veri Doğrulama Sistemi
============================

Bu modül, Time Graph uygulaması için kapsamlı veri doğrulama,
format tespiti ve otomatik düzeltme sistemi sağlar.
"""

import re
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from enum import Enum
import polars as pl

logger = logging.getLogger(__name__)

class DataType(Enum):
    """Veri tipi kategorileri."""
    NUMERIC = "numeric"
    DATETIME = "datetime"
    STRING = "string"
    BOOLEAN = "boolean"
    MIXED = "mixed"
    UNKNOWN = "unknown"

class DateTimeFormat(Enum):
    """Yaygın tarih/saat formatları."""
    ISO_8601 = "%Y-%m-%d %H:%M:%S"
    ISO_DATE = "%Y-%m-%d"
    US_FORMAT = "%m/%d/%Y %H:%M:%S"
    EU_FORMAT = "%d/%m/%Y %H:%M:%S"
    TURKISH_FORMAT = "%d.%m.%Y %H:%M:%S"
    TIMESTAMP = "timestamp"
    EXCEL_SERIAL = "excel_serial"

class ValidationResult:
    """Doğrulama sonucu sınıfı."""
    
    def __init__(self):
        self.is_valid = True
        self.data_type = DataType.UNKNOWN
        self.issues = []
        self.suggestions = []
        self.confidence = 0.0
        self.detected_format = None
        self.sample_values = []
        self.statistics = {}

class DataValidator:
    """Akıllı veri doğrulama sistemi."""
    
    def __init__(self):
        self.datetime_patterns = self._initialize_datetime_patterns()
        self.validation_cache = {}
        
    def _initialize_datetime_patterns(self) -> Dict[str, Dict]:
        """Tarih/saat pattern'lerini başlat."""
        return {
            'iso_datetime': {
                'pattern': r'^\d{4}-\d{2}-\d{2}[\sT]\d{2}:\d{2}:\d{2}',
                'format': '%Y-%m-%d %H:%M:%S',
                'confidence': 0.95
            },
            'iso_date': {
                'pattern': r'^\d{4}-\d{2}-\d{2}$',
                'format': '%Y-%m-%d',
                'confidence': 0.90
            },
            'us_datetime': {
                'pattern': r'^\d{1,2}/\d{1,2}/\d{4}\s\d{1,2}:\d{2}:\d{2}',
                'format': '%m/%d/%Y %H:%M:%S',
                'confidence': 0.85
            },
            'eu_datetime': {
                'pattern': r'^\d{1,2}/\d{1,2}/\d{4}\s\d{1,2}:\d{2}:\d{2}',
                'format': '%d/%m/%Y %H:%M:%S',
                'confidence': 0.80
            },
            'turkish_datetime': {
                'pattern': r'^\d{1,2}\.\d{1,2}\.\d{4}\s\d{1,2}:\d{2}:\d{2}',
                'format': '%d.%m.%Y %H:%M:%S',
                'confidence': 0.85
            },
            'timestamp': {
                'pattern': r'^\d{10}(\.\d+)?$',
                'format': 'timestamp',
                'confidence': 0.90
            },
            'excel_serial': {
                'pattern': r'^\d{5}(\.\d+)?$',
                'format': 'excel_serial',
                'confidence': 0.70
            }
        }
    
    def validate_dataframe(self, df: pd.DataFrame) -> Dict[str, ValidationResult]:
        """DataFrame'in tüm kolonlarını doğrula."""
        results = {}
        
        for column in df.columns:
            logger.debug(f"Validating column: {column}")
            results[column] = self.validate_column(df[column], column)
            
        return results
    
    def validate_column(self, series: pd.Series, column_name: str = None) -> ValidationResult:
        """Tek bir kolonu doğrula."""
        result = ValidationResult()
        
        # Cache kontrolü
        cache_key = f"{column_name}_{len(series)}_{hash(str(series.iloc[0]) if len(series) > 0 else '')}"
        if cache_key in self.validation_cache:
            return self.validation_cache[cache_key]
        
        # Temel istatistikler
        result.statistics = self._calculate_basic_stats(series)
        
        # Veri tipi tespiti
        result.data_type, result.confidence = self._detect_data_type(series)
        
        # Spesifik doğrulamalar
        if result.data_type == DataType.DATETIME:
            self._validate_datetime_column(series, result)
        elif result.data_type == DataType.NUMERIC:
            self._validate_numeric_column(series, result)
        elif result.data_type == DataType.STRING:
            self._validate_string_column(series, result)
        elif result.data_type == DataType.MIXED:
            self._validate_mixed_column(series, result)
        
        # Genel sorunları kontrol et
        self._check_common_issues(series, result)
        
        # Önerileri oluştur
        self._generate_suggestions(series, result)
        
        # Sample değerleri kaydet
        result.sample_values = series.dropna().head(5).tolist()
        
        # Cache'e kaydet
        self.validation_cache[cache_key] = result
        
        return result
    
    def _calculate_basic_stats(self, series: pd.Series) -> Dict[str, Any]:
        """Temel istatistikleri hesapla."""
        stats = {
            'total_count': len(series),
            'null_count': series.isnull().sum(),
            'unique_count': series.nunique(),
            'null_percentage': (series.isnull().sum() / len(series)) * 100
        }
        
        # Numeric ise ek istatistikler
        try:
            numeric_series = pd.to_numeric(series, errors='coerce')
            if not numeric_series.isnull().all():
                stats.update({
                    'min_value': numeric_series.min(),
                    'max_value': numeric_series.max(),
                    'mean_value': numeric_series.mean(),
                    'std_value': numeric_series.std()
                })
        except:
            pass
        
        return stats
    
    def _detect_data_type(self, series: pd.Series) -> Tuple[DataType, float]:
        """Veri tipini akıllı tespit et."""
        non_null_series = series.dropna()
        
        if len(non_null_series) == 0:
            return DataType.UNKNOWN, 0.0
        
        # Numeric kontrol
        numeric_confidence = self._check_numeric_type(non_null_series)
        if numeric_confidence > 0.8:
            return DataType.NUMERIC, numeric_confidence
        
        # DateTime kontrol
        datetime_confidence, detected_format = self._check_datetime_type(non_null_series)
        if datetime_confidence > 0.7:
            return DataType.DATETIME, datetime_confidence
        
        # Boolean kontrol
        boolean_confidence = self._check_boolean_type(non_null_series)
        if boolean_confidence > 0.8:
            return DataType.BOOLEAN, boolean_confidence
        
        # Mixed kontrol
        if self._is_mixed_type(non_null_series):
            return DataType.MIXED, 0.6
        
        # Varsayılan string
        return DataType.STRING, 0.5
    
    def _check_numeric_type(self, series: pd.Series) -> float:
        """Numeric tip kontrolü."""
        try:
            # Pandas numeric conversion
            numeric_series = pd.to_numeric(series, errors='coerce')
            valid_count = (~numeric_series.isnull()).sum()
            confidence = valid_count / len(series)
            
            # Ek kontroller
            if confidence > 0.9:
                # Tüm değerler numeric
                return confidence
            elif confidence > 0.7:
                # Çoğu değer numeric, bazıları string olabilir
                return confidence * 0.9
            else:
                return confidence * 0.5
                
        except:
            return 0.0
    
    def _check_datetime_type(self, series: pd.Series) -> Tuple[float, Optional[str]]:
        """DateTime tip kontrolü."""
        max_confidence = 0.0
        best_format = None
        
        # String değerleri kontrol et
        string_series = series.astype(str)
        
        for pattern_name, pattern_info in self.datetime_patterns.items():
            try:
                pattern = pattern_info['pattern']
                format_str = pattern_info['format']
                base_confidence = pattern_info['confidence']
                
                if format_str == 'timestamp':
                    # Unix timestamp kontrolü
                    matches = string_series.str.match(pattern).sum()
                elif format_str == 'excel_serial':
                    # Excel serial date kontrolü
                    matches = string_series.str.match(pattern).sum()
                else:
                    # Normal datetime format kontrolü
                    matches = string_series.str.match(pattern).sum()
                
                if matches > 0:
                    confidence = (matches / len(series)) * base_confidence
                    if confidence > max_confidence:
                        max_confidence = confidence
                        best_format = format_str
                        
            except Exception as e:
                logger.debug(f"DateTime pattern check failed for {pattern_name}: {e}")
                continue
        
        # Pandas datetime parsing denemesi
        if max_confidence < 0.7:
            try:
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    pd.to_datetime(series.head(10), errors='raise')
                max_confidence = max(max_confidence, 0.8)
                best_format = best_format or 'auto'
            except:
                pass
        
        return max_confidence, best_format
    
    def _check_boolean_type(self, series: pd.Series) -> float:
        """Boolean tip kontrolü."""
        string_series = series.astype(str).str.lower()
        boolean_values = {'true', 'false', '1', '0', 'yes', 'no', 'y', 'n', 'evet', 'hayır'}
        
        matches = string_series.isin(boolean_values).sum()
        return matches / len(series)
    
    def _is_mixed_type(self, series: pd.Series) -> bool:
        """Mixed tip kontrolü."""
        # Farklı veri tiplerinin karışımı var mı?
        types = set()
        
        for value in series.head(20):  # İlk 20 değeri kontrol et
            if pd.isna(value):
                continue
            
            str_val = str(value)
            
            # Numeric mi?
            try:
                float(str_val)
                types.add('numeric')
            except:
                pass
            
            # DateTime mi?
            try:
                pd.to_datetime(str_val)
                types.add('datetime')
            except:
                pass
            
            # String
            if not str_val.isdigit():
                types.add('string')
        
        return len(types) > 1
    
    def _validate_datetime_column(self, series: pd.Series, result: ValidationResult):
        """DateTime kolonu doğrulaması."""
        # Format tespiti
        confidence, detected_format = self._check_datetime_type(series.dropna())
        result.detected_format = detected_format
        
        # Conversion test
        try:
            if detected_format == 'timestamp':
                converted = pd.to_datetime(series, unit='s', errors='coerce')
            elif detected_format == 'excel_serial':
                # Excel serial date conversion
                converted = pd.to_datetime(series, unit='D', origin='1899-12-30', errors='coerce')
            else:
                converted = pd.to_datetime(series, errors='coerce')
            
            failed_conversions = converted.isnull().sum() - series.isnull().sum()
            
            if failed_conversions > 0:
                result.issues.append(f"{failed_conversions} değer datetime formatına çevrilemedi")
                result.is_valid = False
            
            # Tarih aralığı kontrolü
            if not converted.dropna().empty:
                min_date = converted.min()
                max_date = converted.max()
                
                # Makul tarih aralığı kontrolü
                if min_date < datetime(1900, 1, 1):
                    result.issues.append(f"Çok eski tarih tespit edildi: {min_date}")
                
                if max_date > datetime.now() + timedelta(days=365):
                    result.issues.append(f"Gelecek tarih tespit edildi: {max_date}")
                
                result.statistics.update({
                    'min_date': min_date,
                    'max_date': max_date,
                    'date_range_days': (max_date - min_date).days
                })
                
        except Exception as e:
            result.issues.append(f"DateTime doğrulama hatası: {str(e)}")
            result.is_valid = False
    
    def _validate_numeric_column(self, series: pd.Series, result: ValidationResult):
        """Numeric kolonu doğrulaması."""
        try:
            numeric_series = pd.to_numeric(series, errors='coerce')
            
            # Conversion başarısızlıkları
            failed_conversions = numeric_series.isnull().sum() - series.isnull().sum()
            if failed_conversions > 0:
                result.issues.append(f"{failed_conversions} değer sayısal formata çevrilemedi")
            
            # Outlier kontrolü
            if not numeric_series.dropna().empty:
                Q1 = numeric_series.quantile(0.25)
                Q3 = numeric_series.quantile(0.75)
                IQR = Q3 - Q1
                
                outliers = numeric_series[
                    (numeric_series < Q1 - 1.5 * IQR) | 
                    (numeric_series > Q3 + 1.5 * IQR)
                ]
                
                if len(outliers) > 0:
                    result.issues.append(f"{len(outliers)} outlier değer tespit edildi")
                
                # Infinity kontrolü
                inf_count = np.isinf(numeric_series).sum()
                if inf_count > 0:
                    result.issues.append(f"{inf_count} sonsuz değer tespit edildi")
                    result.is_valid = False
                
        except Exception as e:
            result.issues.append(f"Numeric doğrulama hatası: {str(e)}")
            result.is_valid = False
    
    def _validate_string_column(self, series: pd.Series, result: ValidationResult):
        """String kolonu doğrulaması."""
        string_series = series.astype(str)
        
        # Encoding kontrolü
        try:
            for value in string_series.head(10):
                value.encode('utf-8')
        except UnicodeEncodeError:
            result.issues.append("Encoding sorunu tespit edildi")
            result.is_valid = False
        
        # Boş string kontrolü
        empty_strings = (string_series == '').sum()
        if empty_strings > 0:
            result.issues.append(f"{empty_strings} boş string tespit edildi")
        
        # Çok uzun string kontrolü
        max_length = string_series.str.len().max()
        if max_length > 1000:
            result.issues.append(f"Çok uzun string tespit edildi: {max_length} karakter")
    
    def _validate_mixed_column(self, series: pd.Series, result: ValidationResult):
        """Mixed tip kolonu doğrulaması."""
        result.issues.append("Karışık veri tipleri tespit edildi")
        result.is_valid = False
        
        # Tip dağılımını analiz et
        type_counts = {}
        for value in series.dropna().head(50):
            value_type = self._get_value_type(value)
            type_counts[value_type] = type_counts.get(value_type, 0) + 1
        
        result.statistics['type_distribution'] = type_counts
    
    def _get_value_type(self, value) -> str:
        """Tek bir değerin tipini tespit et."""
        str_val = str(value)
        
        # Numeric
        try:
            float(str_val)
            return 'numeric'
        except:
            pass
        
        # DateTime
        try:
            pd.to_datetime(str_val)
            return 'datetime'
        except:
            pass
        
        # Boolean
        if str_val.lower() in {'true', 'false', '1', '0', 'yes', 'no'}:
            return 'boolean'
        
        return 'string'
    
    def _check_common_issues(self, series: pd.Series, result: ValidationResult):
        """Yaygın sorunları kontrol et."""
        # Yüksek null oranı
        null_percentage = result.statistics['null_percentage']
        if null_percentage > 50:
            result.issues.append(f"Yüksek null oranı: %{null_percentage:.1f}")
            result.is_valid = False
        elif null_percentage > 20:
            result.issues.append(f"Orta seviye null oranı: %{null_percentage:.1f}")
        
        # Düşük unique değer sayısı
        if result.statistics['unique_count'] == 1:
            result.issues.append("Tüm değerler aynı (constant column)")
        elif result.statistics['unique_count'] < 3 and len(series) > 10:
            result.issues.append("Çok az unique değer")
        
        # Çok yüksek unique değer sayısı
        unique_ratio = result.statistics['unique_count'] / result.statistics['total_count']
        if unique_ratio > 0.95 and len(series) > 100:
            result.issues.append("Çok yüksek unique değer oranı (ID kolonu olabilir)")
    
    def _generate_suggestions(self, series: pd.Series, result: ValidationResult):
        """Önerileri oluştur."""
        # Veri tipine göre öneriler
        if result.data_type == DataType.DATETIME:
            if result.detected_format:
                result.suggestions.append(f"Tespit edilen format: {result.detected_format}")
            
            if 'datetime formatına çevrilemedi' in str(result.issues):
                result.suggestions.extend([
                    "Farklı datetime formatları deneyin",
                    "Manuel format belirtmeyi düşünün",
                    "Hatalı değerleri temizleyin"
                ])
        
        elif result.data_type == DataType.NUMERIC:
            if 'sayısal formata çevrilemedi' in str(result.issues):
                result.suggestions.extend([
                    "Sayısal olmayan karakterleri temizleyin",
                    "Decimal separator'ı kontrol edin (. vs ,)",
                    "Currency sembolleri varsa kaldırın"
                ])
            
            if 'outlier' in str(result.issues):
                result.suggestions.extend([
                    "Outlier değerleri inceleyin",
                    "Veri temizleme uygulayın",
                    "Log transformation düşünün"
                ])
        
        elif result.data_type == DataType.MIXED:
            result.suggestions.extend([
                "Veri tiplerini standardize edin",
                "Farklı kolonlara ayırmayı düşünün",
                "Veri temizleme uygulayın"
            ])
        
        # Genel öneriler
        if result.statistics['null_percentage'] > 20:
            result.suggestions.extend([
                "Null değerleri doldurmayı düşünün",
                "Null değerleri kaldırmayı düşünün",
                "Veri kaynağını kontrol edin"
            ])
    
    def suggest_fixes(self, df: pd.DataFrame, validation_results: Dict[str, ValidationResult]) -> Dict[str, Dict]:
        """Otomatik düzeltme önerileri."""
        fixes = {}
        
        for column, result in validation_results.items():
            column_fixes = {}
            
            if not result.is_valid:
                if result.data_type == DataType.DATETIME:
                    column_fixes['datetime_conversion'] = {
                        'method': 'pd.to_datetime',
                        'params': {'errors': 'coerce'},
                        'format': result.detected_format
                    }
                
                elif result.data_type == DataType.NUMERIC:
                    column_fixes['numeric_conversion'] = {
                        'method': 'pd.to_numeric',
                        'params': {'errors': 'coerce'}
                    }
                
                elif result.data_type == DataType.MIXED:
                    column_fixes['type_separation'] = {
                        'suggestion': 'Split into multiple columns by type'
                    }
            
            # Null değer düzeltmeleri
            if result.statistics['null_percentage'] > 0:
                if result.data_type == DataType.NUMERIC:
                    column_fixes['null_handling'] = {
                        'methods': ['fillna(mean)', 'fillna(median)', 'interpolate']
                    }
                elif result.data_type == DataType.DATETIME:
                    column_fixes['null_handling'] = {
                        'methods': ['fillna(method="ffill")', 'fillna(method="bfill")']
                    }
                else:
                    column_fixes['null_handling'] = {
                        'methods': ['fillna("Unknown")', 'dropna()']
                    }
            
            if column_fixes:
                fixes[column] = column_fixes
        
        return fixes
    
    def auto_fix_dataframe(self, df: pd.DataFrame, validation_results: Dict[str, ValidationResult]) -> pd.DataFrame:
        """Otomatik veri düzeltme."""
        df_fixed = df.copy()
        
        for column, result in validation_results.items():
            if not result.is_valid and column in df_fixed.columns:
                try:
                    if result.data_type == DataType.DATETIME and result.detected_format:
                        if result.detected_format == 'timestamp':
                            df_fixed[column] = pd.to_datetime(df_fixed[column], unit='s', errors='coerce')
                        elif result.detected_format == 'excel_serial':
                            df_fixed[column] = pd.to_datetime(df_fixed[column], unit='D', origin='1899-12-30', errors='coerce')
                        else:
                            df_fixed[column] = pd.to_datetime(df_fixed[column], errors='coerce')
                        
                        logger.info(f"Auto-fixed datetime column: {column}")
                    
                    elif result.data_type == DataType.NUMERIC:
                        df_fixed[column] = pd.to_numeric(df_fixed[column], errors='coerce')
                        logger.info(f"Auto-fixed numeric column: {column}")
                
                except Exception as e:
                    logger.warning(f"Auto-fix failed for column {column}: {e}")
        
        return df_fixed
    
    def generate_validation_report(self, validation_results: Dict[str, ValidationResult]) -> Dict[str, Any]:
        """Doğrulama raporu oluştur."""
        total_columns = len(validation_results)
        valid_columns = sum(1 for r in validation_results.values() if r.is_valid)
        
        issues_by_type = {}
        for result in validation_results.values():
            for issue in result.issues:
                issue_type = issue.split(':')[0] if ':' in issue else issue
                issues_by_type[issue_type] = issues_by_type.get(issue_type, 0) + 1
        
        data_type_distribution = {}
        for result in validation_results.values():
            dt = result.data_type.value
            data_type_distribution[dt] = data_type_distribution.get(dt, 0) + 1
        
        return {
            'summary': {
                'total_columns': total_columns,
                'valid_columns': valid_columns,
                'invalid_columns': total_columns - valid_columns,
                'validation_success_rate': (valid_columns / total_columns) * 100
            },
            'data_type_distribution': data_type_distribution,
            'common_issues': issues_by_type,
            'recommendations': self._generate_global_recommendations(validation_results)
        }
    
    def _generate_global_recommendations(self, validation_results: Dict[str, ValidationResult]) -> List[str]:
        """Global öneriler oluştur."""
        recommendations = []
        
        # Datetime sorunları
        datetime_issues = sum(1 for r in validation_results.values() 
                            if r.data_type == DataType.DATETIME and not r.is_valid)
        if datetime_issues > 0:
            recommendations.append(f"{datetime_issues} kolonda datetime format sorunu var. Import dialog'da format ayarlarını kontrol edin.")
        
        # Mixed type sorunları
        mixed_issues = sum(1 for r in validation_results.values() 
                         if r.data_type == DataType.MIXED)
        if mixed_issues > 0:
            recommendations.append(f"{mixed_issues} kolonda karışık veri tipleri var. Veri temizleme gerekli.")
        
        # Yüksek null oranı
        high_null_columns = sum(1 for r in validation_results.values() 
                              if r.statistics['null_percentage'] > 50)
        if high_null_columns > 0:
            recommendations.append(f"{high_null_columns} kolonda yüksek null oranı var. Veri kaynağını kontrol edin.")
        
        return recommendations
