#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Özellik Kararlılık Takip Sistemi
================================

Bu modül, uygulamanın özelliklerinin kararlılığını takip eder ve
"eskiden çalışan ama şimdi bozuk" durumlarını önlemek için
sürekli monitoring sağlar.
"""

import os
import sys
import json
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum
import traceback
import threading
import pickle

# PyQt5 import'ları (varsa)
try:
    from PyQt5.QtCore import QTimer, QObject, pyqtSignal
    from PyQt5.QtWidgets import QApplication
    PYQT_AVAILABLE = True
except ImportError:
    PYQT_AVAILABLE = False
    QObject = object
    def pyqtSignal(*args, **kwargs):
        return None

logger = logging.getLogger(__name__)

class StabilityStatus(Enum):
    """Kararlılık durumu."""
    STABLE = "stable"
    UNSTABLE = "unstable"
    BROKEN = "broken"
    UNKNOWN = "unknown"
    RECOVERING = "recovering"

class FeatureType(Enum):
    """Özellik tipi."""
    UI_COMPONENT = "ui_component"
    DATA_PROCESSING = "data_processing"
    FILE_OPERATION = "file_operation"
    VISUALIZATION = "visualization"
    CALCULATION = "calculation"
    SYSTEM_INTEGRATION = "system_integration"

@dataclass
class FeatureTest:
    """Özellik testi tanımı."""
    feature_name: str
    feature_type: FeatureType
    test_function: Callable
    expected_behavior: str
    dependencies: List[str]
    timeout_seconds: int = 30
    critical: bool = False
    auto_fix_function: Optional[Callable] = None

@dataclass
class StabilitySnapshot:
    """Kararlılık anlık görüntüsü."""
    feature_name: str
    timestamp: datetime
    status: StabilityStatus
    test_result: Dict[str, Any]
    performance_metrics: Dict[str, float]
    error_details: Optional[str] = None
    fix_attempted: bool = False
    fix_successful: bool = False

class FeatureStabilityTracker(QObject if PYQT_AVAILABLE else object):
    """Özellik kararlılık takipçisi."""
    
    # Sinyaller (PyQt5 varsa)
    if PYQT_AVAILABLE:
        feature_status_changed = pyqtSignal(str, str)  # feature_name, status
        instability_detected = pyqtSignal(str, str)    # feature_name, details
        recovery_completed = pyqtSignal(str, bool)     # feature_name, success
    
    def __init__(self, storage_dir: str = "stability_data"):
        if PYQT_AVAILABLE:
            super().__init__()
        
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        
        self.snapshots_file = self.storage_dir / "stability_snapshots.json"
        self.features_file = self.storage_dir / "feature_definitions.json"
        
        self.feature_tests: Dict[str, FeatureTest] = {}
        self.snapshots: Dict[str, List[StabilitySnapshot]] = {}
        self.monitoring_active = False
        self.monitoring_thread = None
        
        # Built-in feature tests'leri kaydet
        self._register_builtin_features()
        
        # Verileri yükle
        self._load_data()
    
    def _register_builtin_features(self):
        """Built-in özellik testlerini kaydet."""
        
        # Static Limits özelliği
        self.feature_tests['static_limits'] = FeatureTest(
            feature_name='static_limits',
            feature_type=FeatureType.UI_COMPONENT,
            test_function=self._test_static_limits,
            expected_behavior='Static limits panel should create and configure limits',
            dependencies=['static_limits_panel', 'PyQt5'],
            critical=True,
            auto_fix_function=self._fix_static_limits
        )
        
        # Deviation özelliği
        self.feature_tests['deviation_analysis'] = FeatureTest(
            feature_name='deviation_analysis',
            feature_type=FeatureType.CALCULATION,
            test_function=self._test_deviation_analysis,
            expected_behavior='Deviation analysis should calculate trends and fluctuations',
            dependencies=['basic_deviation_panel', 'numpy'],
            critical=True,
            auto_fix_function=self._fix_deviation_analysis
        )
        
        # Graph rendering özelliği
        self.feature_tests['graph_rendering'] = FeatureTest(
            feature_name='graph_rendering',
            feature_type=FeatureType.VISUALIZATION,
            test_function=self._test_graph_rendering,
            expected_behavior='Graphs should render data correctly',
            dependencies=['graph_renderer', 'pyqtgraph'],
            critical=True
        )
        
        # Data loading özelliği
        self.feature_tests['data_loading'] = FeatureTest(
            feature_name='data_loading',
            feature_type=FeatureType.FILE_OPERATION,
            test_function=self._test_data_loading,
            expected_behavior='Data files should load without errors',
            dependencies=['polars'],
            critical=True,
            auto_fix_function=self._fix_data_loading
        )
        
        # Signal processing özelliği
        self.feature_tests['signal_processing'] = FeatureTest(
            feature_name='signal_processing',
            feature_type=FeatureType.DATA_PROCESSING,
            test_function=self._test_signal_processing,
            expected_behavior='Signal processing should handle data transformations',
            dependencies=['signal_processor', 'numpy'],
            critical=False
        )
    
    def _load_data(self):
        """Kayıtlı verileri yükle."""
        try:
            if self.snapshots_file.exists():
                with open(self.snapshots_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for feature_name, snapshots_data in data.items():
                        self.snapshots[feature_name] = [
                            StabilitySnapshot(
                                feature_name=s['feature_name'],
                                timestamp=datetime.fromisoformat(s['timestamp']),
                                status=StabilityStatus(s['status']),
                                test_result=s['test_result'],
                                performance_metrics=s['performance_metrics'],
                                error_details=s.get('error_details'),
                                fix_attempted=s.get('fix_attempted', False),
                                fix_successful=s.get('fix_successful', False)
                            ) for s in snapshots_data
                        ]
        except Exception as e:
            logger.error(f"Error loading stability data: {e}")
    
    def _save_data(self):
        """Verileri kaydet."""
        try:
            snapshots_data = {}
            for feature_name, snapshots in self.snapshots.items():
                snapshots_data[feature_name] = [
                    {
                        **asdict(snapshot),
                        'timestamp': snapshot.timestamp.isoformat(),
                        'status': snapshot.status.value
                    } for snapshot in snapshots
                ]
            
            with open(self.snapshots_file, 'w', encoding='utf-8') as f:
                json.dump(snapshots_data, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            logger.error(f"Error saving stability data: {e}")
    
    def test_feature(self, feature_name: str) -> StabilitySnapshot:
        """Tek bir özelliği test et."""
        if feature_name not in self.feature_tests:
            raise ValueError(f"Unknown feature: {feature_name}")
        
        feature_test = self.feature_tests[feature_name]
        start_time = time.time()
        
        try:
            # Test'i çalıştır
            test_result = feature_test.test_function()
            execution_time = time.time() - start_time
            
            # Sonucu değerlendir
            status = self._evaluate_test_result(test_result, feature_test)
            
            # Snapshot oluştur
            snapshot = StabilitySnapshot(
                feature_name=feature_name,
                timestamp=datetime.now(),
                status=status,
                test_result=test_result,
                performance_metrics={'execution_time': execution_time},
                error_details=test_result.get('error') if isinstance(test_result, dict) else None
            )
            
            # Auto-fix denemesi (eğer bozuksa ve auto-fix varsa)
            if status == StabilityStatus.BROKEN and feature_test.auto_fix_function:
                logger.info(f"Attempting auto-fix for {feature_name}")
                snapshot.fix_attempted = True
                
                try:
                    fix_result = feature_test.auto_fix_function()
                    if fix_result.get('success', False):
                        snapshot.fix_successful = True
                        snapshot.status = StabilityStatus.RECOVERING
                        logger.info(f"Auto-fix successful for {feature_name}")
                    else:
                        logger.warning(f"Auto-fix failed for {feature_name}: {fix_result.get('error', 'Unknown error')}")
                except Exception as fix_error:
                    logger.error(f"Auto-fix error for {feature_name}: {fix_error}")
            
            # Snapshot'ı kaydet
            if feature_name not in self.snapshots:
                self.snapshots[feature_name] = []
            
            self.snapshots[feature_name].append(snapshot)
            
            # Son 50 snapshot'ı tut
            if len(self.snapshots[feature_name]) > 50:
                self.snapshots[feature_name] = self.snapshots[feature_name][-50:]
            
            self._save_data()
            
            # Sinyal gönder (PyQt5 varsa)
            if PYQT_AVAILABLE and hasattr(self, 'feature_status_changed'):
                self.feature_status_changed.emit(feature_name, status.value)
                
                if status == StabilityStatus.BROKEN:
                    self.instability_detected.emit(feature_name, snapshot.error_details or "Unknown error")
                elif snapshot.fix_successful:
                    self.recovery_completed.emit(feature_name, True)
            
            return snapshot
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_details = f"{type(e).__name__}: {str(e)}"
            
            snapshot = StabilitySnapshot(
                feature_name=feature_name,
                timestamp=datetime.now(),
                status=StabilityStatus.BROKEN,
                test_result={'error': error_details, 'traceback': traceback.format_exc()},
                performance_metrics={'execution_time': execution_time},
                error_details=error_details
            )
            
            if feature_name not in self.snapshots:
                self.snapshots[feature_name] = []
            
            self.snapshots[feature_name].append(snapshot)
            self._save_data()
            
            if PYQT_AVAILABLE and hasattr(self, 'instability_detected'):
                self.instability_detected.emit(feature_name, error_details)
            
            return snapshot
    
    def _evaluate_test_result(self, test_result: Any, feature_test: FeatureTest) -> StabilityStatus:
        """Test sonucunu değerlendir."""
        if isinstance(test_result, dict):
            if test_result.get('error'):
                return StabilityStatus.BROKEN
            elif test_result.get('success', True):
                return StabilityStatus.STABLE
            else:
                return StabilityStatus.UNSTABLE
        elif isinstance(test_result, bool):
            return StabilityStatus.STABLE if test_result else StabilityStatus.BROKEN
        else:
            return StabilityStatus.UNKNOWN
    
    def test_all_features(self) -> Dict[str, StabilitySnapshot]:
        """Tüm özellikleri test et."""
        results = {}
        
        for feature_name in self.feature_tests:
            logger.info(f"Testing feature: {feature_name}")
            results[feature_name] = self.test_feature(feature_name)
        
        return results
    
    def get_stability_report(self) -> Dict[str, Any]:
        """Kararlılık raporu oluştur."""
        report = {
            'timestamp': datetime.now().isoformat(),
            'total_features': len(self.feature_tests),
            'feature_status': {},
            'stability_summary': {
                'stable': 0,
                'unstable': 0,
                'broken': 0,
                'unknown': 0,
                'recovering': 0
            },
            'critical_issues': [],
            'recent_instabilities': [],
            'recommendations': []
        }
        
        # Her özellik için son durum
        for feature_name, feature_test in self.feature_tests.items():
            latest_snapshot = self.get_latest_snapshot(feature_name)
            
            if latest_snapshot:
                status = latest_snapshot.status.value
                report['feature_status'][feature_name] = {
                    'status': status,
                    'last_tested': latest_snapshot.timestamp.isoformat(),
                    'critical': feature_test.critical,
                    'performance': latest_snapshot.performance_metrics,
                    'error': latest_snapshot.error_details
                }
                
                # Özet istatistikleri güncelle
                report['stability_summary'][status] += 1
                
                # Kritik sorunları topla
                if feature_test.critical and latest_snapshot.status == StabilityStatus.BROKEN:
                    report['critical_issues'].append({
                        'feature': feature_name,
                        'error': latest_snapshot.error_details,
                        'timestamp': latest_snapshot.timestamp.isoformat()
                    })
            else:
                report['feature_status'][feature_name] = {
                    'status': 'not_tested',
                    'critical': feature_test.critical
                }
                report['stability_summary']['unknown'] += 1
        
        # Son 24 saatteki instabiliteler
        cutoff_time = datetime.now() - timedelta(hours=24)
        for feature_name, snapshots in self.snapshots.items():
            recent_broken = [s for s in snapshots if s.timestamp >= cutoff_time and s.status == StabilityStatus.BROKEN]
            if recent_broken:
                report['recent_instabilities'].extend([
                    {
                        'feature': feature_name,
                        'timestamp': s.timestamp.isoformat(),
                        'error': s.error_details
                    } for s in recent_broken
                ])
        
        # Öneriler oluştur
        report['recommendations'] = self._generate_recommendations(report)
        
        return report
    
    def _generate_recommendations(self, report: Dict[str, Any]) -> List[str]:
        """Öneriler oluştur."""
        recommendations = []
        
        summary = report['stability_summary']
        
        if summary['broken'] > 0:
            recommendations.append(f"🚨 {summary['broken']} özellik bozuk! Acil müdahale gerekli.")
        
        if summary['unstable'] > 0:
            recommendations.append(f"⚠️ {summary['unstable']} özellik kararsız. İzleme altında tutun.")
        
        if len(report['critical_issues']) > 0:
            recommendations.append(f"❌ {len(report['critical_issues'])} kritik özellik çalışmıyor!")
        
        if len(report['recent_instabilities']) > 5:
            recommendations.append("📈 Son 24 saatte çok fazla instability var. Sistem gözden geçirilmeli.")
        
        # Özellik tipine göre öneriler
        broken_features = [name for name, status in report['feature_status'].items() 
                          if status.get('status') == 'broken']
        
        ui_issues = [name for name in broken_features if self.feature_tests[name].feature_type == FeatureType.UI_COMPONENT]
        if ui_issues:
            recommendations.append(f"🖥️ UI bileşenlerinde sorun: {', '.join(ui_issues)}")
        
        data_issues = [name for name in broken_features if self.feature_tests[name].feature_type == FeatureType.DATA_PROCESSING]
        if data_issues:
            recommendations.append(f"📊 Veri işleme sorunları: {', '.join(data_issues)}")
        
        if not recommendations:
            recommendations.append("✅ Tüm özellikler stabil!")
        
        return recommendations
    
    def get_latest_snapshot(self, feature_name: str) -> Optional[StabilitySnapshot]:
        """Bir özelliğin son snapshot'ını al."""
        if feature_name in self.snapshots and self.snapshots[feature_name]:
            return self.snapshots[feature_name][-1]
        return None
    
    def get_feature_history(self, feature_name: str, days: int = 7) -> List[StabilitySnapshot]:
        """Bir özelliğin geçmişini al."""
        if feature_name not in self.snapshots:
            return []
        
        cutoff_date = datetime.now() - timedelta(days=days)
        return [s for s in self.snapshots[feature_name] if s.timestamp >= cutoff_date]
    
    def detect_regressions(self) -> List[Dict[str, Any]]:
        """Regression'ları tespit et."""
        regressions = []
        
        for feature_name, snapshots in self.snapshots.items():
            if len(snapshots) < 2:
                continue
            
            latest = snapshots[-1]
            previous = snapshots[-2]
            
            # Durum kötüleşmesi
            if (previous.status in [StabilityStatus.STABLE, StabilityStatus.UNSTABLE] and 
                latest.status == StabilityStatus.BROKEN):
                
                regressions.append({
                    'feature': feature_name,
                    'type': 'status_regression',
                    'previous_status': previous.status.value,
                    'current_status': latest.status.value,
                    'timestamp': latest.timestamp.isoformat(),
                    'error': latest.error_details,
                    'critical': self.feature_tests[feature_name].critical
                })
            
            # Performans regresyonu
            if (previous.status == StabilityStatus.STABLE and latest.status == StabilityStatus.STABLE):
                prev_time = previous.performance_metrics.get('execution_time', 0)
                curr_time = latest.performance_metrics.get('execution_time', 0)
                
                if prev_time > 0 and curr_time > prev_time * 2:  # %100 yavaşlama
                    regressions.append({
                        'feature': feature_name,
                        'type': 'performance_regression',
                        'previous_time': prev_time,
                        'current_time': curr_time,
                        'slowdown_factor': curr_time / prev_time,
                        'timestamp': latest.timestamp.isoformat()
                    })
        
        return regressions
    
    def start_continuous_monitoring(self, interval_minutes: int = 30):
        """Sürekli monitoring başlat."""
        if self.monitoring_active:
            logger.warning("Monitoring already active")
            return
        
        self.monitoring_active = True
        
        def monitoring_loop():
            while self.monitoring_active:
                try:
                    logger.info("Running scheduled stability check...")
                    results = self.test_all_features()
                    
                    # Kritik sorunları kontrol et
                    critical_broken = [name for name, snapshot in results.items() 
                                     if self.feature_tests[name].critical and snapshot.status == StabilityStatus.BROKEN]
                    
                    if critical_broken:
                        logger.critical(f"CRITICAL FEATURES BROKEN: {', '.join(critical_broken)}")
                    
                    # Regression kontrolü
                    regressions = self.detect_regressions()
                    if regressions:
                        logger.warning(f"REGRESSIONS DETECTED: {len(regressions)}")
                        for regression in regressions:
                            logger.warning(f"- {regression['feature']}: {regression['type']}")
                    
                except Exception as e:
                    logger.error(f"Monitoring cycle error: {e}")
                
                # Bekleme
                time.sleep(interval_minutes * 60)
        
        self.monitoring_thread = threading.Thread(target=monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        logger.info(f"Continuous monitoring started (interval: {interval_minutes} minutes)")
    
    def stop_continuous_monitoring(self):
        """Sürekli monitoring durdur."""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        
        logger.info("Continuous monitoring stopped")
    
    # Test metodları
    def _test_static_limits(self) -> Dict[str, Any]:
        """Static limits testi."""
        try:
            from static_limits_panel import StaticLimitsPanel
            
            test_signals = ['test_signal_1', 'test_signal_2']
            panel = StaticLimitsPanel(test_signals)
            
            # Temel işlevsellik testi
            success = (panel is not None and 
                      len(panel.all_signals) == len(test_signals) and
                      hasattr(panel, 'limit_configs'))
            
            return {
                'success': success,
                'panel_created': panel is not None,
                'signals_loaded': len(panel.all_signals) == len(test_signals),
                'ui_elements': hasattr(panel, 'limit_configs')
            }
            
        except ImportError:
            return {'success': False, 'error': 'StaticLimitsPanel import failed'}
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _test_deviation_analysis(self) -> Dict[str, Any]:
        """Deviation analysis testi."""
        try:
            from basic_deviation_panel import BasicDeviationPanel
            import numpy as np
            
            test_signals = ['test_signal']
            panel = BasicDeviationPanel(test_signals)
            
            # Test verisi ile hesaplama
            test_data = np.random.normal(0, 1, 1000)
            result = panel.calculate_deviation_for_signal(test_data, 'test_signal')
            
            success = (isinstance(result, dict) and 
                      'deviations' in result and
                      'bands' in result and
                      'alerts' in result)
            
            return {
                'success': success,
                'panel_created': panel is not None,
                'calculation_works': isinstance(result, dict),
                'has_required_keys': 'deviations' in result if isinstance(result, dict) else False
            }
            
        except ImportError:
            return {'success': False, 'error': 'BasicDeviationPanel import failed'}
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _test_graph_rendering(self) -> Dict[str, Any]:
        """Graph rendering testi."""
        try:
            from graph_renderer import GraphRenderer
            from signal_processor import SignalProcessor
            import numpy as np
            
            # Mock data
            signal_processor = SignalProcessor()
            graph_signal_mapping = {0: {0: ['test_signal']}}
            
            renderer = GraphRenderer(signal_processor, graph_signal_mapping)
            
            success = renderer is not None
            
            return {
                'success': success,
                'renderer_created': renderer is not None,
                'has_methods': hasattr(renderer, 'apply_segmented_filter')
            }
            
        except ImportError:
            return {'success': False, 'error': 'GraphRenderer import failed'}
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _test_data_loading(self) -> Dict[str, Any]:
        """Data loading testi."""
        try:
            import polars as pl
            import tempfile
            import os
            
            # Test CSV oluştur
            test_data = {'a': [1, 2, 3], 'b': [4, 5, 6]}
            test_csv_path = 'temp_test.csv'
            pl.DataFrame(test_data).write_csv(test_csv_path)
            
            # CSV yükleme
            loaded_data = pl.read_csv(test_csv_path)
            os.remove(test_csv_path)
            
            success = (len(loaded_data) == 3 and 
                      loaded_data.height == 3)
            
            return {
                'success': success,
                'csv_loaded': len(loaded_data) == 3,
                'vaex_conversion': loaded_data.height == 3 # polars height is equivalent to vaex length
            }
                
        except ImportError:
            return {'success': False, 'error': 'Data loading libraries import failed'}
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _test_signal_processing(self) -> Dict[str, Any]:
        """Signal processing testi."""
        try:
            from signal_processor import SignalProcessor
            import numpy as np
            import polars as pl
            
            processor = SignalProcessor()
            
            # Test data
            test_data = pl.DataFrame({
                'time': np.linspace(0, 10, 100),
                'signal': np.sin(np.linspace(0, 10, 100))
            })
            
            processor.set_data(test_data)
            signals = processor.get_all_signals()
            
            success = len(signals) > 0
            
            return {
                'success': success,
                'processor_created': processor is not None,
                'data_loaded': len(signals) > 0,
                'signal_count': len(signals)
            }
            
        except ImportError:
            return {'success': False, 'error': 'SignalProcessor import failed'}
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    # Auto-fix metodları
    def _fix_static_limits(self) -> Dict[str, Any]:
        """Static limits auto-fix."""
        try:
            # Basit fix: Import'u kontrol et ve yeniden dene
            import importlib
            
            try:
                import static_limits_panel
                importlib.reload(static_limits_panel)
                
                # Test et
                test_result = self._test_static_limits()
                return {'success': test_result.get('success', False), 'method': 'module_reload'}
                
            except Exception as e:
                return {'success': False, 'error': f'Reload failed: {e}'}
                
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _fix_deviation_analysis(self) -> Dict[str, Any]:
        """Deviation analysis auto-fix."""
        try:
            import importlib
            
            try:
                import basic_deviation_panel
                importlib.reload(basic_deviation_panel)
                
                test_result = self._test_deviation_analysis()
                return {'success': test_result.get('success', False), 'method': 'module_reload'}
                
            except Exception as e:
                return {'success': False, 'error': f'Reload failed: {e}'}
                
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _fix_data_loading(self) -> Dict[str, Any]:
        """Data loading auto-fix."""
        try:
            # Geçici dosyaları temizle
            import tempfile
            import glob
            
            temp_dir = tempfile.gettempdir()
            temp_files = glob.glob(os.path.join(temp_dir, "tmp*.csv"))
            
            cleaned_count = 0
            for temp_file in temp_files:
                try:
                    os.unlink(temp_file)
                    cleaned_count += 1
                except:
                    pass
            
            # Test et
            test_result = self._test_data_loading()
            return {
                'success': test_result.get('success', False), 
                'method': 'temp_cleanup',
                'cleaned_files': cleaned_count
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def export_stability_report(self, file_path: str = None) -> str:
        """Kararlılık raporunu dosyaya aktar."""
        if not file_path:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            file_path = f"stability_report_{timestamp}.json"
        
        # Tam test çalıştır
        test_results = self.test_all_features()
        
        # Rapor oluştur
        report = self.get_stability_report()
        
        # Regression analizi
        regressions = self.detect_regressions()
        
        # Birleşik rapor
        full_report = {
            'test_results': {name: asdict(snapshot) for name, snapshot in test_results.items()},
            'stability_report': report,
            'regressions': regressions,
            'export_time': datetime.now().isoformat()
        }
        
        # Datetime'ları string'e çevir
        def convert_datetime(obj):
            if isinstance(obj, dict):
                return {k: convert_datetime(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_datetime(item) for item in obj]
            elif isinstance(obj, datetime):
                return obj.isoformat()
            else:
                return obj
        
        full_report = convert_datetime(full_report)
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(full_report, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Stability report exported to: {file_path}")
            return file_path
            
        except Exception as e:
            logger.error(f"Failed to export stability report: {e}")
            return ""

# CLI interface
def main():
    """Ana fonksiyon."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Feature Stability Tracker')
    parser.add_argument('--test', '-t', help='Test specific feature')
    parser.add_argument('--test-all', action='store_true', help='Test all features')
    parser.add_argument('--monitor', '-m', type=int, help='Start continuous monitoring (interval in minutes)')
    parser.add_argument('--export', '-e', help='Export stability report to file')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    # Logging seviyesini ayarla
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.WARNING)  # Production mode: sadece WARNING+
    
    tracker = FeatureStabilityTracker()
    
    if args.test:
        print(f"🧪 Testing feature: {args.test}")
        snapshot = tracker.test_feature(args.test)
        
        print(f"Status: {snapshot.status.value}")
        print(f"Execution time: {snapshot.performance_metrics.get('execution_time', 0):.3f}s")
        if snapshot.error_details:
            print(f"Error: {snapshot.error_details}")
        if snapshot.fix_attempted:
            print(f"Auto-fix attempted: {'Success' if snapshot.fix_successful else 'Failed'}")
    
    if args.test_all:
        print("🧪 Testing all features...")
        results = tracker.test_all_features()
        
        print(f"\n📊 Test Results:")
        for feature_name, snapshot in results.items():
            status_emoji = {"stable": "✅", "unstable": "⚠️", "broken": "❌", "unknown": "❓", "recovering": "🔄"}
            emoji = status_emoji.get(snapshot.status.value, "❓")
            print(f"{emoji} {feature_name}: {snapshot.status.value}")
        
        # Özet rapor
        report = tracker.get_stability_report()
        summary = report['stability_summary']
        print(f"\n📈 Summary:")
        print(f"Stable: {summary['stable']}")
        print(f"Unstable: {summary['unstable']}")
        print(f"Broken: {summary['broken']}")
        print(f"Recovering: {summary['recovering']}")
        
        if report['critical_issues']:
            print(f"\n🚨 Critical Issues:")
            for issue in report['critical_issues']:
                print(f"- {issue['feature']}: {issue['error']}")
    
    if args.monitor:
        print(f"🔄 Starting continuous monitoring ({args.monitor} minutes interval)...")
        tracker.start_continuous_monitoring(args.monitor)
        
        try:
            print("Monitoring active. Press Ctrl+C to stop.")
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nStopping monitoring...")
            tracker.stop_continuous_monitoring()
    
    if args.export:
        print("📋 Exporting stability report...")
        report_file = tracker.export_stability_report(args.export)
        if report_file:
            print(f"✅ Report exported: {report_file}")

if __name__ == "__main__":
    main()
