"""
Project Operations Module

MPAI proje dosyası (.mpai) işlemlerini yöneten fonksiyonlar.

Bu modül şunları içerir:
- Proje kaydetme (.mpai formatı)
- Proje açma (.mpai formatı)
- Proje validasyonu

Refactored from: app.py (TimeGraphApp sınıfı)
"""

import logging
import os
from typing import Optional, Dict, Any
from datetime import datetime
import polars as pl
from PyQt5.QtWidgets import QFileDialog, QMessageBox
from PyQt5.QtCore import QTimer

logger = logging.getLogger(__name__)


class ProjectOperations:
    """
    MPAI proje dosyası işlemlerini yöneten sınıf.
    TimeGraphApp'den ayrıştırılmış, bağımsız çalışabilen modül.
    """
    
    def __init__(self, main_window):
        """
        Args:
            main_window: TimeGraphApp ana pencere referansı
        """
        self.main_window = main_window
        
    def save_project_dialog(self):
        """
        Proje kaydetme dialog'unu göster ve kaydetme işlemini başlat.
        
        Refactored from: app.py -> TimeGraphApp._on_project_save()
        
        Returns:
            bool: İşlem başarılı ise True
        """
        active_widget = self.main_window.widget_container_manager.get_active_widget()
        if not active_widget:
            QMessageBox.warning(
                self.main_window,
                "Proje Kaydedilemedi",
                "Kaydedilecek bir veri yok. Önce bir dosya yükleyin."
            )
            return False
        
        # Check if data exists
        if not hasattr(active_widget, 'data_manager'):
            QMessageBox.warning(
                self.main_window,
                "Proje Kaydedilemedi",
                "Veri yöneticisi bulunamadı."
            )
            return False
        
        # Get dataframe from data manager
        dataframe = active_widget.data_manager.get_data()
<<<<<<< HEAD
        
        # CHECK: If we have a reader, we don't strictly need the dataframe populated in RAM
        has_reader = hasattr(active_widget.data_manager, 'reader') and active_widget.data_manager.reader is not None
        
        if (dataframe is None or (hasattr(dataframe, 'height') and dataframe.height == 0)) and not has_reader:
=======
        if dataframe is None or (hasattr(dataframe, 'height') and dataframe.height == 0):
>>>>>>> a00000f060d03177d5efc0e2a3c7d946dd33992b
            QMessageBox.warning(
                self.main_window,
                "Proje Kaydedilemedi",
                "Kaydedilecek veri bulunamadı."
            )
            return False
        
        try:
            # Get current file info
            active_file_index = self.main_window.file_manager.active_file_index
            file_data = self.main_window.file_manager.get_file_data(active_file_index)
            
            # Add calculated parameters to dataframe
<<<<<<< HEAD
            # CRITICAL FIX: Only attempt this if we successfully loaded a DataFrame into RAM
            # For large datasets (MpaiDirectoryReader), dataframe might be a Reader object or None
            if isinstance(dataframe, pl.DataFrame) and hasattr(active_widget, 'signal_processor'):
=======
            if hasattr(active_widget, 'signal_processor'):
>>>>>>> a00000f060d03177d5efc0e2a3c7d946dd33992b
                all_signals = active_widget.signal_processor.get_all_signals()
                if all_signals:
                    # Get time column
                    time_col = file_data.get('time_column', '') if file_data else None
                    if not time_col and hasattr(active_widget.data_manager, 'time_column'):
                        time_col = active_widget.data_manager.time_column
                    
                    # Get time data from first signal or dataframe
                    time_data = None
                    if time_col and time_col in dataframe.columns:
                        time_data = dataframe.get_column(time_col).to_numpy()
                    elif all_signals:
                        first_signal = list(all_signals.values())[0]
                        if 'x_data' in first_signal:
                            time_data = first_signal['x_data']
                    
                    if time_data is not None:
                        # Add calculated parameters
                        calculated_params = {}
                        original_columns = set(dataframe.columns)
                        
                        for signal_name, signal_data in all_signals.items():
                            if signal_name not in original_columns and signal_name != time_col:
                                if 'y_data' in signal_data:
                                    y_data = signal_data['y_data']
                                    if len(y_data) == len(time_data):
                                        calculated_params[signal_name] = y_data
                        
                        if calculated_params:
                            import numpy as np
                            for param_name, param_data in calculated_params.items():
                                dataframe = dataframe.with_columns(
                                    pl.Series(param_name, param_data)
                                )
                            logger.info(f"Added {len(calculated_params)} calculated parameters")
            
            # Suggest filename
            suggested_name = "project.mpai"
            if file_data:
                original_name = file_data.get('filename', 'project')
                base_name = os.path.splitext(original_name)[0]
                suggested_name = f"{base_name}.mpai"
            
            # File dialog
            file_path, _ = QFileDialog.getSaveFileName(
                self.main_window,
                "Proje Kaydet (.mpai)",
                suggested_name,
                "MPAI Project Files (*.mpai)"
            )
            
            if not file_path:
                return False
            
            # Show loading overlay
            if hasattr(self.main_window, 'loading_manager') and self.main_window.loading_manager:
                self.main_window.loading_manager.start_operation("project_save", "Proje kaydediliyor...")
            
            # Get layout config
            layout_config = active_widget.get_layout_config()
            
            # Prepare metadata
            metadata = {
                'original_file': file_data.get('filename', 'unknown') if file_data else 'unknown',
                'original_file_path': file_data.get('file_path', '') if file_data else '',
                'time_column': file_data.get('time_column', '') if file_data else '',
                'saved_date': datetime.now().isoformat(),
            }
            
            # Save project
<<<<<<< HEAD
            # DETECT DATA SOURCE TYPE
            # DETECT DATA SOURCE TYPE
            # Check if raw_data is an MpaiDirectoryReader (not a DataFrame)
            raw_data = active_widget.data_manager.raw_data if hasattr(active_widget.data_manager, 'raw_data') else None
            is_directory_reader = raw_data is not None and hasattr(raw_data, 'mpai_path')
            
            if is_directory_reader:
                # This is a large dataset managed by MpaiDirectoryReader
                # We should package the existing directory instead of trying to save a dataframe
                source_path = str(raw_data.mpai_path)
                logger.info(f"Detected large dataset project (Director Mode). Source: {source_path}")
                
                # ✅ NEW: Include calculated parameters in the project save
                # Copy calculated param directories into source before packaging
                calc_params_copied = 0
                if hasattr(active_widget, 'signal_processor'):
                    sp = active_widget.signal_processor
                    calc_readers = getattr(sp, '_calc_param_readers', {})
                    
                    logger.info(f"[SAVE] Checking for calculated params. Found {len(calc_readers)} readers")
                    
                    if calc_readers:
                        import shutil
                        for param_name, reader in calc_readers.items():
                            logger.info(f"[SAVE] Processing calc param: {param_name}")
                            
                            # Get the mpai_path - could be Path or string
                            calc_path = getattr(reader, 'mpai_path', None)
                            if calc_path is None:
                                logger.warning(f"[SAVE] Calc param '{param_name}' has no mpai_path")
                                continue
                                
                            calc_source = str(calc_path)
                            logger.info(f"[SAVE] Calc param source: {calc_source}")
                            
                            # Create a subdirectory for calculated params
                            calc_dest = os.path.join(source_path, f"calculated_{param_name}")
                            logger.info(f"[SAVE] Calc param dest: {calc_dest}")
                            
                            # Copy the calculated param MPAI directory
                            if os.path.exists(calc_source):
                                if calc_source != source_path:
                                    try:
                                        if os.path.exists(calc_dest):
                                            shutil.rmtree(calc_dest)
                                        shutil.copytree(calc_source, calc_dest)
                                        calc_params_copied += 1
                                        logger.info(f"[SAVE] ✓ Copied calculated param '{param_name}' to project")
                                    except Exception as e:
                                        logger.error(f"[SAVE] ✗ Failed to copy calc param '{param_name}': {e}", exc_info=True)
                                else:
                                    logger.warning(f"[SAVE] Skipping calc param - source equals dest")
                            else:
                                logger.warning(f"[SAVE] Calc param source doesn't exist: {calc_source}")
                        
                        if calc_params_copied > 0:
                            logger.info(f"[SAVE] Included {calc_params_copied} calculated parameters in project")
                        else:
                            logger.warning(f"[SAVE] No calculated parameters were copied!")


                
                success = self.main_window.project_manager.save_directory_project(
                    source_path,
                    file_path,
                    layout_config,
                    metadata
                )

            else:
                # Standard صغير dataset held in RAM or generic DataFrame
                success = self.main_window.project_manager.save_project(
                    file_path,
                    dataframe,
                    layout_config,
                    metadata
                )
=======
            success = self.main_window.project_manager.save_project(
                file_path,
                dataframe,
                layout_config,
                metadata
            )
>>>>>>> a00000f060d03177d5efc0e2a3c7d946dd33992b
            
            # Hide loading overlay
            if hasattr(self.main_window, 'loading_manager') and self.main_window.loading_manager:
                self.main_window.loading_manager.finish_operation("project_save")
            
            if success:
                if hasattr(self.main_window, 'status_bar'):
                    self.main_window.status_bar.showMessage(
                        f"✅ Proje başarıyla kaydedildi: {os.path.basename(file_path)}", 
                        5000
                    )
                logger.info(f"Project saved: {file_path}")
                
                # Mark as not modified
                if file_data:
                    file_data['is_data_modified'] = False
                    
                return True
            else:
                QMessageBox.critical(
                    self.main_window,
                    "Proje Kaydetme Hatası",
                    "Proje kaydedilemedi. Lütfen log dosyasını kontrol edin."
                )
                return False
                
        except Exception as e:
            error_msg = f"Proje kaydedilemedi: {str(e)}"
            logger.error(error_msg, exc_info=True)
            
            # Hide loading overlay safely
            try:
                if hasattr(self.main_window, 'loading_manager') and self.main_window.loading_manager:
                    self.main_window.loading_manager.finish_operation("project_save")
            except:
                pass
            
            QMessageBox.critical(
                self.main_window,
                "Proje Kaydetme Hatası",
                error_msg
            )
            return False
    
    def open_project_dialog(self):
        """
        Proje açma dialog'unu göster ve yükleme işlemini başlat.
        
        Refactored from: app.py -> TimeGraphApp._on_project_open()
        
        Returns:
            bool: İşlem başarılı ise True
        """
        # File dialog
        file_path, _ = QFileDialog.getOpenFileName(
            self.main_window,
            "Proje Aç (.mpai)",
            "",
            "MPAI Project Files (*.mpai)"
        )
        
        if not file_path:
            return False
        
        try:
            # Validate project file
            is_valid, message = self.main_window.project_manager.validate_project(file_path)
            if not is_valid:
                QMessageBox.warning(
                    self.main_window,
                    "Geçersiz Proje Dosyası",
                    f"Proje dosyası geçerli değil:\n\n{message}"
                )
                return False
            
            # Show loading overlay
            if hasattr(self.main_window, 'loading_manager') and self.main_window.loading_manager:
                self.main_window.loading_manager.start_operation("project_load", "Proje yükleniyor...")
            
            # Load project
            project_data = self.main_window.project_manager.load_project(file_path)
            
            if project_data is None:
                if hasattr(self.main_window, 'loading_manager') and self.main_window.loading_manager:
                    self.main_window.loading_manager.finish_operation("project_load")
                QMessageBox.critical(
                    self.main_window,
                    "Proje Yükleme Hatası",
                    "Proje yüklenemedi. Lütfen log dosyasını kontrol edin."
                )
                return False
            
<<<<<<< HEAD
            # Extract data - handle both DataFrame and Reader-based projects
            is_directory_based = project_data.get('is_directory_based', False)
            
            if is_directory_based:
                # Directory-based project has reader instead of dataframe
                reader = project_data.get('reader')
                dataframe = reader  # Store reader as the data source
                data_source = reader
            else:
                dataframe = project_data['dataframe']
                data_source = dataframe
                reader = None
                
=======
            # Extract data
            dataframe = project_data['dataframe']
>>>>>>> a00000f060d03177d5efc0e2a3c7d946dd33992b
            layout_config = project_data['layout_config']
            metadata = project_data['metadata']
            
            # Get filename from metadata
            original_filename = metadata.get('custom', {}).get('original_file', os.path.basename(file_path))
<<<<<<< HEAD
            time_column = metadata.get('time_column', 'time')
=======
            time_column = metadata.get('custom', {}).get('time_column', 'time')
>>>>>>> a00000f060d03177d5efc0e2a3c7d946dd33992b
            
            # Check if file is already open
            file_manager = self.main_window.file_manager
            existing_index = file_manager.is_file_already_open(file_path)
            if existing_index >= 0:
                if file_manager.file_tab_widget:
                    file_manager.file_tab_widget.setCurrentIndex(existing_index)
                if hasattr(self.main_window, 'loading_manager') and self.main_window.loading_manager:
                    self.main_window.loading_manager.finish_operation("project_load")
                if hasattr(self.main_window, 'status_bar'):
                    self.main_window.status_bar.showMessage(f"Proje zaten açık: {original_filename}", 3000)
                return True
            
            # Check file limit
            if not file_manager.can_add_file():
                if hasattr(self.main_window, 'loading_manager') and self.main_window.loading_manager:
                    self.main_window.loading_manager.finish_operation("project_load")
                QMessageBox.warning(
                    self.main_window,
                    "Dosya Limiti",
                    f"Maksimum {file_manager.max_files} dosya aynı anda açık olabilir.\nBir dosyayı kapatıp tekrar deneyin."
                )
                return False
            
            # Add file metadata
            file_metadata = {
                'file_path': file_path,
                'filename': original_filename,
<<<<<<< HEAD
                'df': data_source,  # Store reader or dataframe
                'time_column': time_column,
                'widget': None,
                'is_data_modified': False,
                'is_project_file': True,
                'is_directory_based': is_directory_based
=======
                'df': dataframe,
                'time_column': time_column,
                'widget': None,
                'is_data_modified': False,
                'is_project_file': True
>>>>>>> a00000f060d03177d5efc0e2a3c7d946dd33992b
            }
            
            file_index = file_manager.add_file(file_metadata)
            
            if file_index < 0:
                if hasattr(self.main_window, 'loading_manager') and self.main_window.loading_manager:
                    self.main_window.loading_manager.finish_operation("project_load")
                QMessageBox.warning(
                    self.main_window,
                    "Dosya Eklenemedi",
                    "Dosya listeye eklenemedi."
                )
                return False
            
            # Create new widget for this project
            new_widget = self.main_window.widget_container_manager.create_widget_for_file(file_index)
            
            # Connect widget signals
            if hasattr(self.main_window, '_connect_widget_signals'):
                self.main_window._connect_widget_signals(new_widget)
            
            # Update file metadata with widget
            file_data = file_manager.get_file_data(file_index)
            if file_data:
                file_data['widget'] = new_widget
            
            # Load data into widget
            new_widget.update_data(dataframe, time_column)
            
            # Apply layout after delay (only if layout exists)
            if layout_config is not None:
                QTimer.singleShot(500, lambda: new_widget.set_layout_config(layout_config))
            else:
                logger.info("No layout config to apply (binary MPAI format)")
            
<<<<<<< HEAD
            # ✅ FIXED: Register calculated parameters AFTER data loading completes
            # The signal_processor.clear_all_data() is called during update_data(),
            # so we must register calc params after that completes.
            calc_params = metadata.get('calculated_params', {})
            logger.info(f"[LOAD] Checking for calculated params in metadata: {len(calc_params)} found")
            
            if calc_params:
                # Create a closure to capture calc_params and new_widget
                def register_calc_params_delayed():
                    logger.info(f"[LOAD] Delayed registration of {len(calc_params)} calculated params...")
                    
                    if not hasattr(new_widget, 'signal_processor'):
                        logger.error("[LOAD] signal_processor not available!")
                        return
                    
                    sp = new_widget.signal_processor
                    
                    if not hasattr(sp, '_calc_param_readers'):
                        sp._calc_param_readers = {}
                    
                    registered = []
                    from src.data.data_reader import MpaiDirectoryReader
                    
                    for param_name, param_path in calc_params.items():
                        try:
                            logger.info(f"[LOAD] Loading calc param '{param_name}' from: {param_path}")
                            calc_reader = MpaiDirectoryReader(param_path)
                            row_count = calc_reader.get_row_count()
                            t_min, t_max = calc_reader.get_time_range()
                            
                            # Register exactly like streaming param worker does
                            sp.signal_data[param_name] = {
                                'mpai_reader': calc_reader,
                                'column_name': param_name,
                                'time_column': 'Time',
                                'row_count': row_count,
                                'time_range': (t_min, t_max),
                                'metadata': {
                                    'memory_mapped': True,
                                    'calculated': True,
                                    'mpai': True,
                                    'full_count': row_count,
                                    'full_time_range': (t_min, t_max)
                                }
                            }
                            sp._calc_param_readers[param_name] = calc_reader
                            registered.append(param_name)
                            logger.info(f"[LOAD] ✓ Registered calc param: {param_name} ({row_count:,} rows)")
                                    
                        except Exception as e:
                            logger.error(f"[LOAD] ✗ Failed to load calc param '{param_name}': {e}", exc_info=True)
                    
                    # Add to parameter list UI - try multiple ways to access it
                    if registered:
                        # Correct attribute name is parameters_panel_manager (not params_panel_manager)
                        pm = getattr(new_widget, 'parameters_panel_manager', None)
                        
                        if pm and hasattr(pm, 'parameter_list') and pm.parameter_list:
                            for param_name in registered:
                                pm.parameter_list.add_parameter(param_name)
                                logger.info(f"[LOAD] ✓ Added '{param_name}' to parameter list UI")
                        else:
                            logger.warning(f"[LOAD] parameters_panel_manager not accessible: pm={pm}")
                    
                    # Re-apply layout now that calc params are registered
                    if registered and layout_config is not None:
                        logger.info("[LOAD] Re-applying layout config after calc params registered...")
                        try:
                            new_widget.set_layout_config(layout_config)
                            logger.info("[LOAD] ✓ Layout re-applied successfully")
                        except Exception as e:
                            logger.warning(f"[LOAD] Layout re-apply failed: {e}")
                    
                    logger.info(f"[LOAD] Completed: {len(registered)} calculated parameters registered")
                
                # Delay to run AFTER update_data and set_layout_config complete
                QTimer.singleShot(1500, register_calc_params_delayed)






            
=======
>>>>>>> a00000f060d03177d5efc0e2a3c7d946dd33992b
            # Switch to new widget
            self.main_window.widget_container_manager.switch_to_file_widget(file_index)
            
            # Hide loading overlay
            if hasattr(self.main_window, 'loading_manager') and self.main_window.loading_manager:
                self.main_window.loading_manager.finish_operation("project_load")
            
            # Show success
            if hasattr(self.main_window, 'status_bar'):
                self.main_window.status_bar.showMessage(
                    f"✅ Proje başarıyla yüklendi: {original_filename} (Parquet - Hızlı!)", 
                    5000
                )
            logger.info(f"Project loaded: {file_path}")
            
            return True
                
        except Exception as e:
            error_msg = f"Proje yüklenemedi: {str(e)}"
            logger.error(error_msg, exc_info=True)
            
            # Hide loading overlay safely
            try:
                if hasattr(self.main_window, 'loading_manager') and self.main_window.loading_manager:
                    self.main_window.loading_manager.finish_operation("project_load")
            except:
                pass
            
            QMessageBox.critical(
                self.main_window,
                "Proje Yükleme Hatası",
                error_msg
            )
            return False
