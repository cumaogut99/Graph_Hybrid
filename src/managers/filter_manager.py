"""
Filter Manager - Range filter logic and calculations
"""

import logging
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from PyQt5.QtCore import QObject, pyqtSignal as Signal, QThread, QTimer

from src.managers.cpp_filter_manager import CppFilterCalculationWorker

logger = logging.getLogger(__name__)


class FilterCalculationWorker(QObject):
    """Worker for calculating filter segments in background thread."""
    
    finished = Signal(list)  # Emits calculated segments
    error = Signal(str)
    progress = Signal(int)  # Progress percentage
    
    def __init__(self, all_signals: dict, conditions: list):
        super().__init__()
        # Deep copy to avoid data race conditions
        self.all_signals = {k: {'x_data': v['x_data'], 'y_data': v['y_data']} 
                           for k, v in all_signals.items()}
        self.conditions = [c.copy() for c in conditions]
        self.should_stop = False
        self._is_running = False
    
    def run(self):
        """Calculate filter segments."""
        try:
            self._is_running = True
            
            if self.should_stop:
                return
                
            segments = self._calculate_segments()
            
            if not self.should_stop:
                self.finished.emit(segments)
        except Exception as e:
            logger.error(f"Filter calculation error: {e}")
            if not self.should_stop:
                self.error.emit(str(e))
        finally:
            self._is_running = False
    
    def stop(self):
        """Stop the calculation."""
        self.should_stop = True
    
    def is_running(self) -> bool:
        """Check if worker is currently running."""
        return self._is_running
    
    def _calculate_segments(self) -> list:
        """Calculate time segments that satisfy all filter conditions."""
        if not self.conditions or not self.all_signals:
            return []
        
        # Get time data from first available signal
        time_data = None
        for signal_name, signal_data in self.all_signals.items():
            if 'x_data' in signal_data and len(signal_data['x_data']) > 0:
                time_data = np.array(signal_data['x_data'])
                break
        
        if time_data is None:
            return []
        
        # Create a boolean mask for all time points
        combined_mask = np.ones(len(time_data), dtype=bool)
        
        # Apply each condition with progress reporting
        total_conditions = len(self.conditions)
        for idx, condition in enumerate(self.conditions):
            if self.should_stop:
                return []
            
            # Report progress
            progress = int((idx / total_conditions) * 100)
            self.progress.emit(progress)
                
            param_name = condition['parameter']
            ranges = condition['ranges']
            
            if param_name not in self.all_signals:
                continue
            
            # Use view instead of copy for better performance
            param_data = np.asarray(self.all_signals[param_name]['y_data'])
            condition_mask = np.zeros(len(param_data), dtype=bool)
            
            # Apply all ranges for this parameter (OR logic within parameter)
            for range_filter in ranges:
                if self.should_stop:
                    return []
                    
                range_type = range_filter['type']
                operator = range_filter['operator']
                value = range_filter['value']
                
                if range_type == 'lower':
                    if operator == '>=':
                        range_mask = param_data >= value
                    elif operator == '>':
                        range_mask = param_data > value
                    else:
                        continue
                elif range_type == 'upper':
                    if operator == '<=':
                        range_mask = param_data <= value
                    elif operator == '<':
                        range_mask = param_data < value
                    else:
                        continue
                else:
                    continue
                
                condition_mask |= range_mask
            
            # Combine with overall mask (AND logic between parameters)
            combined_mask &= condition_mask
        
        # Find continuous segments
        segments = self._find_continuous_segments(time_data, combined_mask)
        
        return segments


class PolarsFilterWorker(QObject):
    """Polars-based filter calculation (lazy/streaming-friendly for DataFrame path)."""
    
    finished = Signal(list)
    error = Signal(str)
    progress = Signal(int)
    
    def __init__(self, polars_df, time_column: str, conditions: list):
        super().__init__()
        self.polars_df = polars_df
        self.time_column = time_column
        self.conditions = [c.copy() for c in conditions]
        self.should_stop = False
        self._is_running = False
    
    def run(self):
        try:
            self._is_running = True
            if self.should_stop:
                return
            segments = self._calculate_segments_polars()
            if not self.should_stop:
                self.finished.emit(segments)
        except Exception as e:
            logger.error(f"Polars filter calculation error: {e}")
            if not self.should_stop:
                self.error.emit(str(e))
        finally:
            self._is_running = False
    
    def stop(self):
        self.should_stop = True
    
    def is_running(self) -> bool:
        return self._is_running
    
    def _build_expr(self, pl):
        """Build combined Polars expression from conditions."""
        if not self.conditions:
            return None
        exprs = []
        total = len(self.conditions)
        for idx, condition in enumerate(self.conditions):
            if self.should_stop:
                return None
            self.progress.emit(int((idx / max(total, 1)) * 100))
            param = condition["parameter"]
            ranges = condition["ranges"]
            or_expr = None
            for r in ranges:
                if r["type"] == "lower":
                    if r["operator"] == ">=":
                        part = pl.col(param) >= r["value"]
                    elif r["operator"] == ">":
                        part = pl.col(param) > r["value"]
                    else:
                        continue
                elif r["type"] == "upper":
                    if r["operator"] == "<=":
                        part = pl.col(param) <= r["value"]
                    elif r["operator"] == "<":
                        part = pl.col(param) < r["value"]
                    else:
                        continue
                else:
                    continue
                or_expr = part if or_expr is None else (or_expr | part)
            if or_expr is not None:
                exprs.append(or_expr)
        if not exprs:
            return None
        combined = exprs[0]
        for e in exprs[1:]:
            combined = combined & e
        return combined
    
    def _calculate_segments_polars(self) -> list:
        import polars as pl
        expr = self._build_expr(pl)
        if expr is None:
            return []
        # Lazy filter + select time column
        lf = self.polars_df.lazy().filter(expr).select(pl.col(self.time_column))
        times = lf.collect(streaming=True).get_column(self.time_column).to_numpy()
        if times.size == 0:
            return []
        times_sorted = np.sort(times)
        # Build continuous segments (assumes time increasing)
        segments = []
        start = times_sorted[0]
        prev = times_sorted[0]
        for t in times_sorted[1:]:
            if self.should_stop:
                return segments
            if t - prev > 0:  # time strictly increasing; if gap, close segment
                # Treat any step as continuous; gaps >0 still contiguous in this simplistic check
                # We only split when there is a true gap (same value repeated allowed)
                pass
            # Detect gap bigger than 0? Use equality to split when non-consecutive indices not known.
            if t < prev:  # should not happen after sort
                continue
            # If there is a jump larger than 0 and we want to split on gaps, choose a small epsilon
            if t - prev > 0 and False:
                segments.append((start, prev))
                start = t
            prev = t
        segments.append((start, prev))
        return segments
    
    def _find_continuous_segments(self, time_data: np.ndarray, mask: np.ndarray) -> List[Tuple[float, float]]:
        """Find continuous time segments where mask is True."""
        if not np.any(mask):
            return []
        
        # Find indices where mask is True
        true_indices = np.where(mask)[0]
        
        if len(true_indices) == 0:
            return []
        
        segments = []
        start_idx = true_indices[0]
        
        for i in range(1, len(true_indices)):
            if self.should_stop:
                return segments
                
            # Check if there's a gap
            if true_indices[i] - true_indices[i-1] > 1:
                # End current segment
                end_idx = true_indices[i-1]
                segments.append((time_data[start_idx], time_data[end_idx]))
                start_idx = true_indices[i]
        
        # Add the last segment
        segments.append((time_data[start_idx], time_data[true_indices[-1]]))
        
        return segments


class FilterManager:
    """Manages range filter calculations and operations."""
    
    def __init__(self, parent_widget=None):
        self.active_filters = {}  # {GLOBAL_FILTER_KEY: filter_data} while a filter is applied
        self.filter_applied = False
        self.parent_widget = parent_widget
        
        # Multiple threads support - one thread per calculation
        self.calculation_threads = {}  # {identifier: thread}
        self.calculation_workers = {}  # {identifier: worker}
        self._cleanup_in_progress = False
        
        # Debouncing mechanism to prevent rapid successive calls (e.g., double-click)
        self._last_calculation_time = {}  # Per-graph debouncing
        # TODO: Implement smart debounce based on conditions hash instead of time
        self._calculation_debounce_ms = 0  # DISABLED - was causing legitimate calls to be blocked

    
    def calculate_filter_segments_threaded(self, all_signals: dict, conditions: list, callback=None, tab_index: int = 0, graph_index: int = 0):
        """Calculate time segments that satisfy all filter conditions in background thread."""
        import time
        
        logger.info(f"[FILTER THREADED] Starting calculation for tab={tab_index}, graph={graph_index}")
        logger.info(f"[FILTER THREADED] Conditions count: {len(conditions) if conditions else 0}")
        
        # Create unique identifier for this calculation
        calc_id = f"tab{tab_index}_graph{graph_index}"
        
        current_time = time.time() * 1000  # Convert to milliseconds
        
        # Check if this is a rapid successive call for this specific graph
        last_time = self._last_calculation_time.get(calc_id, 0)
        if current_time - last_time < self._calculation_debounce_ms:
            logger.warning(f"[FILTER THREADED] DEBOUNCE: Skipping rapid call for {calc_id} (elapsed: {current_time - last_time:.1f}ms < {self._calculation_debounce_ms}ms)")
            # ✅ FIX: Don't silently return - call callback with None to signal debounce
            # This allows the caller to hide loading overlay
            if callback:
                callback(None)  # None = debounced, different from [] = no results
            return
        
        # Update last calculation time for this graph
        self._last_calculation_time[calc_id] = current_time
        
        if not conditions or not all_signals:
            logger.info(f"[FILTER THREADED] Empty conditions or signals, returning empty list")
            if callback:
                callback([])
            return
        
        # Stop any existing calculation for this specific graph
        logger.info(f"[FILTER THREADED] Stopping any existing calculation for {calc_id}")
        self._stop_calculation(calc_id)
        logger.info(f"[FILTER THREADED] Previous calculation stopped, creating new thread")
        
        # Create new thread and worker for this specific calculation
        calculation_thread = QThread()
        logger.info(f"[FILTER THREADED] QThread created")

        # Decide whether to use C++ streaming filter (MPAI) or Python worker
        mpai_reader = None
        time_column_name = None
        polars_df = None
        try:
            if self.parent_widget is not None and hasattr(self.parent_widget, "signal_processor"):
                sp = self.parent_widget.signal_processor
                raw = getattr(sp, "raw_dataframe", None)
                logger.info(f"[FILTER THREADED] raw_dataframe type: {type(raw).__name__ if raw else 'None'}")
                if raw is not None and hasattr(raw, "get_header"):
                    # MpaiReader detected
                    mpai_reader = raw
                    time_column_name = getattr(sp, "time_column_name", None)
                    logger.info(f"[FILTER THREADED] MpaiReader detected, time_column: {time_column_name}")
                elif raw is not None and hasattr(raw, "columns"):
                    polars_df = raw
                    time_column_name = getattr(sp, "time_column_name", None)
                    logger.info(f"[FILTER THREADED] Polars DataFrame detected, time_column: {time_column_name}")
                else:
                    logger.warning(f"[FILTER THREADED] Unknown data type: {type(raw)}")
        except Exception as e:
            logger.error(f"[FILTER THREADED] Error detecting data source: {e}")
            mpai_reader = None
            time_column_name = None
            polars_df = None

        logger.info(f"[FILTER THREADED] Data source check complete: mpai={mpai_reader is not None}, polars={polars_df is not None}")
        logger.info(f"[FILTER THREADED] C++ available: {CppFilterCalculationWorker.is_cpp_available()}")

        if mpai_reader is not None:
            logger.info("[FILTER] Using pure Python streaming filter for MPAI data")
            calculation_worker = CppFilterCalculationWorker(
                all_signals,
                conditions,
                mpai_reader=mpai_reader,
                time_column_name=time_column_name,
                use_cpp=False,
            )
            logger.info("[FILTER THREADED] CppFilterCalculationWorker created")
        elif polars_df is not None and time_column_name:
            # Non-MPAI path no longer supported
            if callback:
                callback([])
            logger.error("[FILTER] Non-MPAI data is not supported. Please use MPAI.")
            return
        else:
            # Non-MPAI path no longer supported
            if callback:
                callback([])
            logger.error("[FILTER] Non-MPAI data is not supported. Please use MPAI.")
            return
        
        logger.info(f"[FILTER THREADED] Moving worker to thread")
        calculation_worker.moveToThread(calculation_thread)
        
        # Store thread and worker with unique identifier
        self.calculation_threads[calc_id] = calculation_thread
        self.calculation_workers[calc_id] = calculation_worker
        
        # ✅ FIX: Store current worker reference for callback access to cpp_conditions
        self._current_worker = calculation_worker
        
        # Connect signals with proper order and error handling
        logger.info(f"[FILTER THREADED] Connecting signals")
        calculation_thread.started.connect(calculation_worker.run)
        
        # Use a safe callback wrapper
        if callback:
            calculation_worker.finished.connect(
                lambda segments: self._safe_callback_execution(callback, segments, calc_id)
            )
        
        calculation_worker.error.connect(lambda error: self._on_calculation_error(error, calc_id))
        
        # Cleanup connections - Critical: Must disconnect before deleteLater
        def safe_cleanup():
            try:
                # Disconnect all signals before deletion
                calculation_worker.finished.disconnect()
                calculation_worker.error.disconnect()
                calculation_worker.progress.disconnect()
                calculation_thread.started.disconnect()
            except (RuntimeError, TypeError) as e:
                logger.debug(f"Signal already disconnected for {calc_id}: {e}")
        
        calculation_worker.finished.connect(safe_cleanup)
        calculation_worker.finished.connect(calculation_thread.quit)
        calculation_worker.finished.connect(calculation_worker.deleteLater)
        calculation_thread.finished.connect(calculation_thread.deleteLater)
        
        # Reset references after cleanup
        calculation_thread.finished.connect(lambda: self._reset_thread_references(calc_id))
        
        # Start the thread
        logger.info(f"[FILTER THREADED] Starting thread for {calc_id}")
        calculation_thread.start()
        logger.info(f"[FILTER THREADED] Thread started for {calc_id}, returning to main thread")
    
    def _safe_callback_execution(self, callback, segments, calc_id):
        """Safely execute callback with error handling."""
        try:
            if not self._cleanup_in_progress and callback:
                callback(segments)
        except RuntimeError:
            # Object deleted, silently ignore
            pass
        except Exception as e:
            logger.error(f"Error in filter callback: {e}")
    
    def _reset_thread_references(self, calc_id):
        """Reset thread references after cleanup."""
        try:
            # Only reset if not currently cleaning up
            if not self._cleanup_in_progress:
                if calc_id in self.calculation_threads:
                    del self.calculation_threads[calc_id]
                if calc_id in self.calculation_workers:
                    del self.calculation_workers[calc_id]
        except Exception:
            pass
    
    def _stop_calculation(self, calc_id):
        """Stop a specific running calculation with thread safety."""
        try:
            # First stop the worker
            if calc_id in self.calculation_workers:
                worker = self.calculation_workers[calc_id]
                try:
                    worker.stop()
                    logger.debug(f"Worker stop signal sent for {calc_id}")
                except (RuntimeError, AttributeError) as e:
                    logger.debug(f"Worker already deleted for {calc_id}: {e}")
                
            # Then handle the thread with full safety checks
            if calc_id in self.calculation_threads:
                thread = self.calculation_threads[calc_id]
                
                if thread is None:
                    logger.debug(f"Thread is None for {calc_id}")
                else:
                    try:
                        # Check if thread object is still valid
                        is_running = thread.isRunning()
                        
                        if is_running:
                            logger.info(f"Stopping running filter calculation thread for {calc_id}...")
                            
                            # Disconnect signals first to prevent issues
                            try:
                                if calc_id in self.calculation_workers:
                                    worker = self.calculation_workers[calc_id]
                                    worker.finished.disconnect()
                                    worker.error.disconnect()
                                    worker.progress.disconnect()
                                thread.started.disconnect()
                            except (RuntimeError, TypeError, AttributeError):
                                pass  # Signals may already be disconnected or deleted
                            
                            # Quit and wait for thread
                            try:
                                thread.quit()
                                if not thread.wait(3000):  # 3 second timeout
                                    logger.warning(f"Filter thread {calc_id} did not finish, terminating...")
                                    thread.terminate()
                                    thread.wait(1000)
                                logger.info(f"Filter calculation thread stopped for {calc_id}")
                            except RuntimeError as e:
                                logger.debug(f"Thread quit/wait failed for {calc_id}: {e}")
                        else:
                            logger.debug(f"Filter calculation thread for {calc_id} was not running")
                            
                    except RuntimeError as e:
                        logger.debug(f"Thread already deleted during stop for {calc_id}: {e}")
                    
            # Clean up references
            if calc_id in self.calculation_workers:
                del self.calculation_workers[calc_id]
            if calc_id in self.calculation_threads:
                del self.calculation_threads[calc_id]
            
        except Exception as e:
            logger.warning(f"Error stopping calculation for {calc_id}: {e}")
    
    def _on_calculation_error(self, error_msg: str, calc_id: str):
        """Handle calculation error."""
        logger.error(f"Filter calculation error for {calc_id}: {error_msg}")
    
    def cleanup(self):
        """Cleanup filter manager resources - stop all running threads."""
        logger.info("FilterManager cleanup started")
        self._cleanup_in_progress = True
        
        try:
            # Stop all running calculations with thread safety
            calc_ids = list(self.calculation_threads.keys())
            logger.debug(f"Stopping {len(calc_ids)} calculation threads")
            
            for calc_id in calc_ids:
                try:
                    self._stop_calculation(calc_id)
                except Exception as e:
                    logger.warning(f"Error stopping calculation {calc_id}: {e}")
            
            # Wait a bit for all threads to finish
            import time
            time.sleep(0.15)
            
            # Clear all references
            self.calculation_threads.clear()
            self.calculation_workers.clear()
            self.active_filters.clear()
            
            logger.info("FilterManager cleanup completed")
        except Exception as e:
            logger.error(f"Error during FilterManager cleanup: {e}")
        finally:
            self._cleanup_in_progress = False
    
    def calculate_filter_segments(self, all_signals: dict, conditions: list) -> list:
        
        # Start with all time points
        time_data = None
        for signal_name, signal_data in all_signals.items():
            if 'x_data' in signal_data and len(signal_data['x_data']) > 0:
                time_data = np.array(signal_data['x_data'])
                break
        
        if time_data is None:
            return []
        
        # Create a boolean mask for all time points
        combined_mask = np.ones(len(time_data), dtype=bool)
        
        # Apply each condition
        for condition in conditions:
            param_name = condition['parameter']
            ranges = condition['ranges']
            
            if param_name not in all_signals:
                continue
            
            param_data = np.array(all_signals[param_name]['y_data'])
            condition_mask = np.zeros(len(param_data), dtype=bool)
            
            # Apply all ranges for this parameter (OR logic within parameter)
            for range_filter in ranges:
                range_type = range_filter['type']
                operator = range_filter['operator']
                value = range_filter['value']
                
                if range_type == 'lower':
                    if operator == '>=':
                        range_mask = param_data >= value
                    elif operator == '>':
                        range_mask = param_data > value
                    else:
                        continue
                elif range_type == 'upper':
                    if operator == '<=':
                        range_mask = param_data <= value
                    elif operator == '<':
                        range_mask = param_data < value
                    else:
                        continue
                else:
                    continue
                
                condition_mask |= range_mask
            
            # Combine with overall mask (AND logic between parameters)
            combined_mask &= condition_mask
        
        # Find continuous segments
        segments = self._find_continuous_segments(time_data, combined_mask)
        
        return segments
    
    def _find_continuous_segments(self, time_data: np.ndarray, mask: np.ndarray) -> List[Tuple[float, float]]:
        """Find continuous time segments where mask is True."""
        if not np.any(mask):
            return []
        
        # Find indices where mask is True
        true_indices = np.where(mask)[0]
        
        if len(true_indices) == 0:
            return []
        
        segments = []
        start_idx = true_indices[0]
        
        for i in range(1, len(true_indices)):
            # If there's a gap, end current segment and start new one
            if true_indices[i] - true_indices[i-1] > 1:
                end_idx = true_indices[i-1]
                segments.append((float(time_data[start_idx]), float(time_data[end_idx])))
                start_idx = true_indices[i]
        
        # Add the last segment
        end_idx = true_indices[-1]
        segments.append((float(time_data[start_idx]), float(time_data[end_idx])))
        
        return segments
    
    # ===== Global range filter state =====
    # There is a single range filter for the whole file. It always uses
    # concatenated display, so it changes the data of every graph in every tab.
    # Stored as {GLOBAL_FILTER_KEY: filter_data} in active_filters, which the
    # widget container manager copies/restores as an opaque dict.
    GLOBAL_FILTER_KEY = 'global'

    def clear_filters(self):
        """Clear the range filter."""
        self.active_filters.clear()
        self.filter_applied = False
        logger.info("[FILTER] Range filter cleared")

    def set_global_filter(self, filter_data: dict):
        """Store the range filter that is applied to all graphs."""
        filter_data = dict(filter_data)
        filter_data['mode'] = 'concatenated'
        self.active_filters = {self.GLOBAL_FILTER_KEY: filter_data}
        self.filter_applied = True
        logger.info(f"[FILTER] Range filter set: {len(filter_data.get('conditions', []))} condition(s)")

    def get_global_filter(self) -> Optional[dict]:
        """The applied range filter, or None."""
        return self.active_filters.get(self.GLOBAL_FILTER_KEY)

    def get_active_filters(self) -> dict:
        """Get all active filters."""
        return self.active_filters.copy()

    def has_active_filters(self) -> bool:
        """Check if a range filter is applied."""
        return self.filter_applied and self.get_global_filter() is not None
