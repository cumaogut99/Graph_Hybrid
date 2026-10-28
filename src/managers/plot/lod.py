"""
Level-of-detail data for the visible range: refetching on zoom/pan and downsampling.

Part of PlotManager; moved from src/managers/plot_manager.py without changes.
"""

import logging
from typing import TYPE_CHECKING
import numpy as np
from src.graphics.lod_engine import LODEngine
if TYPE_CHECKING:
    from src.ui.time_graph_widget import TimeGraphWidget

logger = logging.getLogger(__name__)


class PlotLodMixin:
    """Level-of-detail data for the visible range: refetching on zoom/pan and downsampling."""

    def _on_view_changed(self, plot_index: int, ranges):
        """
        Handle view range changes for LOD (Level of Detail) optimization.
        
        Dynamic LOD Switch:
        - RAW mode: sample_count < pixel_width * 2 → fetch exact raw data
        - AGGREGATED mode: sample_count >= pixel_width * 2 → use C++ downsampler
        """
        # Debounce rapid zoom/pan events
        if self._lod_timer is not None:
            self._lod_timer.stop()
        
        self._lod_pending_ranges[plot_index] = ranges
        
        from PyQt5.QtCore import QTimer
        self._lod_timer = QTimer()
        self._lod_timer.setSingleShot(True)
        self._lod_timer.timeout.connect(self._process_pending_lod_updates)
        self._lod_timer.start(50)  # 50ms debounce for responsive zoom

    def _process_pending_lod_updates(self):
        """
        Dynamic LOD Switch - the core of view-dependent visualization.

        For each signal in the visible range:
        - Calculate sample count in visible range
        - IF sample_count < pixel_width * 2: RAW DIRECT mode (exact data from Arrow)
        - IF sample_count >= pixel_width * 2: AGGREGATED mode (C++ downsampler)
        """
        # ✅ CRITICAL FIX: Skip LOD updates when filter is active
        # Segmented/concatenated filters provide pre-filtered PlotDataItems
        # LOD would reload full dataset from MPAI, overriding the filtered view!
        if getattr(self, '_lod_disabled', False):
            logger.debug("[LOD] Skipping LOD update - filter is active (_lod_disabled=True)")
            self._lod_pending_ranges.clear()
            return

        import time
        start_time = time.time()

        if not self._lod_pending_ranges:
            return
        
        signal_processor = self._get_signal_processor()
        if signal_processor is None:
            self._lod_pending_ranges.clear()
            return
        
        # Get pixel width from first plot widget
        pixel_width = 1920  # Default
        if self.plot_widgets:
            try:
                pixel_width = self.plot_widgets[0].width()
            except:
                pass
        
        # Threshold: RAW if samples < 10x pixel width (aggressive raw switching)
        # 10x ratio allows ~20,000 points for typical 2000px viewport
        raw_threshold = pixel_width * 10
        
        # Process each plot's pending range
        for plot_index, ranges in self._lod_pending_ranges.items():
            if plot_index >= len(self.plot_widgets):
                continue
            
            x_range = ranges[0]  # (x_min, x_max)
            view_x_min, view_x_max = x_range
            
            # Get signals on this plot
            signals_on_plot = [
                (key, item) for key, item in self.current_signals.items()
                if key.endswith(f"_{plot_index}")
            ]
            
            for signal_key, plot_item in signals_on_plot:
                signal_name = signal_key.rsplit('_', 1)[0]
                
                # Get signal info from processor
                signal_info = signal_processor.signal_data.get(signal_name)
                if signal_info is None:
                    continue
                
                metadata = signal_info.get('metadata', {})
                is_memory_mapped = metadata.get('memory_mapped', False)
                
                if not is_memory_mapped:
                    # CSV data: already in memory, PyQtGraph handles it
                    continue
                
                # ========== MEMORY-MAPPED MPAI: Dynamic LOD Switch ==========
                reader = signal_info.get('mpai_reader')
                if reader is None:
                    continue
                
                col_name = signal_info.get('column_name', signal_name)
                time_col = signal_info.get('time_column', 'time')
                row_count = signal_info.get('row_count', 0)
                time_range = signal_info.get('time_range', (0.0, 0.0))
                
                if row_count == 0 or time_range[1] <= time_range[0]:
                    continue
                
                # Calculate sample count in visible range
                total_time_span = time_range[1] - time_range[0]
                if total_time_span <= 0:
                    continue
                
                visible_time_span = max(0, min(view_x_max, time_range[1]) - max(view_x_min, time_range[0]))
                if visible_time_span <= 0:
                    continue
                
                visible_fraction = visible_time_span / total_time_span
                estimated_sample_count = int(row_count * visible_fraction)
                
                # ========== LOD MODE DECISION WITH HYSTERESIS ==========
                # Get current mode for this signal (default to AGGREGATED for safety)
                current_mode = self._current_lod_modes.get(signal_key, "AGGREGATED")
                
                # Apply hysteresis to prevent flickering at threshold boundary
                if current_mode == "AGGREGATED":
                    # Currently in AGGREGATED mode: need to go well BELOW threshold to switch to RAW
                    if estimated_sample_count < raw_threshold * self._lod_hysteresis_down:
                        lod_mode = "RAW"
                    else:
                        lod_mode = "AGGREGATED"
                else:  # current_mode == "RAW"
                    # Currently in RAW mode: need to go well ABOVE threshold to switch to AGGREGATED
                    if estimated_sample_count > raw_threshold * self._lod_hysteresis_up:
                        lod_mode = "AGGREGATED"
                    else:
                        lod_mode = "RAW"
                
                # Update mode cache
                self._current_lod_modes[signal_key] = lod_mode
                
                # Set target points based on mode
                if lod_mode == "RAW":
                    target_points = estimated_sample_count  # All samples
                else:
                    target_points = pixel_width * 2  # 2 points per pixel (min/max)
                
                logger.info(f"[LOD] {signal_name}: mode={lod_mode}, samples={estimated_sample_count:,}, target={target_points:,}, range=[{view_x_min:.2f}, {view_x_max:.2f}]")
                
                # Fetch data with appropriate resolution
                new_x, new_y = self._fetch_lod_data(
                    reader, time_col, col_name,
                    view_x_min, view_x_max,
                    row_count, time_range,
                    target_points, lod_mode
                )
                
                if new_x is not None and len(new_x) > 0:
                    # Update plot item with new data
                    try:
                        new_y = signal_processor.normalize_for_display(signal_name, new_y)
                        plot_item.setData(new_x, new_y)
                        logger.debug(f"[LOD] Updated {signal_name} with {len(new_x)} points")
                    except Exception as e:
                        logger.error(f"[LOD] Failed to update {signal_name}: {e}")
        
        elapsed_ms = (time.time() - start_time) * 1000
        logger.info(f"[LOD] Update complete in {elapsed_ms:.1f}ms")
        
        self._lod_pending_ranges.clear()

    def _fetch_lod_data(self, reader, time_col: str, signal_col: str,
                        view_x_min: float, view_x_max: float,
                        row_count: int, time_range: tuple,
                        target_points: int, lod_mode: str):
        """
        Fetch data at appropriate LOD resolution.
        
        Args:
            reader: MPAI reader instance
            time_col: Time column name
            signal_col: Signal column name
            view_x_min, view_x_max: Visible time range
            row_count: Total row count
            time_range: Full data time range (min, max)
            target_points: Target output points
            lod_mode: "RAW" or "AGGREGATED"
        
        Returns:
            (x_data, y_data) numpy arrays
        """
        try:
            # Calculate row range from time range
            total_time = time_range[1] - time_range[0]
            if total_time <= 0:
                return None, None
            
            # Clamp to data range
            clamp_x_min = max(view_x_min, time_range[0])
            clamp_x_max = min(view_x_max, time_range[1])
            
            # Estimate row indices
            start_fraction = (clamp_x_min - time_range[0]) / total_time
            end_fraction = (clamp_x_max - time_range[0]) / total_time
            
            start_row = max(0, int(start_fraction * row_count))
            end_row = min(row_count, int(end_fraction * row_count) + 1)
            slice_count = end_row - start_row
            
            if slice_count <= 0:
                return None, None
            
            if lod_mode == "RAW":
                # ✅ RAW MODE: Load exact samples from Arrow slice
                x_data = np.array(reader.load_column_slice(time_col, start_row, slice_count), dtype=np.float64)
                y_data = np.array(reader.load_column_slice(signal_col, start_row, slice_count), dtype=np.float64)
                
                # Filter to exact view range (row estimate may be imprecise)
                mask = (x_data >= view_x_min) & (x_data <= view_x_max)
                return x_data[mask], y_data[mask]
            
            else:
                # ✅ AGGREGATED MODE: Use pre-computed LOD files if available
                try:
                    # Try LOD parquet files first (FAST PATH: ~10ms vs 6000ms)
                    from src.data.lod_reader import (
                        get_lod_container_path, 
                        get_available_lod_levels,
                        select_lod_level,
                        load_lod_data
                    )
                    
                    # Get MPAI path from signal_processor
                    signal_processor = self._get_signal_processor()
                    mpai_path = getattr(signal_processor, 'current_mpai_path', None)
                    
                    if mpai_path:
                        available = get_available_lod_levels(mpai_path)
                        
                        if available:
                            # Calculate visible samples to select LOD level
                            visible_samples = int(row_count * (view_x_max - view_x_min) / (time_range[1] - time_range[0]))
                            lod_level = select_lod_level(visible_samples, available)
                            
                            if lod_level:
                                import time as perf_time
                                t_start = perf_time.perf_counter()
                                
                                x_data, y_data = load_lod_data(
                                    mpai_path, lod_level, signal_col,
                                    view_x_min, view_x_max
                                )
                                
                                if x_data is not None and len(x_data) > 0:
                                    elapsed = (perf_time.perf_counter() - t_start) * 1000
                                    logger.info(f"[LOD-FAST] {signal_col}: {lod_level} → {len(x_data)} points in {elapsed:.1f}ms")
                                    return x_data, y_data
                                else:
                                    logger.warning(f"[LOD] LOD data empty for {signal_col}, falling back to streaming")
                
                except ImportError as e:
                    logger.warning(f"[LOD] LOD reader not available: {e}")
                except Exception as e:
                    logger.warning(f"[LOD] LOD parquet failed: {e}, falling back to streaming")
                
                # FALLBACK: Python streaming downsampler
                try:
                    # Check for MpaiDirectoryReader with built-in optimized get_render_data
                    if hasattr(reader, 'get_render_data') and hasattr(reader, 'name_to_id'):
                        ch_id = reader.name_to_id.get(signal_col)
                        if ch_id is not None:
                            pixel_width = max(1, target_points // 2)
                            x_data, y_data, src_type = reader.get_render_data(
                                ch_id, view_x_min, view_x_max, pixel_width
                            )
                            if len(x_data) > 0:
                                return x_data, y_data

                    # Python min-max downsample
                    return self._python_minmax_downsample(
                        reader, time_col, signal_col,
                        start_row, slice_count, target_points
                    )

                except Exception as e:
                    logger.error(f"[LOD] Python downsample failed: {e}, falling back to Python min-max")
                    return self._python_minmax_downsample(
                        reader, time_col, signal_col,
                        start_row, slice_count, target_points
                    )
                    
        except Exception as e:
            logger.error(f"[LOD] _fetch_lod_data failed: {e}")
            import traceback
            traceback.print_exc()
            return None, None

    def _python_minmax_downsample(self, reader, time_col: str, signal_col: str,
                                   start_row: int, slice_count: int, target_points: int):
        """Python fallback for min-max downsampling."""
        try:
            # Load slice
            x_data = np.array(reader.load_column_slice(time_col, start_row, slice_count), dtype=np.float64)
            y_data = np.array(reader.load_column_slice(signal_col, start_row, slice_count), dtype=np.float64)
            
            if len(x_data) <= target_points:
                return x_data, y_data
            
            # Min-max bucketing
            num_buckets = target_points // 2
            bucket_size = len(x_data) // num_buckets
            
            x_out = []
            y_out = []
            
            for i in range(num_buckets):
                start = i * bucket_size
                end = min((i + 1) * bucket_size, len(x_data))
                if start >= end:
                    continue
                
                x_bucket = x_data[start:end]
                y_bucket = y_data[start:end]
                
                min_idx = np.argmin(y_bucket)
                max_idx = np.argmax(y_bucket)
                
                if min_idx < max_idx:
                    x_out.extend([x_bucket[min_idx], x_bucket[max_idx]])
                    y_out.extend([y_bucket[min_idx], y_bucket[max_idx]])
                else:
                    x_out.extend([x_bucket[max_idx], x_bucket[min_idx]])
                    y_out.extend([y_bucket[max_idx], y_bucket[min_idx]])
            
            return np.array(x_out), np.array(y_out)
            
        except Exception as e:
            logger.error(f"[LOD] Python fallback failed: {e}")
            return None, None

    def _get_lod_engine(self):
        """Get or create LOD Engine instance."""
        if self._lod_engine is None:
            signal_processor = self._get_signal_processor()
            if signal_processor is not None:
                self._lod_engine = LODEngine(signal_processor)
        return self._lod_engine

    def _downsample_data(self, x_data: np.ndarray, y_data: np.ndarray, max_points: int = 5000):
        """
        Downsample data for faster rendering.
        Uses intelligent downsampling to preserve visual appearance and data range.
        
        Args:
            x_data: X-axis data
            y_data: Y-axis data
            max_points: Maximum number of points to display
            
        Returns:
            Tuple of (downsampled_x, downsampled_y)
        """
        data_len = len(x_data)
        
        # No downsampling needed
        if data_len <= max_points:
            return x_data, y_data
        
        # Calculate step size
        step = max(1, data_len // max_points)
        
        # Simple downsampling - take every Nth point
        # This preserves general shape and is fast
        downsampled_x = x_data[::step]
        downsampled_y = y_data[::step]
        
        # CRITICAL: Ensure first and last points are ALWAYS included
        # This guarantees the full data range is represented in the downsampled data
        result_x = [x_data[0]]
        result_y = [y_data[0]]
        
        # Add the downsampled middle points (skip if they're duplicates of first/last)
        for i in range(len(downsampled_x)):
            if i > 0 and i < len(downsampled_x) - 1:
                result_x.append(downsampled_x[i])
                result_y.append(downsampled_y[i])
        
        # Always add the last point if it's not already there
        if len(x_data) > 1 and (len(result_x) == 0 or x_data[-1] != result_x[-1]):
            result_x.append(x_data[-1])
            result_y.append(y_data[-1])
        
        final_x = np.array(result_x)
        final_y = np.array(result_y)
        
        logger.debug(f"Downsampled {data_len} points to {len(final_x)} points (step={step})")
        return final_x, final_y
