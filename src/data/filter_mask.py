"""
Filter Mask - Non-Destructive Filtering

Implements Boolean Mask strategy for filtering without deleting data.
The mask is applied during visualization and statistics calculation.

Benefits:
- Original data is never modified
- Instant filter toggle on/off
- Multiple filters can be combined
- Memory efficient (1 bit per row)

Usage:
    mask = FilterMask(total_rows=1_000_000)
    
    # Apply filters
    mask.apply_range_filter(voltage_data, min_val=200, max_val=250)
    mask.apply_limit_violations([(100, 200), (5000, 5100)])
    
    # Use mask
    visible_data = data[mask.get_visible_indices()]
    # or
    visible_count = mask.visible_count()
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union, Dict, Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class FilterCondition:
    """Represents a single filter condition"""
    name: str
    column: str
    filter_type: str  # 'range', 'violation', 'expression'
    enabled: bool = True
    
    # For range filter
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    
    # For violation regions
    violation_regions: List[Tuple[int, int]] = field(default_factory=list)
    
    # For expression filter
    expression: Optional[str] = None


class FilterMask:
    """
    Non-destructive filtering using boolean masks.
    
    The mask array is True for visible rows, False for hidden rows.
    Multiple filters can be combined (AND operation by default).
    
    Performance:
    - Memory: 1 bit per row (125 KB for 1M rows)
    - Apply filter: O(affected_rows)
    - Get visible indices: O(total_rows) but vectorized
    """
    
    def __init__(self, total_rows: int):
        """
        Initialize filter mask with all rows visible.
        
        Args:
            total_rows: Total number of rows in the dataset
        """
        self.total_rows = total_rows
        self._base_mask = np.ones(total_rows, dtype=bool)  # All visible initially
        self._combined_mask = self._base_mask.copy()
        
        # Track individual filters for toggle support
        self._filters: Dict[str, FilterCondition] = {}
        self._filter_masks: Dict[str, np.ndarray] = {}
        
        logger.debug(f"FilterMask created for {total_rows:,} rows")
    
    # =========================================================================
    # Filter Application
    # =========================================================================
    
    def add_range_filter(
        self, 
        name: str,
        column: str,
        column_data: np.ndarray,
        min_value: float = -np.inf,
        max_value: float = np.inf,
        inverse: bool = False
    ) -> int:
        """
        Add a range filter - hide rows outside [min_value, max_value].
        
        Args:
            name: Unique name for this filter
            column: Column name being filtered
            column_data: Numpy array of column values
            min_value: Minimum allowed value (inclusive)
            max_value: Maximum allowed value (inclusive)
            inverse: If True, hide rows INSIDE the range instead
            
        Returns:
            Number of rows hidden by this filter
        """
        if len(column_data) != self.total_rows:
            raise ValueError(f"Data length {len(column_data)} != mask length {self.total_rows}")
        
        # Create filter mask
        if inverse:
            # Hide values INSIDE range
            filter_mask = ~((column_data >= min_value) & (column_data <= max_value))
        else:
            # Hide values OUTSIDE range
            filter_mask = (column_data >= min_value) & (column_data <= max_value)
        
        # Store filter
        self._filters[name] = FilterCondition(
            name=name,
            column=column,
            filter_type='range',
            min_value=min_value,
            max_value=max_value,
            enabled=True
        )
        self._filter_masks[name] = filter_mask
        
        # Recalculate combined mask
        self._update_combined_mask()
        
        hidden_count = self.total_rows - np.sum(filter_mask)
        logger.info(f"Range filter '{name}': {hidden_count:,} rows hidden")
        
        return hidden_count
    
    def add_violation_filter(
        self,
        name: str,
        column: str,
        violations: List[Tuple[int, int]],
        hide_violations: bool = True
    ) -> int:
        """
        Add filter based on violation regions from Skip Logic.
        
        Args:
            name: Unique name for this filter
            column: Column name
            violations: List of (start_row, end_row) tuples from find_limit_violations
            hide_violations: If True, hide violation rows. If False, hide non-violation rows.
            
        Returns:
            Number of rows hidden
        """
        # Create mask
        filter_mask = np.ones(self.total_rows, dtype=bool)
        
        for start_row, end_row in violations:
            start = max(0, start_row)
            end = min(self.total_rows, end_row)
            if hide_violations:
                filter_mask[start:end] = False
            else:
                filter_mask[start:end] = True
        
        if not hide_violations:
            # Invert: show only violation regions
            filter_mask = ~filter_mask
            # Then set violation regions to True
            filter_mask[:] = False
            for start_row, end_row in violations:
                filter_mask[start_row:end_row] = True
        
        # Store filter
        self._filters[name] = FilterCondition(
            name=name,
            column=column,
            filter_type='violation',
            violation_regions=violations,
            enabled=True
        )
        self._filter_masks[name] = filter_mask
        
        self._update_combined_mask()
        
        hidden_count = self.total_rows - np.sum(filter_mask)
        logger.info(f"Violation filter '{name}': {hidden_count:,} rows hidden ({len(violations)} regions)")
        
        return hidden_count
    
    def add_expression_filter(
        self,
        name: str,
        expression: str,
        data_columns: Dict[str, np.ndarray]
    ) -> int:
        """
        Add filter based on a boolean expression.
        
        Args:
            name: Unique name for this filter
            expression: Boolean expression like "voltage > 230 and current < 20"
            data_columns: Dict of column_name -> numpy array
            
        Returns:
            Number of rows hidden
        """
        try:
            # Safe evaluation with numpy
            local_vars = {col: data for col, data in data_columns.items()}
            local_vars['np'] = np
            
            # Replace common operators
            expr = expression.replace(' and ', ' & ')
            expr = expr.replace(' or ', ' | ')
            expr = expr.replace(' not ', ' ~ ')
            
            filter_mask = eval(expr, {"__builtins__": {}}, local_vars)
            
            if not isinstance(filter_mask, np.ndarray):
                filter_mask = np.array(filter_mask, dtype=bool)
            
            # Store
            self._filters[name] = FilterCondition(
                name=name,
                column='',
                filter_type='expression',
                expression=expression,
                enabled=True
            )
            self._filter_masks[name] = filter_mask
            
            self._update_combined_mask()
            
            hidden_count = self.total_rows - np.sum(filter_mask)
            logger.info(f"Expression filter '{name}': {hidden_count:,} rows hidden")
            
            return hidden_count
            
        except Exception as e:
            logger.error(f"Failed to evaluate expression '{expression}': {e}")
            raise ValueError(f"Invalid filter expression: {e}")
    
    # =========================================================================
    # Filter Management
    # =========================================================================
    
    def enable_filter(self, name: str) -> None:
        """Enable a filter by name."""
        if name in self._filters:
            self._filters[name].enabled = True
            self._update_combined_mask()
            logger.debug(f"Enabled filter: {name}")
    
    def disable_filter(self, name: str) -> None:
        """Disable a filter by name."""
        if name in self._filters:
            self._filters[name].enabled = False
            self._update_combined_mask()
            logger.debug(f"Disabled filter: {name}")
    
    def toggle_filter(self, name: str) -> bool:
        """Toggle a filter and return new state."""
        if name in self._filters:
            self._filters[name].enabled = not self._filters[name].enabled
            self._update_combined_mask()
            return self._filters[name].enabled
        return False
    
    def remove_filter(self, name: str) -> None:
        """Remove a filter completely."""
        if name in self._filters:
            del self._filters[name]
            del self._filter_masks[name]
            self._update_combined_mask()
            logger.debug(f"Removed filter: {name}")
    
    def clear_all_filters(self) -> None:
        """Remove all filters."""
        self._filters.clear()
        self._filter_masks.clear()
        self._combined_mask = self._base_mask.copy()
        logger.debug("All filters cleared")
    
    def get_filter_info(self) -> List[Dict[str, Any]]:
        """Get information about all active filters."""
        return [
            {
                'name': f.name,
                'column': f.column,
                'type': f.filter_type,
                'enabled': f.enabled,
                'hidden_count': self.total_rows - np.sum(self._filter_masks.get(f.name, self._base_mask))
            }
            for f in self._filters.values()
        ]
    
    # =========================================================================
    # Mask Access
    # =========================================================================
    
    def get_visible_indices(self) -> np.ndarray:
        """
        Get indices of all visible (non-masked) rows.
        
        Returns:
            Numpy array of row indices
        """
        return np.where(self._combined_mask)[0]
    
    def get_hidden_indices(self) -> np.ndarray:
        """Get indices of all hidden (masked) rows."""
        return np.where(~self._combined_mask)[0]
    
    def get_mask(self) -> np.ndarray:
        """Get the combined boolean mask array."""
        return self._combined_mask
    
    def visible_count(self) -> int:
        """Get count of visible rows."""
        return int(np.sum(self._combined_mask))
    
    def hidden_count(self) -> int:
        """Get count of hidden rows."""
        return self.total_rows - self.visible_count()
    
    def is_visible(self, row_index: int) -> bool:
        """Check if a specific row is visible."""
        if 0 <= row_index < self.total_rows:
            return bool(self._combined_mask[row_index])
        return False
    
    # =========================================================================
    # Data Filtering
    # =========================================================================
    
    def filter_data(self, data: np.ndarray) -> np.ndarray:
        """
        Apply mask to filter data array.
        
        Args:
            data: Input array (must match total_rows length)
            
        Returns:
            Filtered array containing only visible rows
        """
        if len(data) != self.total_rows:
            raise ValueError(f"Data length {len(data)} != mask length {self.total_rows}")
        return data[self._combined_mask]
    
    def filter_range(self, start_row: int, end_row: int) -> Tuple[np.ndarray, int]:
        """
        Get visible indices within a row range.
        
        Args:
            start_row: Start row (inclusive)
            end_row: End row (exclusive)
            
        Returns:
            (visible_indices, visible_count)
        """
        range_mask = self._combined_mask[start_row:end_row]
        visible_indices = np.where(range_mask)[0] + start_row
        return visible_indices, len(visible_indices)
    
    # =========================================================================
    # Internal Methods
    # =========================================================================
    
    def _update_combined_mask(self) -> None:
        """Recalculate combined mask from all enabled filters."""
        self._combined_mask = self._base_mask.copy()
        
        for name, condition in self._filters.items():
            if condition.enabled and name in self._filter_masks:
                self._combined_mask &= self._filter_masks[name]
    
    def __repr__(self) -> str:
        return (f"FilterMask(total={self.total_rows:,}, "
                f"visible={self.visible_count():,}, "
                f"filters={len(self._filters)})")


# =============================================================================
# Convenience Functions
# =============================================================================

def create_range_mask(
    column_data: np.ndarray,
    min_value: float = -np.inf,
    max_value: float = np.inf
) -> np.ndarray:
    """
    Create a simple range mask without FilterMask object.
    
    Args:
        column_data: Data array
        min_value: Minimum allowed value
        max_value: Maximum allowed value
        
    Returns:
        Boolean mask array (True = visible)
    """
    return (column_data >= min_value) & (column_data <= max_value)


def merge_violation_regions(
    violations: List[Tuple[int, int]]
) -> List[Tuple[int, int]]:
    """
    Merge overlapping violation regions.
    
    Args:
        violations: List of (start, end) tuples
        
    Returns:
        Merged list with no overlaps
    """
    if not violations:
        return []
    
    # Sort by start
    sorted_violations = sorted(violations, key=lambda x: x[0])
    
    merged = [sorted_violations[0]]
    for start, end in sorted_violations[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            # Overlapping - extend
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    
    return merged
