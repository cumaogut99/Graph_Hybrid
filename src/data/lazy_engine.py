"""
Lazy Evaluation Engine - Expression DAG

Implements lazy/deferred calculations using an expression DAG (Directed Acyclic Graph).
Formulas are stored as expressions and evaluated only when data is needed.

Benefits:
- Memory efficient: No duplicate data storage
- Compute on demand: Only calculate what's visible
- Streaming compatible: Process chunk-by-chunk
- Composable: Complex formulas from simple operations

Example:
    engine = LazyEngine(reader)
    
    # Define calculated signals (not executed yet)
    engine.define("power", "voltage * current")
    engine.define("power_kw", "power / 1000")
    engine.define("rms_power", "sqrt(mean(power^2))")
    
    # Evaluate only when needed (for visible range)
    data = engine.evaluate("power_kw", start_row=0, end_row=1000)
"""

import logging
import operator
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# Expression Node Types
# =============================================================================

class NodeType(Enum):
    """Types of expression nodes"""
    COLUMN = auto()      # Reference to a data column
    CONSTANT = auto()    # Numeric constant
    BINARY_OP = auto()   # Binary operation (a + b, a * b, etc.)
    UNARY_OP = auto()    # Unary operation (-a, abs(a), etc.)
    FUNCTION = auto()    # Function call (sqrt, sin, etc.)
    AGGREGATE = auto()   # Aggregation (mean, sum, min, max)


class ExpressionNode(ABC):
    """Base class for expression tree nodes"""
    
    @property
    @abstractmethod
    def node_type(self) -> NodeType:
        """Return the type of this node"""
        pass
    
    @abstractmethod
    def evaluate(self, context: 'EvaluationContext') -> np.ndarray:
        """Evaluate this node and return result array"""
        pass
    
    @abstractmethod
    def get_dependencies(self) -> Set[str]:
        """Get set of column names this node depends on"""
        pass
    
    @abstractmethod
    def __repr__(self) -> str:
        pass


@dataclass
class ColumnNode(ExpressionNode):
    """Reference to a data column"""
    column_name: str
    
    @property
    def node_type(self) -> NodeType:
        return NodeType.COLUMN
    
    def evaluate(self, context: 'EvaluationContext') -> np.ndarray:
        return context.get_column(self.column_name)
    
    def get_dependencies(self) -> Set[str]:
        return {self.column_name}
    
    def __repr__(self) -> str:
        return f"Column({self.column_name})"


@dataclass
class ConstantNode(ExpressionNode):
    """Numeric constant"""
    value: float
    
    @property
    def node_type(self) -> NodeType:
        return NodeType.CONSTANT
    
    def evaluate(self, context: 'EvaluationContext') -> np.ndarray:
        # Return scalar, numpy will broadcast
        return np.float64(self.value)
    
    def get_dependencies(self) -> Set[str]:
        return set()
    
    def __repr__(self) -> str:
        return f"Const({self.value})"


@dataclass
class BinaryOpNode(ExpressionNode):
    """Binary operation between two expressions"""
    left: ExpressionNode
    right: ExpressionNode
    op: str  # '+', '-', '*', '/', '^', '%'
    
    @property
    def node_type(self) -> NodeType:
        return NodeType.BINARY_OP
    
    # Operator mapping (class variable, not instance)
    _OPERATORS = {
        '+': operator.add,
        '-': operator.sub,
        '*': operator.mul,
        '/': operator.truediv,
        '^': np.power,
        '**': np.power,
        '%': operator.mod,
    }
    
    def evaluate(self, context: 'EvaluationContext') -> np.ndarray:
        left_val = self.left.evaluate(context)
        right_val = self.right.evaluate(context)
        
        if self.op not in self._OPERATORS:
            raise ValueError(f"Unknown operator: {self.op}")
        
        return self._OPERATORS[self.op](left_val, right_val)
    
    def get_dependencies(self) -> Set[str]:
        return self.left.get_dependencies() | self.right.get_dependencies()
    
    def __repr__(self) -> str:
        return f"({self.left} {self.op} {self.right})"


@dataclass
class UnaryOpNode(ExpressionNode):
    """Unary operation on an expression"""
    operand: ExpressionNode
    op: str  # '-', 'abs', 'sqrt', etc.
    
    @property
    def node_type(self) -> NodeType:
        return NodeType.UNARY_OP
    
    _OPERATORS = {
        '-': operator.neg,
        'abs': np.abs,
        'sqrt': np.sqrt,
        'sin': np.sin,
        'cos': np.cos,
        'tan': np.tan,
        'log': np.log,
        'log10': np.log10,
        'exp': np.exp,
        'floor': np.floor,
        'ceil': np.ceil,
        'round': np.round,
    }
    
    def evaluate(self, context: 'EvaluationContext') -> np.ndarray:
        val = self.operand.evaluate(context)
        
        if self.op not in self._OPERATORS:
            raise ValueError(f"Unknown unary operator: {self.op}")
        
        return self._OPERATORS[self.op](val)
    
    def get_dependencies(self) -> Set[str]:
        return self.operand.get_dependencies()
    
    def __repr__(self) -> str:
        return f"{self.op}({self.operand})"


@dataclass
class FunctionNode(ExpressionNode):
    """Multi-argument function call"""
    func_name: str
    args: List[ExpressionNode]
    
    @property
    def node_type(self) -> NodeType:
        return NodeType.FUNCTION
    
    _FUNCTIONS = {
        'min': np.minimum,
        'max': np.maximum,
        'clip': np.clip,
        'where': np.where,
        'power': np.power,
    }
    
    def evaluate(self, context: 'EvaluationContext') -> np.ndarray:
        arg_vals = [arg.evaluate(context) for arg in self.args]
        
        if self.func_name not in self._FUNCTIONS:
            raise ValueError(f"Unknown function: {self.func_name}")
        
        return self._FUNCTIONS[self.func_name](*arg_vals)
    
    def get_dependencies(self) -> Set[str]:
        deps = set()
        for arg in self.args:
            deps |= arg.get_dependencies()
        return deps
    
    def __repr__(self) -> str:
        args_str = ", ".join(str(arg) for arg in self.args)
        return f"{self.func_name}({args_str})"


@dataclass
class AggregateNode(ExpressionNode):
    """Aggregation operation (returns scalar or reduced array)"""
    operand: ExpressionNode
    agg_func: str  # 'mean', 'sum', 'min', 'max', 'std', 'var'
    
    @property
    def node_type(self) -> NodeType:
        return NodeType.AGGREGATE
    
    _AGGREGATORS = {
        'mean': np.mean,
        'sum': np.sum,
        'min': np.min,
        'max': np.max,
        'std': np.std,
        'var': np.var,
        'median': np.median,
        'count': lambda x: np.float64(len(x)),
    }
    
    def evaluate(self, context: 'EvaluationContext') -> np.ndarray:
        val = self.operand.evaluate(context)
        
        if self.agg_func not in self._AGGREGATORS:
            raise ValueError(f"Unknown aggregation: {self.agg_func}")
        
        return self._AGGREGATORS[self.agg_func](val)
    
    def get_dependencies(self) -> Set[str]:
        return self.operand.get_dependencies()
    
    def __repr__(self) -> str:
        return f"{self.agg_func}({self.operand})"


# =============================================================================
# Evaluation Context
# =============================================================================

class EvaluationContext:
    """
    Context for evaluating expressions.
    
    Provides access to column data and manages the row range being evaluated.
    Supports both eager (all data) and streaming (chunk-at-a-time) modes.
    Can resolve calculated signals through a lazy_engine reference.
    """
    
    def __init__(
        self,
        data_source: Any,  # MpaiReader, DataFrame, or dict
        start_row: int = 0,
        end_row: Optional[int] = None,
        chunk_size: Optional[int] = None,
        lazy_engine: Optional['LazyEngine'] = None
    ):
        self.data_source = data_source
        self.start_row = start_row
        self.end_row = end_row
        self.chunk_size = chunk_size
        self.lazy_engine = lazy_engine
        
        # Cache for loaded columns
        self._column_cache: Dict[str, np.ndarray] = {}
        
        # Determine source type
        self._source_type = self._detect_source_type()
    
    def _detect_source_type(self) -> str:
        if hasattr(self.data_source, 'load_column_slice'):
            return 'mpai_reader'
        elif hasattr(self.data_source, 'to_numpy'):
            return 'polars'
        elif isinstance(self.data_source, dict):
            return 'dict'
        else:
            return 'unknown'
    
    def get_column(self, column_name: str) -> np.ndarray:
        """
        Get column data for the current row range.
        Uses caching to avoid repeated loads.
        Can resolve calculated signals through lazy_engine.
        """
        cache_key = f"{column_name}:{self.start_row}:{self.end_row}"
        
        if cache_key in self._column_cache:
            return self._column_cache[cache_key]
        
        # First, check if this is a calculated signal
        if self.lazy_engine and column_name in self.lazy_engine._expressions:
            # Evaluate the calculated signal
            expr = self.lazy_engine._expressions[column_name]
            data = expr.evaluate(self)
            self._column_cache[cache_key] = data
            return data
        
        # Load from source
        if self._source_type == 'mpai_reader':
            end = self.end_row if self.end_row else self.data_source.get_row_count()
            count = end - self.start_row
            data = np.array(self.data_source.load_column_slice(column_name, self.start_row, count))
            
        elif self._source_type == 'polars':
            df = self.data_source
            if self.end_row:
                df = df.slice(self.start_row, self.end_row - self.start_row)
            data = df[column_name].to_numpy()
            
        elif self._source_type == 'dict':
            data = np.array(self.data_source[column_name])
            if self.end_row:
                data = data[self.start_row:self.end_row]
            
        else:
            raise ValueError(f"Cannot get column from source type: {self._source_type}")
        
        self._column_cache[cache_key] = data
        return data
    
    def clear_cache(self):
        """Clear the column cache."""
        self._column_cache.clear()


# =============================================================================
# Expression Parser
# =============================================================================

class ExpressionParser:
    """
    Parses expression strings into expression trees.
    
    Supported syntax:
    - Columns: voltage, current, time
    - Operators: +, -, *, /, ^, %
    - Functions: sqrt(x), abs(x), sin(x), cos(x), log(x)
    - Aggregates: mean(x), sum(x), min(x), max(x)
    - Parentheses: (a + b) * c
    - Constants: 3.14, -5, 1e6
    """
    
    # Token patterns
    TOKEN_PATTERNS = [
        (r'\d+\.?\d*(?:[eE][+-]?\d+)?', 'NUMBER'),  # Numbers
        (r'[a-zA-Z_][a-zA-Z0-9_]*', 'IDENTIFIER'),  # Identifiers and functions
        (r'\+', 'PLUS'),
        (r'-', 'MINUS'),
        (r'\*\*', 'POWER'),  # Must come before TIMES
        (r'\*', 'TIMES'),
        (r'/', 'DIVIDE'),
        (r'\^', 'POWER'),
        (r'%', 'MOD'),
        (r'\(', 'LPAREN'),
        (r'\)', 'RPAREN'),
        (r',', 'COMMA'),
        (r'\s+', None),  # Skip whitespace
    ]
    
    FUNCTIONS = {'sqrt', 'abs', 'sin', 'cos', 'tan', 'log', 'log10', 'exp', 'floor', 'ceil', 'round'}
    AGGREGATES = {'mean', 'sum', 'min', 'max', 'std', 'var', 'median', 'count'}
    MULTI_ARG_FUNCTIONS = {'clip', 'where', 'power'}
    
    def __init__(self, known_columns: Set[str]):
        self.known_columns = known_columns
        self._tokens: List[Tuple[str, str]] = []
        self._pos: int = 0
    
    def parse(self, expression: str) -> ExpressionNode:
        """Parse expression string into expression tree."""
        self._tokens = self._tokenize(expression)
        self._pos = 0
        
        result = self._parse_expression()
        
        if self._pos < len(self._tokens):
            raise ValueError(f"Unexpected token: {self._tokens[self._pos]}")
        
        return result
    
    def _tokenize(self, expression: str) -> List[Tuple[str, str]]:
        """Tokenize the expression string."""
        tokens = []
        pos = 0
        
        while pos < len(expression):
            match = None
            for pattern, token_type in self.TOKEN_PATTERNS:
                regex = re.compile(pattern)
                match = regex.match(expression, pos)
                if match:
                    if token_type:  # Skip None (whitespace)
                        tokens.append((match.group(), token_type))
                    pos = match.end()
                    break
            
            if not match:
                raise ValueError(f"Invalid character at position {pos}: '{expression[pos]}'")
        
        return tokens
    
    def _current_token(self) -> Optional[Tuple[str, str]]:
        if self._pos < len(self._tokens):
            return self._tokens[self._pos]
        return None
    
    def _consume(self, expected_type: Optional[str] = None) -> Tuple[str, str]:
        token = self._current_token()
        if token is None:
            raise ValueError("Unexpected end of expression")
        if expected_type and token[1] != expected_type:
            raise ValueError(f"Expected {expected_type}, got {token[1]}")
        self._pos += 1
        return token
    
    def _parse_expression(self) -> ExpressionNode:
        """Parse additive expression (lowest precedence)."""
        return self._parse_additive()
    
    def _parse_additive(self) -> ExpressionNode:
        """Parse + and - operations."""
        left = self._parse_multiplicative()
        
        while self._current_token() and self._current_token()[1] in ('PLUS', 'MINUS'):
            op_token = self._consume()
            op = '+' if op_token[1] == 'PLUS' else '-'
            right = self._parse_multiplicative()
            left = BinaryOpNode(left=left, right=right, op=op)
        
        return left
    
    def _parse_multiplicative(self) -> ExpressionNode:
        """Parse *, /, % operations."""
        left = self._parse_power()
        
        while self._current_token() and self._current_token()[1] in ('TIMES', 'DIVIDE', 'MOD'):
            op_token = self._consume()
            op = {'TIMES': '*', 'DIVIDE': '/', 'MOD': '%'}[op_token[1]]
            right = self._parse_power()
            left = BinaryOpNode(left=left, right=right, op=op)
        
        return left
    
    def _parse_power(self) -> ExpressionNode:
        """Parse ^ and ** operations (right associative)."""
        left = self._parse_unary()
        
        if self._current_token() and self._current_token()[1] == 'POWER':
            self._consume()
            right = self._parse_power()  # Right associative
            return BinaryOpNode(left=left, right=right, op='^')
        
        return left
    
    def _parse_unary(self) -> ExpressionNode:
        """Parse unary minus."""
        if self._current_token() and self._current_token()[1] == 'MINUS':
            self._consume()
            operand = self._parse_unary()
            return UnaryOpNode(operand=operand, op='-')
        
        return self._parse_primary()
    
    def _parse_primary(self) -> ExpressionNode:
        """Parse primary expressions (atoms, function calls, parentheses)."""
        token = self._current_token()
        
        if token is None:
            raise ValueError("Unexpected end of expression")
        
        # Number
        if token[1] == 'NUMBER':
            self._consume()
            return ConstantNode(value=float(token[0]))
        
        # Identifier (column, function, or aggregate)
        if token[1] == 'IDENTIFIER':
            name = self._consume()[0]
            
            # Check for function call
            if self._current_token() and self._current_token()[1] == 'LPAREN':
                self._consume('LPAREN')
                
                # Parse arguments
                args = []
                if self._current_token() and self._current_token()[1] != 'RPAREN':
                    args.append(self._parse_expression())
                    while self._current_token() and self._current_token()[1] == 'COMMA':
                        self._consume('COMMA')
                        args.append(self._parse_expression())
                
                self._consume('RPAREN')
                
                # Determine function type
                if name.lower() in self.FUNCTIONS:
                    if len(args) != 1:
                        raise ValueError(f"Function {name} expects 1 argument")
                    return UnaryOpNode(operand=args[0], op=name.lower())
                
                elif name.lower() in self.AGGREGATES:
                    if len(args) != 1:
                        raise ValueError(f"Aggregate {name} expects 1 argument")
                    return AggregateNode(operand=args[0], agg_func=name.lower())
                
                elif name.lower() in self.MULTI_ARG_FUNCTIONS:
                    return FunctionNode(func_name=name.lower(), args=args)
                
                else:
                    raise ValueError(f"Unknown function: {name}")
            
            # Column reference
            if name in self.known_columns:
                return ColumnNode(column_name=name)
            else:
                # Might be a calculated column - let engine handle
                return ColumnNode(column_name=name)
        
        # Parenthesized expression
        if token[1] == 'LPAREN':
            self._consume('LPAREN')
            expr = self._parse_expression()
            self._consume('RPAREN')
            return expr
        
        raise ValueError(f"Unexpected token: {token}")


# =============================================================================
# Lazy Engine
# =============================================================================

class LazyEngine:
    """
    Main lazy evaluation engine.
    
    Manages calculated signals as expression trees.
    Evaluates only when data is requested, and only for the requested range.
    
    Usage:
        engine = LazyEngine(data_source)
        engine.define("power", "voltage * current")
        engine.define("power_kw", "power / 1000")
        
        # Lazy - not calculated yet
        data = engine.evaluate("power_kw", start_row=0, end_row=1000)
    """
    
    def __init__(self, data_source: Any):
        """
        Initialize lazy engine.
        
        Args:
            data_source: MpaiReader, Polars DataFrame, or dict of arrays
        """
        self.data_source = data_source
        self._expressions: Dict[str, ExpressionNode] = {}
        self._raw_expressions: Dict[str, str] = {}  # Original strings for debugging
        self._known_columns: Set[str] = self._detect_columns()
        
        logger.debug(f"LazyEngine initialized with {len(self._known_columns)} columns")
    
    def _detect_columns(self) -> Set[str]:
        """Detect available columns from data source."""
        if hasattr(self.data_source, 'get_column_names'):
            return set(self.data_source.get_column_names())
        elif hasattr(self.data_source, 'columns'):
            return set(self.data_source.columns)
        elif isinstance(self.data_source, dict):
            return set(self.data_source.keys())
        return set()
    
    def define(self, name: str, expression: str) -> None:
        """
        Define a calculated signal (lazy).
        
        Args:
            name: Name for the calculated signal
            expression: Expression string (e.g., "voltage * current")
        """
        # Update known columns to include other defined expressions
        all_known = self._known_columns | set(self._expressions.keys())
        
        parser = ExpressionParser(all_known)
        expr_node = parser.parse(expression)
        
        # Check for circular dependencies
        deps = expr_node.get_dependencies()
        if name in deps:
            raise ValueError(f"Circular dependency: {name} depends on itself")
        
        # Check for transitive circular deps
        self._check_circular_deps(name, deps)
        
        self._expressions[name] = expr_node
        self._raw_expressions[name] = expression
        
        logger.info(f"Defined lazy signal: {name} = {expression}")
    
    def _check_circular_deps(self, name: str, deps: Set[str], visited: Optional[Set[str]] = None):
        """Check for circular dependencies."""
        if visited is None:
            visited = {name}
        
        for dep in deps:
            if dep in self._expressions:
                if dep in visited:
                    raise ValueError(f"Circular dependency detected: {' -> '.join(visited)} -> {dep}")
                visited.add(dep)
                sub_deps = self._expressions[dep].get_dependencies()
                self._check_circular_deps(name, sub_deps, visited)
    
    def evaluate(
        self,
        name: str,
        start_row: int = 0,
        end_row: Optional[int] = None,
        chunk_size: Optional[int] = None
    ) -> np.ndarray:
        """
        Evaluate a signal (raw column or calculated).
        
        Args:
            name: Signal name (column or calculated)
            start_row: Start row (inclusive)
            end_row: End row (exclusive), None = all rows
            chunk_size: If set, process in chunks (for streaming)
            
        Returns:
            Numpy array of evaluated values
        """
        # Check if it's a raw column
        if name in self._known_columns and name not in self._expressions:
            ctx = EvaluationContext(self.data_source, start_row, end_row, chunk_size, lazy_engine=self)
            return ctx.get_column(name)
        
        # Check if it's a defined expression
        if name not in self._expressions:
            raise KeyError(f"Unknown signal: {name}")
        
        expr = self._expressions[name]
        
        if chunk_size and chunk_size > 0:
            # Streaming evaluation
            return self._evaluate_streaming(expr, start_row, end_row, chunk_size)
        else:
            # Eager evaluation
            ctx = EvaluationContext(self.data_source, start_row, end_row, lazy_engine=self)
            return expr.evaluate(ctx)
    
    def _evaluate_streaming(
        self,
        expr: ExpressionNode,
        start_row: int,
        end_row: Optional[int],
        chunk_size: int
    ) -> np.ndarray:
        """Evaluate expression in chunks to limit memory usage."""
        if end_row is None:
            if hasattr(self.data_source, 'get_row_count'):
                end_row = self.data_source.get_row_count()
            else:
                raise ValueError("end_row required for streaming evaluation")
        
        results = []
        current = start_row
        
        while current < end_row:
            chunk_end = min(current + chunk_size, end_row)
            
            ctx = EvaluationContext(self.data_source, current, chunk_end, lazy_engine=self)
            chunk_result = expr.evaluate(ctx)
            results.append(chunk_result)
            
            current = chunk_end
        
        return np.concatenate(results)
    
    def get_dependencies(self, name: str) -> Set[str]:
        """Get all dependencies for a signal (recursive)."""
        if name not in self._expressions:
            return set()
        
        deps = self._expressions[name].get_dependencies()
        
        # Resolve transitive dependencies
        all_deps = set(deps)
        for dep in deps:
            if dep in self._expressions:
                all_deps |= self.get_dependencies(dep)
        
        return all_deps
    
    def get_defined_signals(self) -> Dict[str, str]:
        """Get all defined signals and their expressions."""
        return self._raw_expressions.copy()
    
    def remove(self, name: str) -> None:
        """Remove a defined signal."""
        if name in self._expressions:
            del self._expressions[name]
            del self._raw_expressions[name]
    
    def clear(self) -> None:
        """Remove all defined signals."""
        self._expressions.clear()
        self._raw_expressions.clear()
    
    def __repr__(self) -> str:
        return f"LazyEngine(columns={len(self._known_columns)}, defined={len(self._expressions)})"


# =============================================================================
# Convenience Functions
# =============================================================================

def create_power_signal(engine: LazyEngine, name: str, voltage_col: str, current_col: str):
    """Define power = voltage * current"""
    engine.define(name, f"{voltage_col} * {current_col}")


def create_rms_signal(engine: LazyEngine, name: str, source_col: str):
    """Define RMS = sqrt(mean(x^2))"""
    engine.define(name, f"sqrt(mean({source_col}^2))")
