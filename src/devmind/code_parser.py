"""Codebase parser leveraging Python's AST module for semantic symbol analysis."""

import ast
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from devmind.models import ImportInfo, SymbolInfo
from devmind.utils import safe_read_file


class SymbolVisitor(ast.NodeVisitor):
    """AST visitor that extracts functions, classes, imports, and call sites."""

    def __init__(self, file_path: str, rel_path: str):
        self.file_path = file_path
        self.rel_path = rel_path
        self.symbols: List[SymbolInfo] = []
        self.imports: List[ImportInfo] = []
        self.calls: List[Dict[str, Any]] = []
        self.current_class: Optional[str] = None
        self.current_function: Optional[str] = None

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self.imports.append(
                ImportInfo(
                    module=alias.name,
                    names=[alias.name],
                    alias=alias.asname,
                    file_path=self.file_path,
                    line_number=node.lineno,
                    is_from=False,
                )
            )
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module = node.module or ""
        names = [alias.name for alias in node.names]
        self.imports.append(
            ImportInfo(
                module=module,
                names=names,
                alias=node.names[0].asname if len(node.names) == 1 else None,
                file_path=self.file_path,
                line_number=node.lineno,
                is_from=True,
            )
        )
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        prev_class = self.current_class
        self.current_class = node.name

        docstring = ast.get_docstring(node)
        base_classes = []
        for base in node.bases:
            if isinstance(base, ast.Name):
                base_classes.append(base.id)
            elif isinstance(base, ast.Attribute):
                base_classes.append(f"{self._get_attr_name(base)}")

        end_line = getattr(node, "end_lineno", node.lineno)
        self.symbols.append(
            SymbolInfo(
                name=node.name,
                symbol_type="class",
                file_path=self.file_path,
                rel_path=self.rel_path,
                start_line=node.lineno,
                end_line=end_line,
                docstring=docstring,
                parameters=base_classes,  # Record base classes in parameters field
            )
        )

        self.generic_visit(node)
        self.current_class = prev_class

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._handle_function(node, is_async=False)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._handle_function(node, is_async=True)

    def _handle_function(self, node: Any, is_async: bool) -> None:
        prev_func = self.current_function
        func_name = node.name
        qualified_name = f"{self.current_class}.{func_name}" if self.current_class else func_name
        self.current_function = qualified_name

        docstring = ast.get_docstring(node)
        params: List[str] = [arg.arg for arg in node.args.args]
        return_annotation = None
        if node.returns:
            if isinstance(node.returns, ast.Name):
                return_annotation = node.returns.id
            elif isinstance(node.returns, ast.Constant):
                return_annotation = str(node.returns.value)

        end_line = getattr(node, "end_lineno", node.lineno)
        symbol_type = (
            "method" if self.current_class else ("async_function" if is_async else "function")
        )

        # Collect function calls made inside this function
        func_calls: List[str] = []
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                name = self._get_call_name(child.func)
                if name:
                    func_calls.append(name)

        self.symbols.append(
            SymbolInfo(
                name=qualified_name,
                symbol_type=symbol_type,
                file_path=self.file_path,
                rel_path=self.rel_path,
                start_line=node.lineno,
                end_line=end_line,
                docstring=docstring,
                parameters=params,
                return_type=return_annotation,
                calls=list(set(func_calls)),
            )
        )

        self.generic_visit(node)
        self.current_function = prev_func

    def visit_Call(self, node: ast.Call) -> None:
        call_name = self._get_call_name(node.func)
        if call_name:
            self.calls.append(
                {
                    "name": call_name,
                    "line": node.lineno,
                    "file_path": self.file_path,
                    "rel_path": self.rel_path,
                    "caller": self.current_function,
                }
            )
        self.generic_visit(node)

    def _get_call_name(self, func_node: ast.AST) -> Optional[str]:
        if isinstance(func_node, ast.Name):
            return func_node.id
        elif isinstance(func_node, ast.Attribute):
            val = self._get_call_name(func_node.value)
            return f"{val}.{func_node.attr}" if val else func_node.attr
        return None

    def _get_attr_name(self, node: ast.Attribute) -> str:
        if isinstance(node.value, ast.Name):
            return f"{node.value.id}.{node.attr}"
        elif isinstance(node.value, ast.Attribute):
            return f"{self._get_attr_name(node.value)}.{node.attr}"
        return node.attr


class CodeParser:
    """Orchestrates parsing of source files, extracting symbols, imports, and relationships."""

    def __init__(self):
        self.parsed_files: Dict[str, Dict[str, Any]] = {}

    def parse_file(self, file_path: str, rel_path: str = "") -> Optional[Dict[str, Any]]:
        """Parse a source file. Uses AST for Python, graceful metadata for others."""
        content = safe_read_file(file_path)
        if content is None:
            return None

        rel_path = rel_path or file_path
        ext = Path(file_path).suffix.lower()

        if ext == ".py":
            try:
                tree = ast.parse(content, filename=file_path)
                visitor = SymbolVisitor(file_path=file_path, rel_path=rel_path)
                visitor.visit(tree)

                result = {
                    "rel_path": rel_path,
                    "file_path": file_path,
                    "symbols": visitor.symbols,
                    "imports": visitor.imports,
                    "calls": visitor.calls,
                    "line_count": len(content.splitlines()),
                    "is_parsed": True,
                }
                self.parsed_files[rel_path] = result
                return result
            except SyntaxError:
                # Code might have syntax issues; don't crash
                result = {
                    "rel_path": rel_path,
                    "file_path": file_path,
                    "symbols": [],
                    "imports": [],
                    "calls": [],
                    "line_count": len(content.splitlines()),
                    "is_parsed": False,
                }
                self.parsed_files[rel_path] = result
                return result

        # Non-Python files: basic line count and metadata
        result = {
            "rel_path": rel_path,
            "file_path": file_path,
            "symbols": [],
            "imports": [],
            "calls": [],
            "line_count": len(content.splitlines()),
            "is_parsed": False,
        }
        self.parsed_files[rel_path] = result
        return result

    def find_symbols(
        self, query: str, exact: bool = False
    ) -> List[SymbolInfo]:
        """Find symbols across all parsed files by name matching."""
        query_clean = query.strip().lower()
        matched: List[SymbolInfo] = []

        for info in self.parsed_files.values():
            for symbol in info.get("symbols", []):
                sym_name = symbol.name.lower()
                # Check base name without class prefix if method
                short_name = sym_name.split(".")[-1]
                if exact:
                    if sym_name == query_clean or short_name == query_clean:
                        matched.append(symbol)
                else:
                    if query_clean in sym_name or query_clean in short_name:
                        matched.append(symbol)

        return matched

    def find_references(self, symbol_name: str) -> List[Dict[str, Any]]:
        """Find call-sites and references to a symbol across the project."""
        query = symbol_name.strip()
        short_query = query.split(".")[-1]
        references: List[Dict[str, Any]] = []

        for rel_path, info in self.parsed_files.items():
            for call in info.get("calls", []):
                call_name = call.get("name", "")
                call_short = call_name.split(".")[-1]
                if call_name == query or call_short == short_query:
                    references.append(call)

        return references

    def get_module_relationships(self) -> Dict[str, List[str]]:
        """Build dependency graph mapping each file to other files it imports."""
        relationships: Dict[str, List[str]] = {}
        all_rel_paths = list(self.parsed_files.keys())

        for rel_path, info in self.parsed_files.items():
            deps: Set[str] = set()
            for imp in info.get("imports", []):
                mod = imp.module
                # Check if module matches any local file
                for other in all_rel_paths:
                    other_stem = Path(other).stem
                    if mod == other_stem or mod.endswith(f".{other_stem}"):
                        if other != rel_path:
                            deps.add(other)
            relationships[rel_path] = sorted(list(deps))

        return relationships

    def get_project_summary(self) -> Dict[str, Any]:
        """Produce an overview of repository structure, symbols, and dependencies."""
        total_files = len(self.parsed_files)
        total_lines = sum(info.get("line_count", 0) for info in self.parsed_files.values())
        all_symbols: List[SymbolInfo] = []
        for info in self.parsed_files.values():
            all_symbols.extend(info.get("symbols", []))

        classes = [s for s in all_symbols if s.symbol_type == "class"]
        functions = [
            s for s in all_symbols if s.symbol_type in ("function", "async_function", "method")
        ]
        relationships = self.get_module_relationships()

        return {
            "total_files": total_files,
            "total_lines": total_lines,
            "classes_count": len(classes),
            "functions_count": len(functions),
            "classes": [c.name for c in classes],
            "functions": [f.name for f in functions],
            "module_dependencies": relationships,
            "files": list(self.parsed_files.keys()),
        }
