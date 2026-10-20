"""Tests for CodeParser and AST symbol extraction."""

from devmind.code_parser import CodeParser


def test_parse_python_file_extracts_symbols_and_imports(temp_project_dir):
    parser = CodeParser()
    utils_path = str(temp_project_dir / "utils.py")
    res = parser.parse_file(utils_path, "utils.py")

    assert res is not None
    assert res["is_parsed"] is True

    # Check symbols
    symbols = res["symbols"]
    symbol_names = [s.name for s in symbols]

    assert "add" in symbol_names
    assert "Calculator" in symbol_names
    assert "Calculator.multiply" in symbol_names

    # Check docstrings and parameters
    add_sym = next(s for s in symbols if s.name == "add")
    assert add_sym.parameters == ["a", "b"]
    assert add_sym.return_type == "int"
    assert "Add two numbers" in (add_sym.docstring or "")

    calc_sym = next(s for s in symbols if s.name == "Calculator")
    assert calc_sym.symbol_type == "class"


def test_symbol_search(temp_project_dir):
    parser = CodeParser()
    parser.parse_file(str(temp_project_dir / "utils.py"), "utils.py")

    exact_matches = parser.find_symbols("add", exact=True)
    assert len(exact_matches) == 1
    assert exact_matches[0].name == "add"

    # Search method by short name
    method_matches = parser.find_symbols("multiply", exact=True)
    assert len(method_matches) == 1
    assert method_matches[0].name == "Calculator.multiply"

    # Partial search
    partial_matches = parser.find_symbols("calc", exact=False)
    assert any("Calculator" in s.name for s in partial_matches)


def test_call_site_discovery(temp_project_dir):
    parser = CodeParser()
    parser.parse_file(str(temp_project_dir / "main.py"), "main.py")

    refs = parser.find_references("add")
    assert len(refs) >= 1
    assert refs[0]["rel_path"] == "main.py"


def test_module_relationships(temp_project_dir):
    parser = CodeParser()
    parser.parse_file(str(temp_project_dir / "main.py"), "main.py")
    parser.parse_file(str(temp_project_dir / "utils.py"), "utils.py")

    relationships = parser.get_module_relationships()
    assert "main.py" in relationships
    assert "utils.py" in relationships["main.py"]


def test_syntax_error_graceful_handling(tmp_path):
    bad_file = tmp_path / "broken.py"
    bad_file.write_text("def broken_func(:\n    pass\n", encoding="utf-8")

    parser = CodeParser()
    res = parser.parse_file(str(bad_file), "broken.py")

    assert res is not None
    assert res["is_parsed"] is False
    assert res["symbols"] == []
