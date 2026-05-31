class PythonSymbolCollector:
    def visit_FunctionDef(self, node):
        return {"kind": "function", "name": node.name}

    def visit_AsyncFunctionDef(self, node):
        return None

    def visit_ClassDef(self, node):
        return {"kind": "class", "name": node.name}


def parse_repo(files):
    symbols = []
    for file_record in files:
        symbols.extend(parse_file(file_record))
    return symbols


def parse_file(file_record):
    if not file_record["path"].endswith(".py"):
        return []
    return extract_symbols(file_record["path"], file_record["content"])


def extract_symbols(path, content):
    collector = PythonSymbolCollector()
    rows = []
    for line in content.splitlines():
        if line.startswith("async def "):
            result = collector.visit_AsyncFunctionDef(type("Node", (), {"name": line.split()[2].split("(")[0]}))
        elif line.startswith("def "):
            result = collector.visit_FunctionDef(type("Node", (), {"name": line.split()[1].split("(")[0]}))
        elif line.startswith("class "):
            result = collector.visit_ClassDef(type("Node", (), {"name": line.split()[1].split(":")[0]}))
        else:
            result = None
        if result:
            rows.append({"id": f"{path}::{result['name']}", "path": path, "name": result["name"], "calls": []})
    return rows


def extract_imports(content):
    return [line for line in content.splitlines() if line.startswith("import ")]
