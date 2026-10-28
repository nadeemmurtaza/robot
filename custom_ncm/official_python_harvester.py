"""
Comprehensive Official Python.org & Standard Library Harvester.
Harvests verified real-world functions, algorithms, and idioms from:
1. The full Python 3.13 Standard Library (over 10,000 functions scanned)
2. Official Python Documentation tutorials & library guides (https://docs.python.org/3/)
"""
import os
import sys
import ast
import re
import json
import urllib.request
from typing import List, Tuple, Dict, Any, Optional

try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None

# Curated list of docs.python.org URLs covering fundamentals to advanced stdlib
DOCS_URLS = [
    "https://docs.python.org/3/tutorial/controlflow.html",
    "https://docs.python.org/3/tutorial/datastructures.html",
    "https://docs.python.org/3/tutorial/introduction.html",
    "https://docs.python.org/3/tutorial/modules.html",
    "https://docs.python.org/3/tutorial/classes.html",
    "https://docs.python.org/3/tutorial/errors.html",
    "https://docs.python.org/3/library/functions.html",
    "https://docs.python.org/3/library/stdtypes.html",
    "https://docs.python.org/3/library/math.html",
    "https://docs.python.org/3/library/itertools.html",
    "https://docs.python.org/3/library/collections.html",
    "https://docs.python.org/3/library/string.html",
    "https://docs.python.org/3/library/random.html"
]

# Standard library module priority lists
CORE_MODULE_PREFIXES = [
    "bisect", "heapq", "statistics", "random", "string", "functools", "copy",
    "colorsys", "fnmatch", "calendar", "difflib", "operator", "math", "collections",
    "itertools", "json", "csv", "pathlib", "datetime", "decimal", "fractions",
    "hashlib", "secrets", "base64", "uuid", "queue", "contextlib", "dataclasses"
]

class OfficialPythonHarvester:
    """
    Crawls, parses, and AST-verifies official Python documentation and standard library code.
    """
    def __init__(self, max_context_len: int = 128, max_prefix_len: int = 64):
        self.max_context_len = max_context_len
        self.max_prefix_len = max_prefix_len
        self.lib_path = os.path.dirname(os.__file__)

    def harvest_standard_library(self, max_samples: int = 1500) -> List[Dict[str, str]]:
        """
        Parses all Python Standard Library files using AST.
        Selects well-formed functions with clean docstrings and syntactically valid bodies.
        """
        harvested = []
        print(f"[*] Scanning Official Python Standard Library at: {self.lib_path}", flush=True)
        
        py_files = []
        for root, dirs, files in os.walk(self.lib_path):
            # Skip test suites, site-packages, and UI packages to keep code purely algorithmic
            r_lower = root.lower()
            if any(skip in r_lower for skip in ['test', 'site-packages', 'idlelib', 'tkinter', 'turtledemo', 'venv']):
                continue
            for f in files:
                if f.endswith('.py') and not f.startswith('_'):
                    py_files.append(os.path.join(root, f))
                    
        # Sort priority modules first
        def priority_score(path: str) -> int:
            base = os.path.basename(path).lower()
            for idx, p in enumerate(CORE_MODULE_PREFIXES):
                if base.startswith(p):
                    return idx
            return 999
            
        py_files.sort(key=priority_score)
        print(f"[*] Found {len(py_files)} standard library modules to harvest.", flush=True)
        
        for file_path in py_files:
            if len(harvested) >= max_samples:
                break
                
            mod_name = os.path.relpath(file_path, self.lib_path).replace("\\", "/")
            try:
                with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                    content = f.read()
                    
                tree = ast.parse(content)
                source_lines = content.splitlines()
                
                for node in ast.walk(tree):
                    if len(harvested) >= max_samples:
                        break
                        
                    if isinstance(node, ast.FunctionDef):
                        # Filter out internal dunder methods
                        if node.name.startswith("__") and node.name.endswith("__"):
                            continue
                        if not node.body:
                            continue
                            
                        # Extract docstring if present
                        docstring = ast.get_docstring(node) or ""
                        doc_summary = docstring.strip().split("\n")[0] if docstring else f"Function {node.name}"
                        if len(doc_summary) > 70:
                            doc_summary = doc_summary[:67] + "..."
                        doc_summary = doc_summary.replace("\r", "").replace("\n", " ")
                        
                        # Extract signature line from args
                        args_list = [a.arg for a in node.args.args]
                        sig_line = f"def {node.name}(" + ", ".join(args_list) + "):\n"
                        prompt = f"# {doc_summary}\n{sig_line}"
                        
                        # Body lines start after docstring or first statement
                        body_start = node.body[0].lineno - 1
                        if isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant) and isinstance(node.body[0].value.value, str):
                            if len(node.body) > 1:
                                body_start = node.body[1].lineno - 1
                                
                        body_end = min(node.end_lineno, body_start + 6)
                        body_lines = source_lines[body_start:body_end]
                        if not body_lines:
                            continue
                            
                        body_text = "\n".join(body_lines) + "\n"
                        
                        # Verify combined code forms valid Python AST
                        test_code = prompt + body_text
                        try:
                            ast.parse(test_code)
                            harvested.append({
                                "source": f"stdlib:{mod_name}",
                                "func_name": node.name,
                                "prompt": prompt,
                                "body": body_text
                            })
                        except Exception:
                            continue
            except Exception:
                continue
                
        print(f"[+] Harvested {len(harvested)} verified functions from Python Standard Library!", flush=True)
        return harvested

    def harvest_official_docs(self) -> List[Dict[str, str]]:
        """
        Fetches code examples and doctest blocks directly from docs.python.org.
        """
        harvested = []
        if BeautifulSoup is None:
            print("[!] BeautifulSoup not available; skipping HTML docs scraping.", flush=True)
            return harvested
            
        print("[*] Connecting to official Python documentation: docs.python.org...", flush=True)
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) PythonDocumentationIngest/1.0"}
        
        for url in DOCS_URLS:
            try:
                req = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(req, timeout=6) as resp:
                    html = resp.read().decode("utf-8", errors="replace")
                    
                soup = BeautifulSoup(html, "html.parser")
                code_blocks = soup.find_all("div", class_="highlight-python")
                
                for block in code_blocks:
                    text = block.get_text()
                    lines = text.splitlines()
                    clean_lines = [re.sub(r'^(>>>|\.\.\.)\s?', '', line) for line in lines]
                    code_snippet = "\n".join(clean_lines)
                    
                    try:
                        tree = ast.parse(code_snippet)
                        for node in ast.walk(tree):
                            if isinstance(node, ast.FunctionDef):
                                docstring = ast.get_docstring(node) or f"Official docs: {node.name}"
                                doc_summary = docstring.strip().split("\n")[0]
                                if len(doc_summary) > 70:
                                    doc_summary = doc_summary[:67] + "..."
                                    
                                sig = f"def {node.name}(" + ", ".join(a.arg for a in node.args.args) + "):\n"
                                prompt = f"# {doc_summary}\n{sig}"
                                
                                body_nodes = [
                                    b for b in node.body
                                    if not (isinstance(b, ast.Expr) and isinstance(b.value, ast.Constant) and isinstance(b.value.value, str))
                                ]
                                if not body_nodes:
                                    continue
                                    
                                body_segments = [ast.get_source_segment(code_snippet, b) for b in body_nodes[:4]]
                                body_segments = [b for b in body_segments if b is not None]
                                if not body_segments:
                                    continue
                                    
                                body_text = "\n".join("    " + l.strip() for l in "\n".join(body_segments).splitlines()) + "\n"
                                test_code = prompt + body_text
                                ast.parse(test_code)
                                
                                harvested.append({
                                    "source": "docs.python.org",
                                    "func_name": node.name,
                                    "prompt": prompt,
                                    "body": body_text
                                })
                    except Exception:
                        continue
            except Exception as e:
                print(f"   [-] docs.python.org ({os.path.basename(url)}): {e}", flush=True)
                
        print(f"[+] Harvested {len(harvested)} code patterns from docs.python.org!", flush=True)
        return harvested

    def harvest_all(self, save_path: str = "data/official_python_corpus.json") -> List[Dict[str, str]]:
        """Harvests from both Standard Library and docs.python.org, deduplicating and saving."""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        all_data = []
        all_data.extend(self.harvest_standard_library())
        all_data.extend(self.harvest_official_docs())
        
        seen = set()
        unique_corpus = []
        for item in all_data:
            key = item["prompt"].strip()
            if key not in seen:
                seen.add(key)
                unique_corpus.append(item)
                
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(unique_corpus, f, indent=2)
            
        print(f"[OK] Successfully compiled Official Python Knowledge Base: {len(unique_corpus)} unique functions saved to {save_path}!", flush=True)
        return unique_corpus

if __name__ == "__main__":
    harvester = OfficialPythonHarvester()
    harvester.harvest_all()
