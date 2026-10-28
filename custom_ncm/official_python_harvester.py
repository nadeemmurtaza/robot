"""
Official Python Harvester & Knowledge Ingestion Engine.
Harvests functions, docstrings, and verified code from:
1. The local Python Standard Library (e.g., bisect, heapq, statistics, random, functools, string)
2. The official Python Documentation (https://docs.python.org/3/)
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

# Canonical modules from the Python Standard Library with clean algorithmic code
CANONICAL_MODULES = [
    "bisect.py",
    "heapq.py",
    "statistics.py",
    "random.py",
    "string.py",
    "functools.py",
    "copy.py",
    "colorsys.py",
    "fnmatch.py",
    "calendar.py",
    "difflib.py",
    "operator.py"
]

# Official tutorial documentation URLs from docs.python.org
DOCS_URLS = [
    "https://docs.python.org/3/tutorial/controlflow.html",
    "https://docs.python.org/3/tutorial/datastructures.html",
    "https://docs.python.org/3/tutorial/introduction.html"
]

class OfficialPythonHarvester:
    """
    Crawls and parses official Python documentation and standard library source code.
    """
    def __init__(self, max_context_len: int = 64, max_prefix_len: int = 40):
        self.max_context_len = max_context_len
        self.max_prefix_len = max_prefix_len
        self.lib_path = os.path.dirname(os.__file__)

    def harvest_standard_library(self) -> List[Dict[str, str]]:
        """
        Parses official Python Standard Library files using Python's AST parser.
        Extracts clean function signatures, docstrings, and bodies.
        """
        harvested = []
        print(f"[*] Scanning Official Python Standard Library at: {self.lib_path}", flush=True)
        
        for mod_name in CANONICAL_MODULES:
            file_path = os.path.join(self.lib_path, mod_name)
            if not os.path.isfile(file_path):
                continue
                
            try:
                with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                    content = f.read()
                    
                tree = ast.parse(content)
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        # Filter out internal/private dunder methods or ultra-long functions
                        if node.name.startswith("__") and node.name.endswith("__"):
                            continue
                        if not node.body:
                            continue
                            
                        # Extract docstring if present
                        docstring = ast.get_docstring(node) or ""
                        doc_summary = docstring.strip().split("\n")[0] if docstring else f"Function {node.name}"
                        # Truncate doc summary to clean length
                        if len(doc_summary) > 60:
                            doc_summary = doc_summary[:57] + "..."
                            
                        # Extract signature line from source
                        sig_line = f"def {node.name}("
                        args_list = []
                        for arg in node.args.args:
                            args_list.append(arg.arg)
                        sig_line += ", ".join(args_list) + "):\n"
                        
                        prompt = f"# {doc_summary}\n{sig_line}"
                        
                        # Extract body lines from segment
                        source_lines = content.splitlines()
                        # Body starts after def/docstring
                        body_start = node.body[0].lineno - 1
                        if isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant) and isinstance(node.body[0].value.value, str):
                            # Skip docstring node
                            if len(node.body) > 1:
                                body_start = node.body[1].lineno - 1
                                
                        body_end = node.end_lineno
                        # Collect 1-4 lines of body
                        body_lines = source_lines[body_start:min(body_start + 4, body_end)]
                        body_text = "\n".join(body_lines) + "\n"
                        
                        # Verify syntax of definition + body
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
            except Exception as e:
                print(f"   [!] Error parsing {mod_name}: {e}", flush=True)
                
        print(f"[+] Harvested {len(harvested)} functions from Python Standard Library!", flush=True)
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
        
        headers = {"User-Agent": "PythonNextConceptLearner/1.0"}
        for url in DOCS_URLS:
            try:
                req = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(req, timeout=8) as resp:
                    html = resp.read().decode("utf-8", errors="replace")
                    
                soup = BeautifulSoup(html, "html.parser")
                code_blocks = soup.find_all("div", class_="highlight-python")
                
                for block in code_blocks:
                    text = block.get_text()
                    # Look for functions in doctests or code examples
                    lines = text.splitlines()
                    clean_lines = []
                    for line in lines:
                        clean_line = re.sub(r'^(>>>|\.\.\.)\s?', '', line)
                        clean_lines.append(clean_line)
                    code_snippet = "\n".join(clean_lines)
                    
                    try:
                        tree = ast.parse(code_snippet)
                        for node in ast.walk(tree):
                            if isinstance(node, ast.FunctionDef):
                                docstring = ast.get_docstring(node) or f"Official docs example: {node.name}"
                                doc_summary = docstring.strip().split("\n")[0]
                                if len(doc_summary) > 60:
                                    doc_summary = doc_summary[:57] + "..."
                                    
                                sig = f"def {node.name}(" + ", ".join(a.arg for a in node.args.args) + "):\n"
                                prompt = f"# {doc_summary}\n{sig}"
                                
                                body_lines = [
                                    ast.get_source_segment(code_snippet, b) for b in node.body
                                    if not (isinstance(b, ast.Expr) and isinstance(b.value, ast.Constant) and isinstance(b.value.value, str))
                                ]
                                body_lines = [b for b in body_lines if b is not None]
                                if not body_lines:
                                    continue
                                    
                                body_text = "\n".join("    " + l.strip() for l in "\n".join(body_lines[:3]).splitlines()) + "\n"
                                
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
                print(f"   [!] Error fetching {url}: {e}", flush=True)
                
        print(f"[+] Harvested {len(harvested)} code patterns from docs.python.org!", flush=True)
        return harvested

    def harvest_all(self, save_path: str = "data/official_python_corpus.json") -> List[Dict[str, str]]:
        """Harvests from both Standard Library and docs.python.org, deduplicating and saving."""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        all_data = []
        all_data.extend(self.harvest_standard_library())
        all_data.extend(self.harvest_official_docs())
        
        # Deduplicate by prompt
        seen = set()
        unique_corpus = []
        for item in all_data:
            if item["prompt"] not in seen:
                seen.add(item["prompt"])
                unique_corpus.append(item)
                
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(unique_corpus, f, indent=2)
            
        print(f"[OK] Successfully compiled Official Python Knowledge Base: {len(unique_corpus)} unique functions saved to {save_path}!", flush=True)
        return unique_corpus

if __name__ == "__main__":
    harvester = OfficialPythonHarvester()
    harvester.harvest_all()
