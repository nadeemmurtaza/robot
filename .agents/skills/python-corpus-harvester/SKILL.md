---
name: python-corpus-harvester
description: >-
  Harvests, parses, and formats official Python code from the standard library and docs.python.org.
  Use when the user asks to harvest Python code, expand the training dataset, extract stdlib functions,
  or build new corpora for Code Robot.
---

# 📚 Python Corpus Harvester Skill

This skill provides procedures for scraping, harvesting, and validating real Python code from the official Python standard library and documentation to feed Code Robot's curriculum.

---

## 🌾 Harvesting Pipeline

```
[Python Standard Library (419 modules)] + [docs.python.org/3/]
                          │
                          ▼
             [AST Syntax Validator & Filter]
             • Strips private helpers and dunder methods
             • Extracts clean docstrings and type annotations
             • Enforces valid syntax via ast.parse()
                          │
                          ▼
             [Corpus Dataset: data/official_python_corpus.json]
             • prompt: function signature + docstring
             • target: complete verified implementation
             • tests: auto-generated assertion checks
```

---

## 🛠️ Workflows & Commands

### 1. Run the Harvester Across Python Standard Library
Scans all standard library modules available in the local Python runtime:

```powershell
python -c "
from custom_ncm.official_python_harvester import harvest_and_save_dataset
dataset = harvest_and_save_dataset(
    output_path='data/official_python_corpus.json',
    target_count=2000
)
print(f'Successfully harvested {len(dataset)} verified functions!')
"
```

*What to verify:*
- Verify that `data/official_python_corpus.json` is created or updated.
- Confirm that each entry contains `prompt`, `target`, and `module`.

---

### 2. Inspect Harvested Dataset Quality
Verify sample entries and total counts:

```powershell
python -c "
import json
with open('data/official_python_corpus.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
print(f'Total harvested functions: {len(data)}')
print('Sample prompt:')
print(data[0]['prompt'])
print('Sample target:')
print(data[0]['target'])
"
```

---

### 3. Retrain on the Expanded Corpus
After harvesting new data, trigger the training loop:

```powershell
python train_on_official_python.py
```
