---
name: browser-interaction-and-reading
description: >-
  Autonomous web browsing skill for Code Robot. Use when the user asks to browse web pages,
  navigate URLs, scroll, click interactive elements, read articles and documentation,
  extract code snippets, and semantically understand web text.
---

# 🌐 Skill #1: Browser Automation, Scrolling, Clicking, Reading & Understanding Text

This skill equips **Code Robot** with autonomous web browsing capabilities using Google Chrome and Playwright. It enables the robot to navigate real websites, scroll through long documentation, click links and interactive controls, read rendered text, and understand code and concepts.

---

## ⚡ Core Capabilities

```
[Navigate to URL]
       │
       ▼
[Smooth Scrolling] ──> Handles long documentation, lazy loading, and infinite scroll
       │
       ▼
[Element Clicking] ──> Clicks buttons, links, tabs, and interactive examples
       │
       ▼
[Clean Text Reading] ──> Strips ads/scripts and extracts high-fidelity semantic prose
       │
       ▼
[Text & Code Understanding] ──> Identifies H1-H3 outlines, extracts Python code blocks, and parses APIs
```

---

## 🛠️ Step-by-Step Python Workflows

### 1. Launch Browser & Navigate to Web Page
Uses the system Google Chrome instance via Playwright with zero downloads required:

```python
from custom_ncm.browser_robot import BrowserRobot

with BrowserRobot(headless=True) as bot:
    title = bot.goto("https://docs.python.org/3/tutorial/index.html")
    print(f"Loaded: {title}")
```

---

### 2. Scrolling Through Long Content
Supports directional scrolling, pixel amounts, or scrolling directly to the bottom:

```python
with BrowserRobot(headless=True) as bot:
    bot.goto("https://docs.python.org/3/tutorial/introduction.html")
    
    # Scroll down 600 pixels
    scroll_info = bot.scroll(direction="down", pixels=600)
    print(f"Scrolled to {scroll_info['scrollY']}px / {scroll_info['scrollHeight']}px")
    
    # Scroll all the way to the bottom to trigger lazy loading
    bot.scroll(direction="bottom")
```

---

### 3. Clicking Interactive Elements & Links
Smart clicking resolves targets by CSS selector, visible link text, or button role:

```python
with BrowserRobot(headless=True) as bot:
    bot.goto("https://docs.python.org/3/tutorial/introduction.html")
    
    # Click by visible link text
    bot.click("Table of Contents")
    print("New page title:", bot.page.title())
    
    # Click by CSS selector or button text
    # bot.click("button.run-code")
    # bot.click("a:has-text('Next')")
```

---

### 4. Reading Clean Rendered Text
Extracts readable prose while stripping away navigation bars, footer boilerplate, scripts, and stylesheets:

```python
with BrowserRobot(headless=True) as bot:
    bot.goto("https://docs.python.org/3/library/functions.html")
    clean_text = bot.read_text("body")
    print(f"Read {len(clean_text):,} characters of clean documentation!")
    print(clean_text[:500])
```

---

### 5. Extracting Code Blocks & Semantic Understanding
Parses the page hierarchy (H1, H2, H3), extracts all `<pre><code>` blocks, and discovers navigable links:

```python
with BrowserRobot(headless=True) as bot:
    bot.goto("https://docs.python.org/3/tutorial/introduction.html")
    
    # 1. Extract all Python code blocks with preceding context
    code_blocks = bot.read_code_blocks()
    print(f"Discovered {len(code_blocks)} code examples!")
    for b in code_blocks[:3]:
        print(f"Context: {b['context']}")
        print(f"Code:\n{b['code']}\n")
        
    # 2. Get full semantic breakdown
    analysis = bot.understand_page()
    print("Title:", analysis["title"])
    print("Headings:")
    for h in analysis["headings"]:
        print(f"  H{h['level']}: {h['text']}")
```

---

### 6. Full-Site Spider, Master Manifest & Live Neural Training
Traverses ANY documentation website, collects all pages, sub-pages, menus, sub-menus, and third-party links, compiles `crawl_manifest.json`, and systematically trains Code Robot live:

```bash
# Universal Full-Site Spider + Live Neural Training
python browse.py "https://docs.python.org/3/" --spider --train

# Or for ANY documentation website
python browse.py "https://fastapi.tiangolo.com/" --spider --train
python browse.py "https://react.dev/" --spider --train
```

---

## 🏃 Run the Self-Contained Test

To verify browser navigation, scrolling, clicking, reading, and understanding immediately:

```powershell
python custom_ncm/browser_robot.py
```

*Expected Output:*
- Navigates to `https://docs.python.org/3/tutorial/introduction.html`
- Scrolls down 600px
- Reads 19,000+ characters of clean text
- Extracts 80+ code blocks
- Understands 12+ page headings
- Automatically clicks the "Table of Contents" link and navigates to the next page
