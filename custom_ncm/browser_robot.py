"""
Browser Robot: Autonomous Web Browser Skill for Code Robot.
Capabilities:
1. Launch & Control Google Chrome via Playwright (headed or headless)
2. Smooth scrolling (page down, scroll to element, infinite scroll)
3. Smart element clicking (by selector, visible text, links, buttons)
4. Reading rendered visible text & DOM content cleanly
5. Understanding text: extracting code blocks, docstrings, section outlines, and semantic meaning
"""
from typing import List, Dict, Any, Optional
import time
import re
from playwright.sync_api import sync_playwright, Browser, Page, Playwright

class BrowserRobot:
    """
    Autonomous browser agent for navigating, scrolling, clicking,
    reading, and understanding web text and code.
    """
    def __init__(self, headless: bool = True, channel: str = "chrome", timeout: int = 30000):
        self.headless = headless
        self.channel = channel
        self.timeout = timeout
        self.playwright: Optional[Playwright] = None
        self.browser: Optional[Browser] = None
        self.page: Optional[Page] = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def start(self):
        """Initializes Playwright and launches Chrome."""
        if self.page is None:
            self.playwright = sync_playwright().start()
            try:
                self.browser = self.playwright.chromium.launch(
                    channel=self.channel,
                    headless=self.headless
                )
            except Exception:
                # Fallback to default chromium if specific channel fails
                self.browser = self.playwright.chromium.launch(headless=self.headless)
                
            self.page = self.browser.new_page(
                viewport={"width": 1280, "height": 800},
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 CodeRobot/1.0"
            )
            self.page.set_default_timeout(self.timeout)

    def close(self):
        """Gracefully closes browser and stops Playwright."""
        if self.page:
            try:
                self.page.close()
            except Exception:
                pass
            self.page = None
        if self.browser:
            try:
                self.browser.close()
            except Exception:
                pass
            self.browser = None
        if self.playwright:
            try:
                self.playwright.stop()
            except Exception:
                pass
            self.playwright = None

    def goto(self, url: str, timeout: int = 15000) -> str:
        """Navigates to URL with resilient fallback for slow third-party sites."""
        if not self.page:
            self.start()
        try:
            self.page.goto(url, wait_until="domcontentloaded", timeout=timeout)
        except Exception:
            try:
                # If domcontentloaded timed out (e.g. slow external tracker/analytics), fallback to commit
                self.page.goto(url, wait_until="commit", timeout=5000)
            except Exception:
                pass
        time.sleep(0.5)
        try:
            return self.page.title()
        except Exception:
            return url

    def scroll(self, direction: str = "down", pixels: int = 600) -> Dict[str, Any]:
        """
        Scrolls the page smoothly in given direction.
        direction: 'down', 'up', 'top', 'bottom'
        """
        if not self.page:
            raise RuntimeError("Browser is not open. Call goto() or start() first.")

        if direction == "down":
            self.page.evaluate(f"window.scrollBy(0, {pixels});")
        elif direction == "up":
            self.page.evaluate(f"window.scrollBy(0, -{pixels});")
        elif direction == "top":
            self.page.evaluate("window.scrollTo(0, 0);")
        elif direction == "bottom":
            self.page.evaluate("window.scrollTo(0, document.body.scrollHeight);")
            
        time.sleep(0.3)
        
        # Get current scroll position and total height
        scroll_info = self.page.evaluate("""() => ({
            scrollY: window.scrollY,
            innerHeight: window.innerHeight,
            scrollHeight: document.body.scrollHeight
        })""")
        return scroll_info

    def click(self, target: str, timeout: int = 5000) -> bool:
        """
        Clicks an element by CSS selector, text match, or aria label.
        Examples:
            bot.click('a:has-text(\"Next\")')
            bot.click('text=\"Getting Started\"')
            bot.click('button#run')
        """
        if not self.page:
            raise RuntimeError("Browser is not open.")

        # Try selector directly first
        try:
            self.page.click(target, timeout=timeout)
            time.sleep(0.5)
            return True
        except Exception:
            pass

        # Try matching by visible text
        try:
            loc = self.page.get_by_text(target, exact=False).first
            loc.click(timeout=timeout)
            time.sleep(0.5)
            return True
        except Exception:
            pass

        # Try matching by role button or link
        try:
            loc = self.page.get_by_role("link", name=target).first
            loc.click(timeout=timeout)
            time.sleep(0.5)
            return True
        except Exception:
            pass

        return False

    def click_next_page(self, timeout: int = 10000) -> bool:
        """
        Locates and clicks the 'Next' page / chapter button at the bottom of the page,
        waiting for the new page to load. Returns True if navigation to a new URL succeeded.
        """
        if not self.page:
            raise RuntimeError("Browser is not open.")

        # Scroll to the bottom first where pagination & next buttons live
        self.scroll("bottom")
        time.sleep(0.4)

        prev_url = self.page.url

        selectors = [
            # 1. Docusaurus / modern doc sites (e.g. Next > [Changing Title])
            'a.pagination-nav__link--next',
            '.pagination-nav__link--next',
            '.pagination-nav__item--next a',
            'a[class*="pagination"][class*="next"]',
            'a:has(.pagination-nav__sublabel:has-text("Next"))',
            
            # 2. VitePress / VuePress / MkDocs / Sphinx
            '.md-footer__link--next',
            'a[rel="next"]',
            'a.pager-link.next',
            '.prev-next .pager a.next',
            '.pagination-next a',
            '.next-page a',
            '.nav-next a',
            'a[title*="next chapter" i]',
            'a[title*="next page" i]',
            'a[title*="next" i]',
            
            # 3. Text patterns with chevrons or labels
            'a:text-is("Next »")',
            'a:text-is("Next >")',
            'a:text-is("Next ›")',
            'a:text-is("Next →")',
            'a:text-is("Next")',
            'a:has-text("Next >")',
            'a:has-text("Next ›")',
            'a:has-text("Next »")',
            'a:has-text("Next →")',
            'a:has-text("Next chapter")',
            'a:has-text("Next page")',
            'a:has-text("Next")',
            'button:has-text("Next")'
        ]

        for sel in selectors:
            try:
                loc = self.page.locator(sel)
                count = loc.count()
                if count > 0:
                    # Prefer the bottom-most / last next button on the page
                    target = loc.last
                    target.scroll_into_view_if_needed()
                    target.click(timeout=3000)
                    self.page.wait_for_load_state("domcontentloaded", timeout=timeout)
                    time.sleep(0.5)
                    if self.page.url != prev_url:
                        return True
            except Exception:
                continue

        # JavaScript deep fallback: Scans all links in the footer/pagination for "Next"
        try:
            clicked_js = self.page.evaluate("""() => {
                // 1. Direct class match for modern doc frameworks
                const el = document.querySelector(
                    'a.pagination-nav__link--next, .pagination-nav__link--next, .pagination-nav__item--next a, .md-footer__link--next, a[rel="next"], .pager a.next, a.next'
                );
                if (el) {
                    el.scrollIntoView();
                    el.click();
                    return true;
                }
                
                // 2. Scan anchors from bottom up (catches changing title under Next)
                const anchors = Array.from(document.querySelectorAll('a[href]'));
                for (let i = anchors.length - 1; i >= 0; i--) {
                    const a = anchors[i];
                    const fullText = (a.innerText || a.textContent || '').trim();
                    const lines = fullText.split('\\n').map(l => l.trim().toLowerCase());
                    // Matches "Next", "Next >", "Next »", "Next →" as any line in the button
                    if (lines.some(l => /^next(\\s*[>»›→]|$)/.test(l))) {
                        a.scrollIntoView();
                        a.click();
                        return true;
                    }
                }
                return false;
            }""")
            if clicked_js:
                self.page.wait_for_load_state("domcontentloaded", timeout=timeout)
                time.sleep(0.5)
                if self.page.url != prev_url:
                    return True
        except Exception:
            pass

        return False

    def extract_table_of_contents(self) -> List[Dict[str, str]]:
        """
        Universal Table of Contents (TOC) extractor across ALL documentation sites.
        Extracts all chapter, section, and sub-page URLs from sidebars, TOC trees, and navigation menus.
        Compatible with Sphinx, Docusaurus, VitePress, ReadTheDocs, MkDocs, GitBook, and custom docs.
        """
        if not self.page:
            raise RuntimeError("Browser is not open.")

        toc_items = self.page.evaluate("""() => {
            const results = [];
            const seenUrls = new Set();
            const currentHost = window.location.hostname;

            const tocSelectors = [
                // Docusaurus
                '.theme-doc-sidebar-menu',
                '.menu__list',
                'nav.menu',
                // Sphinx / ReadTheDocs
                '.wy-menu',
                '.toctree-wrapper',
                '.sphinxsidebar',
                '.sidebar-drawer',
                'ul.current',
                'div.toctree',
                // VitePress / VuePress
                '.VPSidebar',
                '.sidebar-links',
                // MkDocs / Material
                '.md-nav',
                '.md-sidebar--primary',
                // GitBook
                'nav[aria-label="Table of contents"]',
                '[data-testid="page-list"]',
                // Universal / HTML5 semantic fallbacks
                'nav.toc',
                '.table-of-contents',
                '.tableofcontents',
                'nav.sidebar',
                'aside.sidebar',
                'nav',
                'aside'
            ];

            let candidateElements = [];
            for (const sel of tocSelectors) {
                const els = document.querySelectorAll(sel);
                if (els.length > 0) {
                    els.forEach(el => candidateElements.push(el));
                }
            }

            if (candidateElements.length === 0) {
                candidateElements = [document.body];
            }

            for (const container of candidateElements) {
                const links = container.querySelectorAll('a[href]');
                for (const a of links) {
                    const text = (a.innerText || a.textContent || '').trim();
                    let href = a.href;
                    if (!href) continue;

                    try {
                        const urlObj = new URL(href, window.location.href);
                        if (urlObj.hostname !== currentHost) continue;
                        if (/\\.(zip|tar|gz|pdf|png|jpg|svg|exe)$/i.test(urlObj.pathname)) continue;
                        
                        const cleanUrl = urlObj.origin + urlObj.pathname;
                        if (text.length >= 2 && !seenUrls.has(cleanUrl)) {
                            seenUrls.add(cleanUrl);
                            results.push({
                                title: text.replace(/[\\r\\n]+/g, ' ').trim(),
                                url: cleanUrl
                            });
                        }
                    } catch (e) {
                        continue;
                    }
                }
                if (results.length >= 8) {
                    break;
                }
            }

            return results;
        }""")

        return toc_items

    def extract_all_links(self, include_third_party: bool = True) -> List[Dict[str, Any]]:
        """
        Extracts ALL links across the entire page (menus, sidebars, submenus,
        dropdowns, table of contents, article body, in-text citations, and footer).
        Includes third-party and external links when include_third_party=True.
        Filters out fragments, mailto/javascript, and binary/asset files.
        Works across ANY website and documentation framework.
        """
        if not self.page:
            raise RuntimeError("Browser is not open.")

        return self.page.evaluate("""(includeThirdParty) => {
            const links = [];
            const seen = new Set();
            const currentHost = window.location.hostname;
            const allAnchors = document.querySelectorAll('a[href]');

            for (const a of allAnchors) {
                try {
                    const u = new URL(a.href, window.location.href);
                    if (!['http:', 'https:'].includes(u.protocol)) continue;
                    if (/\\.(zip|tar|gz|bz2|xz|pdf|png|jpg|jpeg|gif|svg|ico|exe|whl|css|js|map|wasm)$/i.test(u.pathname)) continue;
                    
                    const isThirdParty = (u.hostname !== currentHost);
                    if (!includeThirdParty && isThirdParty) continue;

                    const cleanUrl = u.origin + u.pathname;
                    if (seen.has(cleanUrl)) continue;
                    seen.add(cleanUrl);

                    let title = (a.innerText || a.textContent || a.getAttribute('title') || a.getAttribute('aria-label') || '').trim();
                    title = title.replace(/\\s+/g, ' ');
                    if (!title) {
                        const segs = u.pathname.split('/').filter(Boolean);
                        title = segs.length > 0 ? segs[segs.length - 1] : (u.hostname || 'Page');
                    }

                    links.push({
                        title: title,
                        url: cleanUrl,
                        domain: u.hostname,
                        is_third_party: isThirdParty
                    });
                } catch (e) {
                    continue;
                }
            }
            return links;
        }""", include_third_party)

    def collect_full_site_manifest(
        self,
        start_url: str,
        max_pages: int = 0,
        max_discovery_depth: int = 3,
        include_third_party: bool = True,
        manifest_file: str = "crawl_manifest.json"
    ) -> List[Dict[str, Any]]:
        """
        Full Site Crawler: Systematically traverses ANY website to collect:
        - All pages
        - All sub-pages
        - Every link
        - Every menu and sub-menu
        - Third-party and external documentation links
        Generates and saves a master crawl manifest JSON file.
        """
        import urllib.parse
        import posixpath
        import json

        parsed_start = urllib.parse.urlparse(start_url)
        base_host = parsed_start.netloc

        # Determine path scope (e.g. "/3/" for docs.python.org/3/)
        p = parsed_start.path
        if p.endswith('.html') or p.endswith('.htm'):
            prefix = posixpath.dirname(p)
            if not prefix.endswith('/'):
                prefix += '/'
        else:
            prefix = p if p.endswith('/') else p + '/'

        clean_start = urllib.parse.urljoin(start_url, parsed_start.path)
        print(f"[*] Starting Full-Site Spider on: {clean_start}")
        print(f"[*] Scope: Host = {base_host} | Path Prefix = {prefix} | Include Third Party = {include_third_party}")

        internal_entries: List[Dict[str, Any]] = []
        third_party_entries: List[Dict[str, Any]] = []
        seen_urls = set()
        queue = [{"url": clean_start, "depth": 0, "title": "Start Page"}]
        visited_pages = set()

        seen_urls.add(clean_start)
        internal_entries.append({
            "title": "Start Page",
            "url": clean_start,
            "domain": base_host,
            "type": "internal",
            "status": "pending"
        })

        discovery_limit = 1000 if max_pages == 0 else max(max_pages * 3, 50)
        discovery_count = 0

        while queue:
            current = queue.pop(0)
            cur_url = current["url"]
            cur_depth = current["depth"]

            if cur_url in visited_pages:
                continue
            visited_pages.add(cur_url)
            discovery_count += 1

            try:
                page_title = self.goto(cur_url)
                # Update title in internal entries
                for m in internal_entries:
                    if m["url"] == cur_url:
                        m["title"] = page_title.strip() or m["title"]
                        break

                safe_t = page_title.encode('ascii', errors='replace').decode('ascii')
                total_found = len(internal_entries) + len(third_party_entries)
                print(f"  [Spider #{discovery_count}] ({len(internal_entries)} internal, {len(third_party_entries)} external links) -> {safe_t}")

                links = self.extract_all_links(include_third_party=include_third_party)
                for item in links:
                    l_url = item["url"]
                    l_title = item["title"]
                    is_tp = item.get("is_third_party", False)
                    l_domain = item.get("domain", "")

                    if not is_tp:
                        # Internal link filter: must match path prefix
                        l_parsed = urllib.parse.urlparse(l_url)
                        if not l_parsed.path.startswith(prefix):
                            continue

                    if l_url not in seen_urls:
                        seen_urls.add(l_url)
                        entry = {
                            "title": l_title,
                            "url": l_url,
                            "domain": l_domain,
                            "type": "third_party" if is_tp else "internal",
                            "status": "pending"
                        }

                        if is_tp:
                            third_party_entries.append(entry)
                        else:
                            internal_entries.append(entry)
                            # Queue internal pages for deeper link discovery
                            if cur_depth + 1 < max_discovery_depth and len(internal_entries) < discovery_limit:
                                queue.append({
                                    "url": l_url,
                                    "depth": cur_depth + 1,
                                    "title": l_title
                                })

                # Save intermediate manifest
                combined = []
                for idx, item in enumerate(internal_entries + third_party_entries, 1):
                    item_copy = dict(item)
                    item_copy["index"] = idx
                    combined.append(item_copy)

                with open(manifest_file, "w", encoding="utf-8") as f:
                    json.dump({
                        "start_url": clean_start,
                        "base_host": base_host,
                        "path_prefix": prefix,
                        "include_third_party": include_third_party,
                        "total_pages": len(combined),
                        "internal_count": len(internal_entries),
                        "third_party_count": len(third_party_entries),
                        "pages": combined
                    }, f, indent=2)

                if (len(internal_entries) + len(third_party_entries)) >= discovery_limit:
                    break

            except Exception as e:
                print(f"  [!] Notice during spidering of {cur_url}: {e}")
                continue

        # Build final unified manifest: internal chapters & pages first, followed by third-party links
        final_manifest: List[Dict[str, Any]] = []
        for idx, item in enumerate(internal_entries + third_party_entries, 1):
            item["index"] = idx
            final_manifest.append(item)

        with open(manifest_file, "w", encoding="utf-8") as f:
            json.dump({
                "start_url": clean_start,
                "base_host": base_host,
                "path_prefix": prefix,
                "include_third_party": include_third_party,
                "total_pages": len(final_manifest),
                "internal_count": len(internal_entries),
                "third_party_count": len(third_party_entries),
                "pages": final_manifest
            }, f, indent=2)

        print(f"\n[+] Master Crawl Manifest saved: {len(final_manifest)} total pages discovered across site!")
        print(f"    - Internal documentation pages : {len(internal_entries)}")
        print(f"    - Third-party & external links  : {len(third_party_entries)}")
        print(f"[+] Manifest file: {manifest_file}")
        return final_manifest

    def read_text(self, selector: Optional[str] = None) -> str:
        """
        Reads the FULL visible text from the page, prioritizing the main article/document body,
        cleaning out script tags, sidebars, navigation bars, and whitespace clutter.
        """
        if not self.page:
            raise RuntimeError("Browser is not open.")

        raw_text = self.page.evaluate("""(userSel) => {
            // Target the main content container if available
            let el = null;
            if (userSel) {
                el = document.querySelector(userSel);
            } else {
                el = document.querySelector('main, article, [role="main"], div.body, div.document, body');
            }
            if (!el) el = document.body;
            if (!el) return '';
            
            // Clone to avoid modifying live DOM
            const clone = el.cloneNode(true);
            
            // Remove boilerplate navigation, sidebars, ads, scripts
            const toRemove = clone.querySelectorAll(
                'script, style, noscript, svg, nav, footer, header, .sphinxsidebar, .related, .sidebar, aside'
            );
            toRemove.forEach(n => n.remove());
            
            return clone.innerText || clone.textContent || '';
        }""", selector)

        # Normalize multiple blank lines and whitespace
        cleaned = re.sub(r'\n\s*\n', '\n\n', raw_text.strip())
        return cleaned

    def read_paragraphs(self) -> List[str]:
        """
        Extracts all full paragraphs (<p>) from the main article body.
        """
        if not self.page:
            raise RuntimeError("Browser is not open.")

        paragraphs = self.page.evaluate("""() => {
            const results = [];
            const container = document.querySelector('main, article, [role="main"], div.body, div.document, body') || document.body;
            container.querySelectorAll('p').forEach(p => {
                const text = p.innerText.trim();
                if (text.length > 20) {
                    results.push(text);
                }
            });
            return results;
        }""")
        return paragraphs

    def read_code_blocks(self) -> List[Dict[str, str]]:
        """
        Finds all code blocks (<pre>, <code>) on the page along with their context.
        """
        if not self.page:
            raise RuntimeError("Browser is not open.")

        code_blocks = self.page.evaluate("""() => {
            const results = [];
            const pres = document.querySelectorAll('pre, .highlight, .codehilite, pre > code');
            pres.forEach((el, idx) => {
                const code = el.innerText.trim();
                if (code.length > 5) {
                    // Try to find preceding heading or paragraph as context
                    let context = '';
                    let prev = el.previousElementSibling;
                    while (prev && !context) {
                        if (/^H[1-6]|P$/i.test(prev.tagName)) {
                            context = prev.innerText.trim();
                        }
                        prev = prev.previousElementSibling;
                    }
                    results.push({
                        id: idx + 1,
                        code: code,
                        context: context
                    });
                }
            });
            return results;
        }""")
        return code_blocks

    def understand_page(self) -> Dict[str, Any]:
        """
        Semantic parsing of page content:
        Extracts title, headers, main outline, code snippets, and interactive links.
        """
        if not self.page:
            raise RuntimeError("Browser is not open.")

        title = self.page.title()
        url = self.page.url
        text = self.read_text("body")
        code_blocks = self.read_code_blocks()

        headings = self.page.evaluate("""() => {
            const items = [];
            document.querySelectorAll('h1, h2, h3').forEach(h => {
                const text = h.innerText.trim();
                if (text) {
                    items.push({
                        level: parseInt(h.tagName[1]),
                        text: text
                    });
                }
            });
            return items;
        }""")

        clickable_links = self.page.evaluate("""() => {
            const links = [];
            document.querySelectorAll('a[href]').forEach(a => {
                const text = a.innerText.trim();
                const href = a.getAttribute('href');
                if (text && href && !href.startsWith('#') && !href.startsWith('javascript:')) {
                    links.push({ text: text, href: href });
                }
            });
            return links.slice(0, 20); // Top 20 links
        }""")

        return {
            "title": title,
            "url": url,
            "headings": headings,
            "text_summary": text[:1000] + ("..." if len(text) > 1000 else ""),
            "full_text_length": len(text),
            "num_code_blocks": len(code_blocks),
            "code_blocks": code_blocks,
            "top_links": clickable_links
        }

    def screenshot(self, path: str = "browser_view.png", full_page: bool = False) -> str:
        """Saves a screenshot of the current page."""
        if not self.page:
            raise RuntimeError("Browser is not open.")
        self.page.screenshot(path=path, full_page=full_page)
        return path


def test_browser_skill():
    print("=" * 70)
    print("  SKILL #1: BROWSER AUTOMATION, SCROLLING, CLICKING, READING")
    print("=" * 70)

    with BrowserRobot(headless=True) as bot:
        # 1. Navigation
        url = "https://docs.python.org/3/tutorial/introduction.html"
        print(f"[*] Navigating to: {url}")
        title = bot.goto(url)
        print(f"[+] Loaded Page: {title}")

        # 2. Scrolling
        print("[*] Scrolling down (600px)...")
        scroll_info = bot.scroll("down", 600)
        print(f"[+] Scroll Position: {scroll_info['scrollY']}px / Total: {scroll_info['scrollHeight']}px")

        # 3. Reading Text
        print("[*] Reading clean rendered text from page...")
        text = bot.read_text()
        print(f"[+] Read {len(text):,} characters of clean text.")
        safe_excerpt = text[:300].encode('ascii', errors='replace').decode('ascii')
        print(f"[+] Excerpt:\n{safe_excerpt}...\n")

        # 4. Extracting & Understanding Code Blocks
        print("[*] Extracting Python code blocks from the page...")
        code_blocks = bot.read_code_blocks()
        print(f"[+] Found {len(code_blocks)} code blocks on the page!")
        if code_blocks:
            first = code_blocks[0]
            ctx = first['context'].encode('ascii', errors='replace').decode('ascii')
            code = first['code'][:200].encode('ascii', errors='replace').decode('ascii')
            print(f"  - Block 1 Context: {ctx}")
            print(f"  - Block 1 Code:\n{code}\n")

        # 5. Full Semantic Page Understanding
        print("[*] Understanding page structure & outline...")
        understanding = bot.understand_page()
        print(f"[+] Found {len(understanding['headings'])} major headings:")
        for h in understanding['headings'][:5]:
            indent = "  " * h['level']
            safe_h = h['text'].encode('ascii', errors='replace').decode('ascii')
            print(f"{indent}- H{h['level']}: {safe_h}")

        print(f"[+] Extracted {len(understanding['top_links'])} navigable links.")

        # 6. Clicking Elements
        print("[*] Testing element clicking (navigation link)...")
        if understanding['top_links']:
            target_link = understanding['top_links'][0]['text']
            print(f"[*] Clicking link: '{target_link}'")
            clicked = bot.click(target_link)
            print(f"[+] Click action result: {clicked}")
            print(f"[+] Current page title after click: {bot.page.title()}")

    print("\n" + "=" * 70)
    print("  SKILL #1 EXECUTION SUCCESSFUL!")
    print("=" * 70)


if __name__ == "__main__":
    test_browser_skill()
