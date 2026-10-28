"""
Browse & Read: Autonomous Web Interaction CLI for Code Robot.
Usage:
    python browse.py "https://your-link-here.com"
    python browse.py "https://your-link-here.com" --head
    python browse.py "https://your-link-here.com" --click "Next"
"""
import sys
import time
import argparse
from custom_ncm.browser_robot import BrowserRobot

def main():
    parser = argparse.ArgumentParser(
        description="Autonomous Browser Agent: Navigate, scroll, click, read, and understand any webpage."
    )
    parser.add_argument(
        "url",
        type=str,
        help="The URL of the webpage you want to browse and read."
    )
    parser.add_argument(
        "--head",
        action="store_true",
        help="Run in visible Chrome browser window (default is headless)."
    )
    parser.add_argument(
        "--scroll",
        type=int,
        default=800,
        help="Pixels to scroll down (default: 800px)."
    )
    parser.add_argument(
        "--click",
        type=str,
        default=None,
        help="Text or selector of a link/button to click after loading."
    )
    parser.add_argument(
        "--screenshot",
        type=str,
        default=None,
        help="Optional filename to save a screenshot (e.g. page.png)."
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Print the entire extracted text in the terminal."
    )
    parser.add_argument(
        "--save",
        type=str,
        default="latest_webpage.txt",
        help="File path to save the full extracted text and code (default: latest_webpage.txt)."
    )
    parser.add_argument(
        "--next",
        action="store_true",
        help="Continually read, scroll, and click 'Next' at the bottom of each page until you stop it (Ctrl+C)."
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Seconds to pause on each page before clicking Next (default: 1.0s)."
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=0,
        help="Maximum pages to read in continuous loop (0 = infinite until stopped with Ctrl+C)."
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Train Code Robot live on every webpage as it reads (updates neural weights in real-time)."
    )
    parser.add_argument(
        "--toc",
        action="store_true",
        help="Universal TOC Mode: Extract the entire Table of Contents and read/learn every page and sub-page one by one."
    )
    args = parser.parse_args()

    print("=" * 70)
    print("  CODE ROBOT: AUTONOMOUS BROWSER & READING AGENT")
    print("=" * 70)
    print(f"[*] Target URL : {args.url}")
    print(f"[*] Window Mode: {'Visible Chrome (Headed)' if args.head else 'Headless Chrome'}")
    print(f"[*] Live Train : {'ENABLED (Model weights update on each page)' if args.train else 'Disabled (Reading & Scraping only)'}")

    learner = None
    if args.train:
        from custom_ncm.online_learner import OnlineCodeLearner
        print("[*] Initializing Online Neural Learner (Live Training Active)...")
        learner = OnlineCodeLearner()

    current_url = args.url
    page_count = 0
    total_characters = 0
    total_words = 0
    total_code_blocks = 0

    with BrowserRobot(headless=not args.head) as bot:
        try:
            # ---------------------------------------------------------
            # MODE A: UNIVERSAL TABLE OF CONTENTS (TOC) CRAWLER & LEARNER
            # ---------------------------------------------------------
            if args.toc:
                print("\n[*] Navigating to documentation URL to extract Table of Contents...")
                bot.goto(args.url)
                toc_items = bot.extract_table_of_contents()
                print(f"[+] Discovered {len(toc_items)} chapters & sub-pages in Table of Contents:")
                for idx, item in enumerate(toc_items[:6], 1):
                    safe_t = item['title'].encode('ascii', errors='replace').decode('ascii')
                    print(f"    {idx}. {safe_t} -> {item['url']}")
                if len(toc_items) > 6:
                    print(f"    ... and {len(toc_items) - 6} more sub-pages in Table of Contents!")

                # Queue up pages to visit
                urls_to_visit = [item['url'] for item in toc_items]
                if args.url not in urls_to_visit:
                    urls_to_visit.insert(0, args.url)

                for item_idx, visit_url in enumerate(urls_to_visit, 1):
                    page_count += 1
                    print("\n" + "=" * 70)
                    print(f"  [TOC PAGE {item_idx}/{len(urls_to_visit)}] OPENING: {visit_url}")
                    print("=" * 70)

                    title = bot.goto(visit_url)
                    safe_title = title.encode('ascii', errors='replace').decode('ascii')
                    print(f"  [+] Page Title: {safe_title}")

                    # Scroll through page
                    print(f"  [*] Scrolling through content ({args.scroll}px)...")
                    scroll_info = bot.scroll("down", args.scroll)
                    print(f"  [+] Scroll: {scroll_info['scrollY']}px / Total: {scroll_info['scrollHeight']}px")

                    # Read full text, paragraphs, code blocks
                    text = bot.read_text()
                    paragraphs = bot.read_paragraphs()
                    words = len(text.split())
                    code_blocks = bot.read_code_blocks()

                    total_characters += len(text)
                    total_words += words
                    total_code_blocks += len(code_blocks)

                    print(f"  [+] Extracted: {len(text):,} chars | {words:,} words | {len(paragraphs)} paragraphs | {len(code_blocks)} code blocks")

                    # Live Neural Training
                    if learner and code_blocks:
                        print(f"  [+] [Live Neural Training] Ingesting {len(code_blocks)} code snippets into model weights...")
                        train_res = learner.train_on_page(
                            title=title,
                            code_blocks=code_blocks,
                            text_paragraphs=paragraphs
                        )
                        if train_res['loss'] is not None:
                            print(f"  [+] Training Loss: {train_res['loss']:.4f} | Weights Updated -> {train_res['checkpoint']}")

                    # Save to corpus
                    if args.save:
                        mode = 'w' if page_count == 1 else 'a'
                        with open(args.save, mode, encoding='utf-8') as f:
                            f.write(f"\n{'='*70}\n")
                            f.write(f"TOC PAGE #{page_count}: {title}\nURL: {visit_url}\n")
                            f.write(f"Stats: {len(text):,} chars, {words:,} words, {len(code_blocks)} code blocks\n")
                            f.write(f"{'-'*70}\n\n")
                            f.write(text)
                            f.write("\n\n--- CODE BLOCKS ---\n\n")
                            for cb in code_blocks:
                                f.write(f"### {cb['context']}\n{cb['code']}\n\n")
                        print(f"  [+] Saved to corpus: {args.save}")

                    if args.full:
                        print("\n" + text.encode('ascii', errors='replace').decode('ascii'))

                    if args.max_pages > 0 and page_count >= args.max_pages:
                        print(f"\n[+] Reached requested max page limit ({args.max_pages}). Done!")
                        break

                    time.sleep(args.delay)

            # ---------------------------------------------------------
            # MODE B: CONTINUOUS 'NEXT >' BUTTON TRAVERSAL
            # ---------------------------------------------------------
            else:
                while True:
                    page_count += 1
                    print("\n" + "=" * 70)
                    print(f"  [PAGE #{page_count}] READING: {current_url}")
                    print("=" * 70)

                    # 1. Navigation
                    title = bot.goto(current_url)
                    safe_title = title.encode('ascii', errors='replace').decode('ascii')
                    print(f"  [+] Page Title: {safe_title}")

                    # 2. Scrolling
                    print(f"  [*] Scrolling through page content ({args.scroll}px)...")
                    scroll_info = bot.scroll("down", args.scroll)
                    print(f"  [+] Scroll: {scroll_info['scrollY']}px / Total Height: {scroll_info['scrollHeight']}px")

                    # 3. Optional user click on first page
                    if args.click and page_count == 1:
                        print(f"  [+] Executing initial click: '{args.click}'...")
                        bot.click(args.click)
                        time.sleep(0.5)

                    # 4. Reading the ENTIRE Page Text
                    print(f"  [*] Extracting FULL text from page...")
                    text = bot.read_text()
                    paragraphs = bot.read_paragraphs()
                    words = len(text.split())
                    code_blocks = bot.read_code_blocks()

                    total_characters += len(text)
                    total_words += words
                    total_code_blocks += len(code_blocks)

                    print(f"  [+] Extracted: {len(text):,} chars | {words:,} words | {len(paragraphs)} paragraphs | {len(code_blocks)} code blocks")

                    if paragraphs:
                        p0 = paragraphs[0].encode('ascii', errors='replace').decode('ascii')
                        print(f"  [+] First Paragraph: \"{p0[:200]}...\"")

                    # 5. Understanding: Code Blocks & Outline
                    if code_blocks:
                        first = code_blocks[0]
                        ctx = first['context'].encode('ascii', errors='replace').decode('ascii')
                        snippet = first['code'][:180].encode('ascii', errors='replace').decode('ascii')
                        print(f"  [+] Sample Code Block (Context: {ctx or 'General'}):")
                        print(f"```python\n{snippet}\n```")

                    understanding = bot.understand_page()
                    if understanding['headings']:
                        print(f"  [+] Page Sections:")
                        for h in understanding['headings'][:5]:
                            indent = "  " * h['level']
                            h_text = h['text'].encode('ascii', errors='replace').decode('ascii')
                            print(f"    {indent}* {h_text}")

                    # 6. Live Neural Training on Page Content
                    if learner and code_blocks:
                        print(f"  [+] [Live Neural Training] Ingesting {len(code_blocks)} code snippets into model weights...")
                        train_res = learner.train_on_page(
                            title=title,
                            code_blocks=code_blocks,
                            text_paragraphs=paragraphs
                        )
                        if train_res['loss'] is not None:
                            print(f"  [+] Training Loss: {train_res['loss']:.4f} | Weights Updated -> {train_res['checkpoint']}")

                    # Save / Append page to output file
                    if args.save:
                        mode = 'w' if page_count == 1 else 'a'
                        with open(args.save, mode, encoding='utf-8') as f:
                            f.write(f"\n{'='*70}\n")
                            f.write(f"PAGE #{page_count}: {title}\nURL: {current_url}\n")
                            f.write(f"Stats: {len(text):,} chars, {words:,} words, {len(code_blocks)} code blocks\n")
                            f.write(f"{'-'*70}\n\n")
                            f.write(text)
                            f.write("\n\n--- CODE BLOCKS ---\n\n")
                            for cb in code_blocks:
                                f.write(f"### {cb['context']}\n{cb['code']}\n\n")
                        print(f"  [+] Content appended to: {args.save}")

                    if args.full:
                        print("\n" + text.encode('ascii', errors='replace').decode('ascii'))

                    if args.screenshot:
                        ss_path = f"page_{page_count}_{args.screenshot}" if page_count > 1 else args.screenshot
                        bot.screenshot(ss_path)
                        print(f"  [+] Screenshot saved to: {ss_path}")

                    # Single page check
                    if not args.next:
                        print("\n[i] Note: Run with --next (or --toc) to continuously auto-read subsequent pages until stopped.")
                        break

                    # Max pages limit check
                    if args.max_pages > 0 and page_count >= args.max_pages:
                        print(f"\n[+] Reached requested max page limit ({args.max_pages}). Done!")
                        break

                    # Continuous mode: Click Next button at bottom of page!
                    print(f"\n[*] Reached end of Page #{page_count}. Looking for 'Next' button...")
                    time.sleep(args.delay)

                    has_next = bot.click_next_page()
                    if not has_next:
                        print(f"\n[!] No 'Next' button found on page (end of documentation reached).")
                        break

                    current_url = bot.page.url
                    print(f"  [>>>] 'Next' clicked successfully! Transitioning to: {current_url}")

        except KeyboardInterrupt:
            print("\n\n" + "!" * 70)
            print("  [STOPPED] Continuous reading loop paused by user (Ctrl+C)!")
            print("!" * 70)

    print("\n" + "=" * 70)
    print("  AUTONOMOUS CRAWL & READ SUMMARY:")
    print(f"  - Total Pages Read   : {page_count}")
    print(f"  - Total Words Read   : {total_words:,}")
    print(f"  - Total Characters   : {total_characters:,}")
    print(f"  - Total Code Blocks  : {total_code_blocks:,}")
    if args.save:
        print(f"  - Complete Corpus In : {args.save}")
    print("=" * 70)

if __name__ == "__main__":
    main()
