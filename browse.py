"""
Browse & Read: Autonomous Web Interaction CLI for Code Robot.
Usage:
    python browse.py "https://your-link-here.com"
    python browse.py "https://your-link-here.com" --head
    python browse.py "https://your-link-here.com" --click "Next"
"""
import sys
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
    args = parser.parse_args()

    print("=" * 70)
    print("  CODE ROBOT: AUTONOMOUS BROWSER & READING AGENT")
    print("=" * 70)
    print(f"[*] Target URL : {args.url}")
    print(f"[*] Window Mode: {'Visible Chrome (Headed)' if args.head else 'Headless Chrome'}")

    with BrowserRobot(headless=not args.head) as bot:
        # 1. Navigation
        print("\n[1/5] Navigating to page...")
        title = bot.goto(args.url)
        safe_title = title.encode('ascii', errors='replace').decode('ascii')
        print(f"  [+] Page Title: {safe_title}")

        # 2. Scrolling
        print(f"\n[2/5] Scrolling down {args.scroll}px...")
        scroll_info = bot.scroll("down", args.scroll)
        print(f"  [+] Current Position: {scroll_info['scrollY']}px / Total Height: {scroll_info['scrollHeight']}px")

        # 3. Optional Clicking
        if args.click:
            print(f"\n[+] Clicking element: '{args.click}'...")
            clicked = bot.click(args.click)
            print(f"  [+] Click result: {'SUCCESS' if clicked else 'Element not found'}")
            new_title = bot.page.title().encode('ascii', errors='replace').decode('ascii')
            print(f"  [+] New Page Title: {new_title}")

        # 4. Reading the ENTIRE Page Text
        print("\n[3/5] Reading the ENTIRE text on page...")
        text = bot.read_text()
        paragraphs = bot.read_paragraphs()
        words = len(text.split())
        print(f"  [+] Extracted FULL page text:")
        print(f"      - Characters : {len(text):,} characters")
        print(f"      - Word Count : {words:,} words")
        print(f"      - Paragraphs : {len(paragraphs)} full paragraphs extracted")

        if paragraphs:
            print(f"\n  [+] Sample First Paragraph Read:")
            p0 = paragraphs[0].encode('ascii', errors='replace').decode('ascii')
            print(f"      \"{p0[:300]}...\"")

        # 5. Understanding: Code Blocks & Outline
        print("\n[4/5] Understanding code blocks & section outline...")
        code_blocks = bot.read_code_blocks()
        print(f"  [+] Found {len(code_blocks)} code blocks across the entire page!")
        if code_blocks:
            first = code_blocks[0]
            ctx = first['context'].encode('ascii', errors='replace').decode('ascii')
            snippet = first['code'][:250].encode('ascii', errors='replace').decode('ascii')
            print(f"  [+] Sample Code Block 1 (Context: {ctx or 'General'}):")
            print(f"```python\n{snippet}\n```")

        understanding = bot.understand_page()
        if understanding['headings']:
            print(f"\n[5/5] Extracted Page Outline ({len(understanding['headings'])} sections):")
            for h in understanding['headings'][:8]:
                indent = "  " * h['level']
                h_text = h['text'].encode('ascii', errors='replace').decode('ascii')
                print(f"  {indent}* {h_text}")

        # Save the full extracted content to file
        if args.save:
            with open(args.save, 'w', encoding='utf-8') as f:
                f.write(f"URL: {args.url}\n")
                f.write(f"Title: {title}\n")
                f.write(f"Extracted: {len(text):,} characters, {words:,} words, {len(paragraphs)} paragraphs, {len(code_blocks)} code blocks\n")
                f.write("=" * 70 + "\n\n")
                f.write("--- FULL EXTRACTED TEXT ---\n\n")
                f.write(text)
                f.write("\n\n" + "=" * 70 + "\n\n")
                f.write(f"--- ALL {len(code_blocks)} CODE BLOCKS ---\n\n")
                for cb in code_blocks:
                    f.write(f"### Block {cb['id']}: {cb['context']}\n")
                    f.write(cb['code'] + "\n\n")
            print(f"\n[+] Full text and all {len(code_blocks)} code blocks SAVED to: {args.save}")

        # Print full text if requested
        if args.full:
            print("\n" + "=" * 70)
            print("  FULL EXTRACTED PAGE TEXT:")
            print("=" * 70)
            print(text.encode('ascii', errors='replace').decode('ascii'))

        # Optional Screenshot
        if args.screenshot:
            bot.screenshot(args.screenshot)
            print(f"\n[+] Screenshot saved to: {args.screenshot}")

    print("\n" + "=" * 70)
    print("  BROWSE & READ COMPLETED SUCCESSFULLY!")
    print("=" * 70)

if __name__ == "__main__":
    main()
