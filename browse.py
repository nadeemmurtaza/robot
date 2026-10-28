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

        # 4. Reading Clean Text
        print("\n[3/5] Reading & extracting page text...")
        text = bot.read_text("body")
        print(f"  [+] Extracted {len(text):,} characters of clean rendered text.")
        safe_preview = text[:400].encode('ascii', errors='replace').decode('ascii')
        print(f"  [+] Content Preview:\n{'-'*50}\n{safe_preview}...\n{'-'*50}")

        # 5. Understanding: Code Blocks & Outline
        print("\n[4/5] Understanding page structure & code...")
        code_blocks = bot.read_code_blocks()
        print(f"  [+] Found {len(code_blocks)} code blocks on the page!")
        if code_blocks:
            first = code_blocks[0]
            ctx = first['context'].encode('ascii', errors='replace').decode('ascii')
            snippet = first['code'][:250].encode('ascii', errors='replace').decode('ascii')
            print(f"  [+] Sample Code Block 1 (Context: {ctx or 'General'}):")
            print(f"```python\n{snippet}\n```")

        understanding = bot.understand_page()
        if understanding['headings']:
            print(f"\n[5/5] Extracted Page Outline ({len(understanding['headings'])} sections):")
            for h in understanding['headings'][:6]:
                indent = "  " * h['level']
                h_text = h['text'].encode('ascii', errors='replace').decode('ascii')
                print(f"  {indent}* {h_text}")

        # Optional Screenshot
        if args.screenshot:
            bot.screenshot(args.screenshot)
            print(f"\n[+] Screenshot saved to: {args.screenshot}")

    print("\n" + "=" * 70)
    print("  BROWSE & READ COMPLETED SUCCESSFULLY!")
    print("=" * 70)

if __name__ == "__main__":
    main()
