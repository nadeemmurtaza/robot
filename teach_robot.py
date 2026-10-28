"""
Interactive CLI for Teaching Python to the Next-Concept Coding Model.

Usage:
  python teach_robot.py --chapter 1      # Teach Chapter 1: Arithmetic
  python teach_robot.py --chapter 2      # Teach Chapter 2: Logic & Conditions
  python teach_robot.py --all            # Teach the full Python curriculum
  python teach_robot.py --interactive    # Interactive Teacher Mode (teach your own functions)
"""
import sys
import argparse
import torch

torch.set_num_threads(8)

from custom_ncm.curriculum import get_curriculum, get_chapters
from custom_ncm.teacher import PythonTeacher

def main():
    parser = argparse.ArgumentParser(description="Teach Python to the Next-Concept Model")
    parser.add_argument("--chapter", type=int, choices=list(range(1, 11)), default=None, help="Teach specific chapter (1-10)")
    parser.add_argument("--all", action="store_true", help="Teach all curriculum chapters")
    parser.add_argument("--list", action="store_true", help="List all chapters and lessons")
    parser.add_argument("--interactive", action="store_true", help="Interactive teaching console")
    args = parser.parse_args()
    
    chapters = get_chapters()
    chapter_names = list(chapters.keys())
    
    if args.list:
        print("\n" + "="*70)
        print("  PYTHON MASTER CURRICULUM SYLLABUS (10 CHAPTERS)")
        print("="*70)
        for idx, (ch_name, lessons) in enumerate(chapters.items(), start=1):
            print(f"\n[Chapter {idx}] {ch_name} ({len(lessons)} lessons):")
            for l in lessons:
                print(f"   - {l.topic:35s} -> def {l.func_name}(...)")
        print("="*70 + "\n")
        return
        
    print("[*] Initializing Python Teacher Engine...", flush=True)
    teacher = PythonTeacher()
    
    if args.interactive:
        print("\n" + "="*70)
        print("  INTERACTIVE PYTHON TEACHING CONSOLE")
        print("  Teach the Next-Concept model any custom function!")
        print("="*70)
        print("Type 'exit' to quit.\n")
        
        while True:
            try:
                func_name = input("\n1. Function Name (e.g. 'cube'): ").strip()
                if not func_name or func_name == "exit":
                    break
                    
                prompt = input("2. Prompt Header (e.g. 'def cube(x):'): ").strip()
                target_code = input("3. Target Body (e.g. '    return x ** 3'): ").strip()
                
                print("\n[Optional] Add a test case (or press Enter to skip):")
                test_in = input("   Input argument (e.g. 3): ").strip()
                test_out = input("   Expected output (e.g. 27): ").strip()
                
                test_cases = []
                if test_in and test_out:
                    try:
                        val_in = eval(test_in)
                        val_out = eval(test_out)
                        test_cases.append(((val_in,), val_out))
                    except Exception as e:
                        print(f"   [!] Error parsing test case: {e}")
                        
                teacher.teach_custom(
                    prompt=prompt if prompt.endswith("\n") else prompt + "\n",
                    target_body=target_code if target_code.endswith("\n") else target_code + "\n",
                    func_name=func_name,
                    test_cases=test_cases,
                    steps=5
                )
            except KeyboardInterrupt:
                break
                
    elif args.chapter is not None:
        idx = args.chapter - 1
        ch_name = chapter_names[idx]
        teacher.teach_chapter(ch_name)
        
    elif args.all:
        for ch_name in chapter_names:
            teacher.teach_chapter(ch_name)
            
    else:
        # Default: teach Chapter 1 (Arithmetic & Math)
        print("\n[!] No option specified. Starting with Chapter 1: Arithmetic & Math!")
        print("    (Tip: run 'python teach_robot.py --interactive' to teach custom functions)\n")
        teacher.teach_chapter(chapter_names[0])

if __name__ == "__main__":
    main()
