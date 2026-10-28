"""
Execution & Syntax Verifier for Error-Free Code Self-Training.
Verifies syntax via AST parsing and semantic correctness via isolated unit-test execution.
"""
import ast
import traceback
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

@dataclass
class VerificationResult:
    is_valid_syntax: bool
    passes_unit_tests: bool
    score: float                      # 0.0: Syntax error, 0.5: Syntax valid, 1.0: All tests pass
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    tests_passed: int = 0
    total_tests: int = 0
    cleaned_code: str = ""


class CodeVerifier:
    """
    Validates Python code completions through static AST checks and execution tests.
    """
    
    @staticmethod
    def clean_and_extract_code(raw_text: str, func_name: Optional[str] = None) -> str:
        """
        Cleans generated code text, strips markdown fences if present,
        and ensures proper indentation.
        """
        text = raw_text.strip()
        # Remove markdown code fences if model output them
        if "```python" in text:
            parts = text.split("```python")
            if len(parts) > 1:
                text = parts[1].split("```")[0]
        elif "```" in text:
            parts = text.split("```")
            if len(parts) > 1:
                text = parts[1].split("```")[0]
                
        # Remove trailing null characters or unprintable control bytes
        text = "".join(c for c in text if c == '\n' or c == '\t' or (' ' <= c <= '~'))
        return text

    @classmethod
    def check_syntax(cls, code: str) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Validates whether code conforms to Python syntax via AST parser.
        """
        try:
            ast.parse(code)
            return True, None, None
        except SyntaxError as e:
            return False, "SyntaxError", f"Line {e.lineno}: {e.msg}"
        except Exception as e:
            return False, type(e).__name__, str(e)

    @classmethod
    def run_tests(
        cls,
        full_code: str,
        func_name: str,
        test_cases: List[Tuple[Any, Any]],
        timeout_seconds: float = 1.0
    ) -> VerificationResult:
        """
        Runs unit tests on the defined function.
        test_cases is a list of (args_tuple, expected_return_value).
        """
        # Step 1: Syntax Check
        is_syntax_valid, err_type, err_msg = cls.check_syntax(full_code)
        if not is_syntax_valid:
            return VerificationResult(
                is_valid_syntax=False,
                passes_unit_tests=False,
                score=0.0,
                error_type=err_type,
                error_message=err_msg,
                tests_passed=0,
                total_tests=len(test_cases),
                cleaned_code=full_code
            )

        # Step 2: Safe In-Memory Execution Sandbox
        # Provide safe builtins
        safe_builtins = {
            "abs": abs, "all": all, "any": any, "bool": bool, "dict": dict,
            "enumerate": enumerate, "filter": filter, "float": float, "int": int,
            "isinstance": isinstance, "len": len, "list": list, "map": map,
            "max": max, "min": min, "range": range, "reversed": reversed,
            "round": round, "set": set, "sorted": sorted, "str": str,
            "sum": sum, "tuple": tuple, "zip": zip, "None": None, "True": True, "False": False
        }
        scope = {"__builtins__": safe_builtins}
        
        try:
            # Execute definition
            exec(full_code, scope)
        except Exception as e:
            return VerificationResult(
                is_valid_syntax=True,
                passes_unit_tests=False,
                score=0.3,
                error_type=type(e).__name__,
                error_message=f"Execution error on definition: {e}",
                tests_passed=0,
                total_tests=len(test_cases),
                cleaned_code=full_code
            )

        # Verify target function was defined
        if func_name not in scope or not callable(scope[func_name]):
            return VerificationResult(
                is_valid_syntax=True,
                passes_unit_tests=False,
                score=0.4,
                error_type="MissingFunction",
                error_message=f"Function '{func_name}' was not defined",
                tests_passed=0,
                total_tests=len(test_cases),
                cleaned_code=full_code
            )

        fn = scope[func_name]
        passed = 0

        # Step 3: Run Each Test Case
        for args, expected in test_cases:
            try:
                # Support single arg or tuple of args
                if isinstance(args, tuple):
                    result = fn(*args)
                else:
                    result = fn(args)
                    
                if result == expected:
                    passed += 1
                else:
                    return VerificationResult(
                        is_valid_syntax=True,
                        passes_unit_tests=False,
                        score=0.5 + 0.4 * (passed / max(len(test_cases), 1)),
                        error_type="AssertionFailure",
                        error_message=f"Failed test with args {args}: expected {expected!r}, got {result!r}",
                        tests_passed=passed,
                        total_tests=len(test_cases),
                        cleaned_code=full_code
                    )
            except Exception as e:
                return VerificationResult(
                    is_valid_syntax=True,
                    passes_unit_tests=False,
                    score=0.4 + 0.3 * (passed / max(len(test_cases), 1)),
                    error_type=type(e).__name__,
                    error_message=f"Runtime error with args {args}: {e}",
                    tests_passed=passed,
                    total_tests=len(test_cases),
                    cleaned_code=full_code
                )

        # Step 4: 100% Passed All Unit Tests! Error-Free!
        return VerificationResult(
            is_valid_syntax=True,
            passes_unit_tests=(passed == len(test_cases)),
            score=1.0,
            error_type=None,
            error_message=None,
            tests_passed=passed,
            total_tests=len(test_cases),
            cleaned_code=full_code
        )
