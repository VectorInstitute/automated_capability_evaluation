import sys
import io
import contextlib
import traceback
import signal

def python_calculator(code: str, timeout: int = 5) -> str:
    """
    Executes Python code in a controlled environment and returns stdout + stderr.
    
    Args:
        code (str): The Python code to execute.
        timeout (int): Timeout in seconds.

    Returns:
        str: The standard output and standard error captured during execution.
    """
    
    # Create a string buffer to capture stdout and stderr
    s = io.StringIO()
    
    # Function to handle timeout
    def handler(signum, frame):
        raise TimeoutError("Execution timed out")
    
    # Register the signal function handler
    # Note: signal.alarm only works on Unix-based systems. 
    # For cross-platform safety in a real app, we'd use multiprocessing or threading.
    # Since we are on Darwin (macOS), this is fine.
    try:
        signal.signal(signal.SIGALRM, handler)
        signal.alarm(timeout)
    except AttributeError:
        # Windows doesn't support SIGALRM, skip timeout enforcement for now if running there
        pass

    try:
        with contextlib.redirect_stdout(s), contextlib.redirect_stderr(s):
            # Define a restricted global environment if needed, but for now we trust the internal agent
            # We add some math helpers by default
            exec_globals = {
                "math": __import__("math"),
                "print": print,
                "range": range,
                "len": len,
                "int": int,
                "float": float,
                "str": str,
                "list": list,
                "dict": dict,
                "set": set,
                "tuple": tuple,
                "abs": abs,
                "round": round,
                "sum": sum,
                "min": min,
                "max": max,
                "sorted": sorted,
                "enumerate": enumerate,
                "zip": zip,
            }
            exec(code, exec_globals)
    except TimeoutError:
        print("Error: Execution timed out.", file=s)
    except Exception:
        print(traceback.format_exc(), file=s)
    finally:
        # Disable alarm
        try:
            signal.alarm(0)
        except AttributeError:
            pass

    return s.getvalue()
