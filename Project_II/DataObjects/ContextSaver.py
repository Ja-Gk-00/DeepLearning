from io import StringIO
import sys
from pathlib import Path

class Capturing:
    """
    Context manager to capture stdout output and save it to a log file.

    Usage:
        with Capturing(log_path) as output_lines:
            # code that prints
        # printed lines saved to log_path, and output_lines is a list of lines.
    """
    def __init__(self, log_path: str):
        self.log_path = log_path
        self._stringio = None
        self._stdout = None
        self.output_lines: list[str] = []

    def __enter__(self):
        self._stdout = sys.stdout
        self._stringio = StringIO()
        sys.stdout = self._stringio
        return self.output_lines

    def __exit__(self, exc_type, exc_value, traceback):
        # Retrieve printed text
        text = self._stringio.getvalue()
        self.output_lines.extend(text.splitlines())
        # Write to log file
        file = Path(self.log_path)
        file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_path, 'a+') as f:
            f.write(text)
        sys.stdout = self._stdout
        del self._stringio