"""
Utility functions for ag-viz package.

Provides helper functions for:
- Locating the ag CLI executable
- Running ag CLI commands with error handling
- Managing output directories
- Formatting model specifications
"""

import os
import shutil
import subprocess
from pathlib import Path
from typing import List, Optional, Dict, Any


def find_ag_executable() -> Optional[Path]:
    """
    Locate the ag CLI tool executable.
    
    Searches in the following order:
    1. AG_EXECUTABLE environment variable
    2. ag in system PATH
    3. Common build locations relative to this package
    
    Returns
    -------
    Optional[Path]
        Path to the ag executable if found, None otherwise.
    
    Examples
    --------
    >>> ag_path = find_ag_executable()
    >>> if ag_path:
    ...     print(f"Found ag at: {ag_path}")
    """
    # Check environment variable first
    env_path = os.environ.get("AG_EXECUTABLE")
    if env_path and Path(env_path).exists():
        return Path(env_path)
    
    # Check system PATH
    ag_path = shutil.which("ag")
    if ag_path:
        return Path(ag_path)
    
    # Check common build locations relative to package
    package_dir = Path(__file__).parent.parent.parent
    common_locations = [
        package_dir / "build" / "src" / "ag",
        package_dir / "build" / "Release" / "src" / "ag",
        package_dir / "build" / "Debug" / "src" / "ag",
    ]
    
    for location in common_locations:
        if location.exists():
            return location
    
    return None


def run_ag_command(args: List[str], check: bool = True) -> subprocess.CompletedProcess:
    """
    Execute an ag CLI command with error handling.
    
    Parameters
    ----------
    args : List[str]
        Command-line arguments for the ag executable (without the 'ag' prefix).
    check : bool, optional
        If True, raise CalledProcessError on non-zero exit code (default: True).
    
    Returns
    -------
    subprocess.CompletedProcess
        The completed process with stdout, stderr, and return code.
    
    Raises
    ------
    RuntimeError
        If the ag executable cannot be found.
    subprocess.CalledProcessError
        If the command fails and check=True.
    
    Examples
    --------
    >>> result = run_ag_command(['fit', '-d', 'data.csv', '-a', '1,0,1', '-g', '1,1'])
    >>> print(result.stdout)
    """
    ag_exec = find_ag_executable()
    if ag_exec is None:
        raise RuntimeError(
            "ag executable not found. Please ensure it is built or set "
            "the AG_EXECUTABLE environment variable to its location."
        )
    
    cmd = [str(ag_exec)] + args
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=check,
        )
        return result
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"ag command failed with exit code {e.returncode}\n"
            f"Command: {' '.join(cmd)}\n"
            f"stderr: {e.stderr}"
        ) from e


def ensure_output_dir(path: Path) -> Path:
    """
    Create output directory if it doesn't exist.
    
    Parameters
    ----------
    path : Path
        Path to the directory to create.
    
    Returns
    -------
    Path
        The created or existing directory path.
    
    Examples
    --------
    >>> output_dir = ensure_output_dir(Path('./output'))
    >>> print(output_dir.exists())
    True
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def format_model_spec(model_json: Dict[str, Any]) -> str:
    """
    Format model specification for display.
    
    Parameters
    ----------
    model_json : Dict[str, Any]
        Model JSON data containing specification and parameters.
    
    Returns
    -------
    str
        Formatted string representation of the model specification.
    
    Examples
    --------
    >>> model = {"spec": {"arima": {"p": 1, "d": 0, "q": 1}, "garch": {"p": 1, "q": 1}}}
    >>> print(format_model_spec(model))
    ARIMA(1,0,1)-GARCH(1,1)
    """
    try:
        spec = model_json.get("spec", {})
        arima = spec.get("arima", {})
        garch = spec.get("garch", {})
        
        arima_str = f"ARIMA({arima.get('p', 0)},{arima.get('d', 0)},{arima.get('q', 0)})"
        garch_str = f"GARCH({garch.get('p', 0)},{garch.get('q', 0)})"
        
        return f"{arima_str}-{garch_str}"
    except Exception:
        return "Unknown Model"
