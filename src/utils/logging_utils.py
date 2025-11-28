import os
import sys
import logging

class Tee(object):
    def __init__(self, *files):
        self.files = files
    
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
    
    def flush(self):
        for f in self.files:
            f.flush()
    
    def isatty(self):
        """Check if any of the underlying files is a terminal."""
        return any(hasattr(f, 'isatty') and f.isatty() for f in self.files)
    
    def fileno(self):
        """Return file descriptor of the first file that has one."""
        for f in self.files:
            if hasattr(f, 'fileno'):
                try:
                    return f.fileno()
                except (AttributeError, OSError):
                    continue
        raise OSError("No file descriptor available")
    
    def readable(self):
        """Check if readable (typically False for stdout/stderr)."""
        return False
    
    def writable(self):
        """Check if writable."""
        return True
    
    def seekable(self):
        """Check if seekable (typically False for stdout/stderr)."""
        return False
    
    def close(self):
        """Close all underlying files."""
        for f in self.files:
            if hasattr(f, 'close'):
                f.close()
    
    def __getattr__(self, name):
        """Delegate any other attributes to the first file."""
        if self.files:
            return getattr(self.files[0], name)
        raise AttributeError(f"'Tee' object has no attribute '{name}'")

def setup_terminal_logging(log_dir, process_idx=0, disable_for_compile=False):
    """
    Setup terminal logging with optional disable for torch.compile compatibility.
    
    Args:
        log_dir: Directory to save log files
        process_idx: Process index for multi-process training
        disable_for_compile: If True, skip stdout/stderr redirection (useful for torch.compile)
    """
    if disable_for_compile:
        # Just create the log directory but don't redirect stdout/stderr
        os.makedirs(log_dir, exist_ok=True)
        print(f"Terminal logging disabled for torch.compile compatibility. Logs saved to {log_dir}")
        return
    
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, f"terminal_output_{process_idx}.txt")
    log_file = open(log_file_path, "a")
    sys.stdout = Tee(sys.__stdout__, log_file)
    sys.stderr = Tee(sys.__stderr__, log_file)

def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='[\033[34m%(asctime)s\033[0m] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
    )
    logger = logging.getLogger(__name__)
    return logger