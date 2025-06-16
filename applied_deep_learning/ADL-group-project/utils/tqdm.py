import sys, os
import time
from typing import Iterable, Optional, Any

# add this file to path so tqdm is called here
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

def tqdm(
    iterable: Iterable, 
    desc: Optional[str] = None, 
    total: Optional[int] = None, 
    leave: bool = True
) -> Iterable:
    """
    Mom: We have tqdm at home
    tqdm at home:
    
    Args:
        iterable: The iterable to wrap
        desc: Optional description to display
        total: Total number of items (calculated from iterable if not provided)
        leave: Whether to leave the progress bar after completion
        
    Returns:
        The original items from the iterable
    """
    if total is None:
        try:
            total = len(iterable)
        except (TypeError, AttributeError):
            total = None
    
    if total is None:
        # Can't show progress without total, just return original iterable
        yield from iterable
        return
    
    start_time = time.time()
    last_update_time = start_time
    update_interval = 0.2  # Update display every 0.2 seconds
    
    if desc:
        prefix = f"{desc}: "
    else:
        prefix = ""
    
    try:
        for i, item in enumerate(iterable):
            # Yield the item first to avoid delaying processing
            yield item
            
            # Update progress display
            current_time = time.time()
            if i == 0 or i == total - 1 or current_time - last_update_time >= update_interval:
                elapsed = current_time - start_time
                progress = (i + 1) / total
                percentage = progress * 100
                
                # Calculate iterations per second
                it_per_sec = (i + 1) / elapsed if elapsed > 0 else 0
                
                # Calculate time remaining
                if progress > 0:
                    remaining = elapsed / progress - elapsed
                else:
                    remaining = 0
                
                # Create progress bar (20 characters wide)
                bar_length = 20
                filled_length = int(bar_length * progress)
                bar = '█' * filled_length + '-' * (bar_length - filled_length)
                
                # Format time strings
                time_str = f"{elapsed:.1f}s elapsed, {remaining:.1f}s remaining"
                
                # Create and print status line
                status = f"\r{prefix}{percentage:5.1f}% |{bar}| {i+1}/{total} [{time_str}, {it_per_sec:.2f}it/s]"
                sys.stdout.write(status)
                sys.stdout.flush()
                
                last_update_time = current_time
        
        # Print newline on completion
        if leave:
            sys.stdout.write('\n')
            sys.stdout.flush()
    except KeyboardInterrupt:
        if leave:
            sys.stdout.write('\n')
            sys.stdout.flush()
        raise
    
    # Print newline if we're not leaving the progress bar
    if not leave:
        sys.stdout.write('\r' + ' ' * len(status) + '\r')
        sys.stdout.flush()