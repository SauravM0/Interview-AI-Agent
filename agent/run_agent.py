"""
Wrapper script to run the agent with Python 3.14 compatibility.
This ensures the event loop is properly initialized in spawned subprocesses.
"""
import asyncio
import multiprocessing
import os
import sys

def init_worker():
    """Initialize event loop in spawned worker process."""
    try:
        asyncio.get_event_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())

if __name__ == "__main__":
    # Set multiprocessing context with initializer
    if os.name == 'nt':  # Windows
        try:
            multiprocessing.set_start_method('spawn', force=True)
        except RuntimeError:
            pass
    
    # Ensure main process has event loop
    try:
        asyncio.get_event_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())
    
    # Import and run the main agent
    import main

# Made with Bob
