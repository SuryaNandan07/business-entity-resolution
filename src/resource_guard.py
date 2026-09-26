"""Windows resource measurements and conservative process watchdog."""
import ctypes
import json
import os
from pathlib import Path
import threading
import time

PROCESS_RAM_LIMIT = int(5.5 * 2**30)     # 5.5 GiB
SYSTEM_AVAIL_LIMIT = int(512 * 2**20)    # 512 MiB
CONSECUTIVE_LOW_THRESHOLD = 5

class ProcessMemory(ctypes.Structure):
    _fields_ = [('cb',ctypes.c_ulong),('PageFaultCount',ctypes.c_ulong)] + [(n,ctypes.c_size_t) for n in ('PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage','QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage','PrivateUsage')]

class SystemMemory(ctypes.Structure):
    _fields_ = [('length',ctypes.c_ulong),('load',ctypes.c_ulong)] + [(n,ctypes.c_ulonglong) for n in ('total','available','total_page','available_page','total_virtual','available_virtual','extended')]

def memory():
    p=ProcessMemory(); p.cb=ctypes.sizeof(p)
    handle=ctypes.windll.kernel32.GetCurrentProcess
    handle.restype=ctypes.c_void_p
    get=ctypes.windll.psapi.GetProcessMemoryInfo
    get.argtypes=[ctypes.c_void_p,ctypes.POINTER(ProcessMemory),ctypes.c_ulong]
    if not get(handle(),ctypes.byref(p),p.cb): raise ctypes.WinError()
    s=SystemMemory(); s.length=ctypes.sizeof(s)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(s)): raise ctypes.WinError()
    return dict(rss=p.WorkingSetSize,private=p.PrivateUsage,peak_rss=p.PeakWorkingSetSize,available=s.available)

_state = threading.local()

def reset_guard_state():
    _state.low_avail_count = 0

def check(consecutive_limit=CONSECUTIVE_LOW_THRESHOLD):
    m=memory()
    if max(m['rss'],m['private']) >= PROCESS_RAM_LIMIT:
        raise MemoryError(f"Process RAM limit reached (>= 5.5 GiB): {m}")
    
    low_count = getattr(_state, 'low_avail_count', 0)
    if m['available'] < SYSTEM_AVAIL_LIMIT:
        low_count += 1
    else:
        low_count = 0
    _state.low_avail_count = low_count
    
    if low_count >= consecutive_limit:
        raise MemoryError(f"Sustained critically low system memory (< 512 MiB for {low_count} observations): {m}")
    return m

def watchdog(path, consecutive_limit=CONSECUTIVE_LOW_THRESHOLD, interval=0.2):
    """Stop before 6 GiB or on sustained critically low system available memory."""
    def run():
        while True:
            try: check(consecutive_limit=consecutive_limit)
            except MemoryError as e:
                Path(path).write_text(json.dumps({'stopped':str(e)}))
                os._exit(86)
            time.sleep(interval)
    threading.Thread(target=run,daemon=True).start()
