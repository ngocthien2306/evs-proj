import multiprocessing
import sys
import time


def restart_usb_devices():
    try:
        import subprocess
        print("Starting Setup ...")
        subprocess.run(['pnputil', '/restart-device', 'USB\\VID*'], 
                      capture_output=True, 
                      text=True, 
                      creationflags=subprocess.CREATE_NO_WINDOW)
        time.sleep(2)  
        return True
    except Exception as e:
        print(f"Failed to restart USB: {e}")
        return False

def signal_handler(signum, frame):
    print(f"\nReceived signal {signum}")
    try:
        current_process = multiprocessing.current_process()
        for child in current_process.children():
            child.terminate()
        
        for child in current_process.children():
            child.join(timeout=3)
    except Exception as e:
        print(f"Error in signal handler: {e}")
    
    sys.exit(0)

def cleanup_old_processes():
    import psutil
    current_process = psutil.Process()
    process_name = current_process.name()
    
    for proc in psutil.process_iter(['pid', 'name']):
        try:
            if proc.info['name'] == process_name and proc.pid != current_process.pid:
                psutil.Process(proc.pid).terminate()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    