import os
import sys
import socket
import subprocess

def disable_windows_quickedit():
    """
    Disables QuickEdit mode in Windows Command Prompt to prevent mouse clicks
    from pausing/freezing the server execution (Windows 'Select' state).
    """
    if sys.platform == "win32":
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            h_stdin = kernel32.GetStdHandle(-10)  # STD_INPUT_HANDLE = -10
            mode = ctypes.c_ulong()
            if kernel32.GetConsoleMode(h_stdin, ctypes.byref(mode)):
                ENABLE_QUICK_EDIT_MODE = 0x0040
                ENABLE_EXTENDED_FLAGS = 0x0080
                new_mode = (mode.value & ~ENABLE_QUICK_EDIT_MODE) | ENABLE_EXTENDED_FLAGS
                kernel32.SetConsoleMode(h_stdin, new_mode)
        except Exception:
            pass

def is_port_in_use(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex((host, port)) == 0

def find_available_port(preferred_port: int = 8765, host: str = "127.0.0.1") -> int:
    if not is_port_in_use(preferred_port, host):
        return preferred_port
    # Give Windows TIME_WAIT a brief moment (1.5s) to release the port
    time.sleep(1.5)
    if not is_port_in_use(preferred_port, host):
        return preferred_port
    print(f"[!] Port {preferred_port} is busy. Probing alternative ports...")
    for alt_port in [8766, 8001, 9001, 5005]:
        if not is_port_in_use(alt_port, host):
            print(f"[+] Found available port: {alt_port}")
            return alt_port
    return preferred_port

def main():
    disable_windows_quickedit()
    print("=" * 60)
    print(" Starting AI Retouching Local Server (FastAPI + Simple-LaMa)")
    print("=" * 60)
    
    print(f"[*] Python executable: {sys.executable}")
    print(f"[*] Python version: {sys.version}")
    
    # Check dependencies
    required = ["fastapi", "uvicorn", "PIL", "torch", "simple_lama_inpainting", "cv2"]
    missing = []
    for pkg in required:
        try:
            if pkg == "PIL":
                __import__("PIL")
            elif pkg == "cv2":
                __import__("cv2")
            else:
                __import__(pkg)
        except ImportError:
            missing.append(pkg)
            
    if missing:
        print(f"[!] Missing required packages: {', '.join(missing)}")
        print("[*] Installing requirements...")
        req_path = os.path.join(os.path.dirname(__file__), "requirements.txt")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", req_path])
    
    port = find_available_port(8765)
    print(f"[*] Launching Uvicorn server on http://127.0.0.1:{port} ...")
    print(f"[+] Web Studio UI available at: http://127.0.0.1:{port}")
    print("[*] Photoshop Plugin & Web Studio ready.")
    print("=" * 60)
    
    import uvicorn
    uvicorn.run("server:app", host="127.0.0.1", port=port, reload=False)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n[ERROR] Server encountered an error: {e}")
        import traceback
        traceback.print_exc()
        input("\nPress Enter to exit...")
