"""Launch both local services; stop both with Ctrl+C. No retraining at startup."""
import subprocess,sys,time,webbrowser,socket
from pathlib import Path
import httpx
ROOT=Path(__file__).resolve().parents[1]

def main():
    for port in [8000,8501]:
        with socket.socket() as sock:
            if sock.connect_ex(('127.0.0.1',port))==0:
                raise SystemExit(f'Port {port} already occupied. Stop the prior app before launching.')
    processes=[]
    try:
        processes.append(subprocess.Popen([sys.executable,'-m','uvicorn','backend.main:app','--host','127.0.0.1','--port','8000'],cwd=ROOT))
        for _ in range(90):
            try:
                if httpx.get('http://127.0.0.1:8000/health',timeout=2).status_code==200:break
            except httpx.HTTPError:pass
            if processes[0].poll() is not None:raise RuntimeError('Backend exited; read the error above')
            time.sleep(1)
        else:raise RuntimeError('Backend did not become healthy')
        processes.append(subprocess.Popen([sys.executable,'-m','streamlit','run','frontend/app.py','--server.port','8501'],cwd=ROOT))
        print('FraudSense: http://127.0.0.1:8501 | API: http://127.0.0.1:8000/docs | Ctrl+C stops both',flush=True)
        if '--no-browser' not in sys.argv:webbrowser.open('http://127.0.0.1:8501')
        while all(p.poll() is None for p in processes):time.sleep(1)
    except KeyboardInterrupt:pass
    finally:
        for p in processes:
            if p.poll() is None:p.terminate()
        for p in processes:
            try:p.wait(timeout=10)
            except subprocess.TimeoutExpired:p.kill()

if __name__=='__main__':main()
