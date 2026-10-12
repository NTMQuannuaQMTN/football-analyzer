"""Inspect/stop only this repository's listeners, never broad process-name kills."""
import argparse
import os
from pathlib import Path
import shlex
import signal
import subprocess


def is_project_server(cwd, command, root, port):
    if not cwd or not command:return False
    root=Path(root).resolve();cwd=Path(cwd).resolve()
    try:words=shlex.split(command)
    except ValueError:return False
    if not words:return False
    if port==8000:
        return cwd==root and 'backend.api.main:app' in words and any(Path(w).name=='uvicorn' for w in words)
    if port==3000:
        return cwd==root/'frontend' and (Path(words[0]).name=='next-server' or
            any(Path(word).name=='next' and words[i+1:i+2]==['dev'] for i,word in enumerate(words)))
    return False


def read_process(pid):
    cwd_result=subprocess.run(['lsof','-a','-p',str(pid),'-d','cwd','-Fn'],capture_output=True,text=True)
    cwd=next((line[1:] for line in cwd_result.stdout.splitlines() if line.startswith('n')),None)
    command_result=subprocess.run(['ps','-p',str(pid),'-o','command='],capture_output=True,text=True)
    return cwd,command_result.stdout.strip() if command_result.returncode==0 else None


def stop_verified(pid,root,port,inspect_process=read_process,send_signal=os.kill):
    cwd,command=inspect_process(pid)
    if not is_project_server(cwd,command,root,port):
        raise RuntimeError(f'Port {port} belongs to another or unverifiable process; it was not stopped')
    send_signal(pid,signal.SIGTERM)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--stop',action='store_true');args=parser.parse_args()
    listeners=[]
    for port in [8000,3000]:
        result=subprocess.run(['lsof','-nP','-t',f'-iTCP:{port}','-sTCP:LISTEN'],capture_output=True,text=True)
        if result.returncode not in [0,1]:raise RuntimeError(f'Could not inspect port {port}; nothing was stopped')
        for value in sorted(set(result.stdout.split())):
            pid=int(value);cwd,command=read_process(pid)
            if not is_project_server(cwd,command,args.root,port):
                raise RuntimeError(f'Port {port} belongs to another or unverifiable process; nothing was stopped')
            listeners.append((port,pid))
    # Inspect every occupied port before stopping anything, then recheck ownership.
    for port,pid in listeners:
        if args.stop:
            try:stop_verified(pid,args.root,port)
            except ProcessLookupError:continue
        print(f'{"Stopped" if args.stop else "Verified"} project listener on port {port}, PID {pid}',flush=True)
    if not listeners:print('Project ports 8000 and 3000 are free',flush=True)


if __name__=='__main__':main()
