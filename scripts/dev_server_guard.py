"""Inspect/stop only this repository's listeners, never broad process-name kills."""
import argparse
import os
from pathlib import Path
import shlex
import signal
import subprocess


def is_project_server(cwd, command, root, port, api_port=8000, web_port=3000):
    if not cwd or not command:return False
    root=Path(root).resolve();cwd=Path(cwd).resolve()
    try:words=shlex.split(command)
    except ValueError:return False
    if not words:return False
    if port==api_port:
        return cwd==root and 'backend.api.main:app' in words and any(Path(w).name=='uvicorn' for w in words)
    if port==web_port:
        return cwd==root/'frontend' and (Path(words[0]).name=='next-server' or
            any(Path(word).name=='next' and words[i+1:i+2]==['dev'] for i,word in enumerate(words)))
    return False


def read_process(pid):
    cwd_result=subprocess.run(['lsof','-a','-p',str(pid),'-d','cwd','-Fn'],capture_output=True,text=True)
    cwd=next((line[1:] for line in cwd_result.stdout.splitlines() if line.startswith('n')),None)
    command_result=subprocess.run(['ps','-p',str(pid),'-o','command='],capture_output=True,text=True)
    return cwd,command_result.stdout.strip() if command_result.returncode==0 else None


def stop_verified(pid,root,port,inspect_process=read_process,send_signal=os.kill,api_port=8000,web_port=3000):
    cwd,command=inspect_process(pid)
    if not is_project_server(cwd,command,root,port,api_port,web_port):
        raise RuntimeError(f'Port {port} belongs to another or unverifiable process; it was not stopped')
    send_signal(pid,signal.SIGTERM)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--stop',action='store_true')
    parser.add_argument('--api-port',type=int,default=8000);parser.add_argument('--web-port',type=int,default=3000)
    args=parser.parse_args()
    listeners=[];foreign=[]
    for port in [args.api_port,args.web_port]:
        result=subprocess.run(['lsof','-nP','-t',f'-iTCP:{port}','-sTCP:LISTEN'],capture_output=True,text=True)
        if result.returncode not in [0,1]:raise RuntimeError(f'Could not inspect port {port}; nothing was stopped')
        for value in sorted(set(result.stdout.split())):
            pid=int(value);cwd,command=read_process(pid)
            if not is_project_server(cwd,command,args.root,port,args.api_port,args.web_port):
                foreign.append(port);continue  # another program: never signalled; run.sh moves to a free port
            listeners.append((port,pid))
    # Inspect every occupied port before stopping anything, then recheck ownership.
    for port,pid in listeners:
        if args.stop:
            try:stop_verified(pid,args.root,port,api_port=args.api_port,web_port=args.web_port)
            except ProcessLookupError:continue
        print(f'{"Stopped" if args.stop else "Verified"} project listener on port {port}, PID {pid}',flush=True)
    for port in sorted(set(foreign)):print(f'Port {port} is used by another program; left untouched',flush=True)
    if not listeners and not foreign:print(f'Project ports {args.api_port} and {args.web_port} are free',flush=True)


if __name__=='__main__':main()
