import signal

import pytest

from scripts.dev_server_guard import is_project_server,stop_verified


def test_foreign_checkout_and_non_app_listener_are_never_signalled(tmp_path):
    root=tmp_path/'project';sent=[]
    for cwd,cmd in [(tmp_path/'other','uvicorn backend.api.main:app --port 8000'),
                    (root,'python -m http.server 8000')]:
        with pytest.raises(RuntimeError,match='not stopped'):
            stop_verified(123,root,8000,lambda ignored,pair=(cwd,cmd):pair,lambda *args:sent.append(args))
    assert sent==[]


def test_verified_api_gets_term_and_frontend_requires_its_own_directory(tmp_path):
    root=tmp_path/'project';sent=[]
    stop_verified(123,root,8000,lambda ignored:(root,'/project/.venv/bin/uvicorn backend.api.main:app --port 8000'),lambda *args:sent.append(args))
    assert sent==[(123,signal.SIGTERM)]
    assert is_project_server(root/'frontend','next-server (v16.1)',root,3000)
    assert not is_project_server(root,'next-server (v16.1)',root,3000)
    assert not is_project_server(root/'frontend','python -m http.server 3000',root,3000)
