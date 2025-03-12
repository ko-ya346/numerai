import ipykernel
import json
from pathlib import Path
from notebook.notebookapp import list_running_servers


def get_notebook_name():
    """
    現在開いているjupyter notebook のファイル名を取得
    """
    kernel_id = ipykernel.connect.get_connection_file().split("-")[1].split(".")[0]
    for server in list_running_servers():
        try:
            sessions = json.load(Path(server["rooot_dir"]).joinpath("api/sessions").open())
            for session in sessions:
                if session["kernel"]["id"] == kernel_id:
                    return session["name"]
        except:
            pass
    return None
