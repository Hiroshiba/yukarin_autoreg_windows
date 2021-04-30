import time
from datetime import datetime, timedelta
from io import BytesIO
from pathlib import Path
from threading import Lock
from typing import Dict
from zipfile import ZipFile

import requests
from watchdog.events import PatternMatchingEventHandler
from watchdog.observers import Observer

linux_url = "http://localhost:8000/"
windows_url = "http://localhost:8001/"

input_dir = Path("hiho_input")
input_dir.mkdir(exist_ok=True)

output_dir = Path("hiho_output")
output_dir.mkdir(exist_ok=True)

lock = Lock()


class FileChangeHandler(PatternMatchingEventHandler):
    def __init__(self, obj: Dict[Path, datetime], *args, **kwargs):
        super().__init__()
        self.obj = obj

    def on_modified(self, event):
        if event.is_directory:
            return

        with lock:
            self.obj[Path(event.src_path)] = datetime.now()


def convert(input_path: Path):
    print(f"start convert '{input_path.stem}'")
    input_text = input_path.stem

    wave = input_path.open("rb").read()
    r = requests.post(
        linux_url,
        data=dict(text=input_text),
        files=dict(wave=BytesIO(wave)),
    )
    assert r.status_code == 200
    feature = r.content
    output_dir.joinpath(f"{input_text}.binary").write_bytes(feature)

    r = requests.post(
        windows_url,
        data=dict(num=3),
        files=dict(feature=BytesIO(feature)),
    )
    assert r.status_code == 200
    output = r.content

    with ZipFile(BytesIO(output), mode="r") as z:
        for name in z.namelist():
            with z.open(name, mode="r") as f:
                output_dir.joinpath(f"{input_text}-{name}.wav").write_bytes(f.read())

    print(f"finish convert '{input_path.stem}'")


file_object: Dict[Path, datetime] = {}
event_handler = FileChangeHandler(file_object)

observer = Observer()
observer.schedule(event_handler, path=str(input_dir))
observer.start()

print("Ready...")
while True:
    time.sleep(5)

    with lock:
        delete_paths = []
        for p, t in file_object.items():
            if datetime.now() - t > timedelta(seconds=5):
                if p.exists():
                    convert(input_path=p)
                delete_paths.append(p)

        for p in delete_paths:
            file_object.pop(p)
