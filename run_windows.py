from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import numpy
import soundfile
import yaml
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseSettings
from yukarin_wavernn.config import Config
from yukarin_wavernn.generator import Generator, SamplingPolicy


class Settings(BaseSettings):
    model_dir: str

    class Config:
        env_file = ".env"


settings = Settings()


config = Config.from_dict(
    yaml.safe_load((Path(settings.model_dir) / "config.yaml").open())
)
generator = Generator(
    config=config,
    predictor=list(Path(settings.model_dir).glob("*.pth"))[0],
    use_gpu=True,
    max_batch_size=3,
    use_fast_inference=True,
)

app = FastAPI()


@app.post("/")
async def to_wave(num: int = Form(...), feature: UploadFile = File(...)):
    assert num <= 3

    with TemporaryDirectory() as d:
        tmp_dir = Path(d)

        array = numpy.frombuffer(await feature.read(), dtype=numpy.float32).reshape(
            1, -1, 40
        )
        local_array = numpy.repeat(array, num, axis=0)

        waves = generator.generate(
            time_length=local_array.shape[1] / 200,
            sampling_policy=SamplingPolicy.random,
            num_generate=num,
            local_array=local_array,
        )

        f = BytesIO()
        with ZipFile(f, "w") as z:
            for i, wave in enumerate(waves):
                tmp_path = tmp_dir.joinpath(f"output_{i}.wav")
                soundfile.write(
                    file=str(tmp_path),
                    data=wave.wave,
                    samplerate=wave.sampling_rate,
                    format="WAV",
                )
                z.write(str(tmp_path), arcname=str(i))

    return StreamingResponse(BytesIO(f.getvalue()))
