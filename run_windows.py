from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import numpy
import soundfile
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseSettings
from yukarin_autoreg.config import create_from_json as create_config
from yukarin_autoreg.generator import Generator, SamplingPolicy


class Settings(BaseSettings):
    model_config: str
    model: str

    class Config:
        env_file = ".env"


settings = Settings()

config = create_config(Path(settings.model_config))
model = Generator.load_model(
    model_config=config.model,
    model_path=Path(settings.model),
    gpu=0,
)
generator = Generator(config=config, model=model, max_batch_size=3)

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
