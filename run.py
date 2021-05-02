import subprocess
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy
from acoustic_feature_extractor.data.f0 import F0, F0Type
from acoustic_feature_extractor.data.linguistic_feature import LinguisticFeature
from acoustic_feature_extractor.data.phoneme import JvsPhoneme
from acoustic_feature_extractor.data.sampling_data import SamplingData
from acoustic_feature_extractor.data.wave import Wave
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import StreamingResponse
from julius4seg import sp_inserter
from openjtalk_label_getter import OutputType, openjtalk_label_getter
from pydantic import BaseSettings


class Settings(BaseSettings):
    model_dir: str

    class Config:
        env_file = ".env"


settings = Settings()

_hmm_model = "/github/segmentation-kit/models/hmmdefs_monof_mix16_gid.binhmm"
_jvs_to_julius = {
    "I": "i",
    "U": "u",
    "E": "e",
    "cl": "q",
    "pau": "sp",
}
_feature_rate = 200
_voiro_mean = numpy.load(
    list(Path(settings.model_dir).glob("*.npy"))[0], allow_pickle=True
).item()["mean"]
_hiho_mean = 4.84241337

app = FastAPI()


@app.post("/")
async def to_feature(text: str = Form(...), wave: UploadFile = File(...)):
    with TemporaryDirectory() as d:
        tmp_dir = Path(d)
        input_audio_path = tmp_dir.joinpath("input.wav")
        input_audio_path.write_bytes(await wave.read())

        # openjtalk
        phonemes = [
            p.label
            for p in openjtalk_label_getter(
                text,
                openjtalk_command="open_jtalk",
                dict_path=Path("/var/lib/mecab/dic/open-jtalk/naist-jdic"),
                htsvoice_path=Path(
                    "/usr/share/hts-voice/nitech-jp-atr503-m001/nitech_jp_atr503_m001.htsvoice"
                ),
                output_wave_path=tmp_dir.joinpath("wave.wav"),
                output_log_path=tmp_dir.joinpath("log.txt"),
                output_type=OutputType.phoneme,
                without_span=False,
            )
        ]

        # julius
        julius_audio_path = tmp_dir.joinpath("julius.wav")
        subprocess.check_call(
            f"sox {input_audio_path} -r 16000 -b 16 {julius_audio_path}".split()
        )

        julius_phonemes = [
            p if p not in _jvs_to_julius else _jvs_to_julius[p]
            for p in phonemes
            if p != "sil"
        ]

        julius_dict_path = tmp_dir.joinpath("2nd.dict")
        julius_dict = sp_inserter.gen_julius_dict_2nd(
            " ".join(julius_phonemes), model_type=sp_inserter.ModelType.gmm
        )
        julius_dict_path.write_text(julius_dict)

        julius_dfa_path = tmp_dir.joinpath("2nd.dfa")
        julius_dfa = sp_inserter.gen_julius_aliment_dfa(julius_dict.count("\n"))
        julius_dfa_path.write_text(julius_dfa)

        julius_output = sp_inserter.julius_phone_alignment(
            str(julius_audio_path),
            str(tmp_dir.joinpath("2nd")),
            _hmm_model,
            model_type=sp_inserter.ModelType.gmm,
            options=None,
        )

        time_alignment_list = sp_inserter.frame_to_second(
            sp_inserter.get_time_alimented_list(julius_output)
        )

        i_phoneme = 0
        new_phonemes = []
        for p in phonemes:
            if p == "pau" and time_alignment_list[i_phoneme][2] != "sp":
                continue
            i_phoneme += 1
            new_phonemes.append(p)

        aligned = JvsPhoneme.convert(
            [
                JvsPhoneme(start=float(o[0]), end=float(o[1]), phoneme=p)
                for p, o in zip(new_phonemes, time_alignment_list)
            ]
        )
        for p in aligned:
            p.verify()

        # world
        f0 = F0.from_wave(
            Wave.load(input_audio_path, sampling_rate=24000, dtype=numpy.float64),
            frame_period=5.0,
            f0_floor=71.0,
            f0_ceil=800,
            with_vuv=False,
            f0_type=F0Type.world,
        )
        converted_f0 = f0.convert(
            input_mean=f0.valid_f0_log.mean(),
            input_var=f0.valid_f0_log.var(),
            target_mean=_voiro_mean,
            target_var=f0.valid_f0_log.var(),
        )
        converted_f0.array = converted_f0.array.astype(numpy.float32).reshape(-1, 1)

        # feature
        phoneme_array = LinguisticFeature(
            phonemes=aligned,
            phoneme_class=JvsPhoneme,
            rate=_feature_rate,
            feature_types=[LinguisticFeature.FeatureType.PHONEME],
        ).make_array()

        phoneme = SamplingData(array=phoneme_array, rate=_feature_rate)

        feature = SamplingData.collect(
            [converted_f0, phoneme],
            rate=_feature_rate,
            mode="min",
            error_time_length=0.015,
        )

    return StreamingResponse(BytesIO(feature.astype(numpy.float32).tobytes()))
