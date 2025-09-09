import time
import queue
import threading
from dataclasses import dataclass
import numpy as np

import warnings
from soundcard import SoundcardRuntimeWarning

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=SoundcardRuntimeWarning)



try:
    from numpy.lib import NumpyVersion
except Exception:
    class NumpyVersion:
        def __init__(self, v): self.v = v
        def __ge__(self, other): return str(self.v) >= str(other)

if NumpyVersion(np.__version__) >= "2.0.0":
    _orig_fromstring = np.fromstring
    def _fromstring_compat(string, dtype=float, count=-1, sep=""):
        if sep == "" and isinstance(string, (bytes, bytearray, memoryview)):
            return np.frombuffer(string, dtype=dtype, count=count)
        return _orig_fromstring(string, dtype=dtype, count=count, sep=sep)
    np.fromstring = _fromstring_compat

import soundcard as sc
from faster_whisper import WhisperModel

try:
    import argostranslate.package as argos_pkg
    import argostranslate.translate as argos_tr
    ARGOS_OK = True
except Exception:
    ARGOS_OK = False

WHISPER_MODEL = "large-v3"# tiny|base|small|medium|large-v3
ASR_TARGET_SR = 16000
BLOCK_SEC = 0.1
CHUNK_SEC = 5.0
OVERLAP_SEC = 1.0
VAD_FILTER = True
PRINT_PARTIALS = True

PREFERRED_SPEAKER_NAME = None

SR_CANDIDATES = [48000, 44100]
CHANNELS_CANDIDATES = [2, 1]


def linear_resample(x: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if src_sr == dst_sr or x.size == 0:
        return x.astype(np.float32)
    x = x.reshape(-1).astype(np.float32)
    n_src = x.shape[0]
    dur = n_src / src_sr
    n_dst = int(round(dur * dst_sr))
    if n_dst <= 1:
        return np.zeros((0,), dtype=np.float32)
    t_src = np.linspace(0.0, dur, num=n_src, endpoint=False)
    t_dst = np.linspace(0.0, dur, num=n_dst, endpoint=False)
    return np.interp(t_dst, t_src, x)


def pick_speaker():
    if PREFERRED_SPEAKER_NAME:
        for sp in sc.all_speakers():
            if PREFERRED_SPEAKER_NAME.lower() in sp.name.lower():
                return sp
    return sc.default_speaker()


@dataclass
class Chunk:
    pcm_f32_16k: np.ndarray
    start_ts: float


def ensure_argos_pair(src_lang: str, tgt_lang: str) -> bool:
    if not ARGOS_OK:
        return False
    s, t = src_lang.split("-")[0], tgt_lang.split("-")[0]
    for lang in argos_tr.get_installed_languages():
        if lang.code == s:
            try:
                _ = lang.translate("", t)
                return True
            except Exception:
                continue
    try:
        for p in argos_pkg.get_available_packages():
            if p.from_code == s and p.to_code == t:
                path = p.download()
                argos_pkg.install_from_path(path)
                return True
    except Exception:
        pass
    return False


def main():
    from_language = input("Enter source language (e.g. en, pl, ru): ")
    to_language = input("Enter target language (e.g. en, pl, ru): ")
    computing_device = input("Enter computing device (e.g. cpu, cuda): ")
    model_size = input("Enter model size (tiny, base, small, medium, large-v3): ")
    speaker = pick_speaker()
    loopback_mic = sc.get_microphone(speaker.name, include_loopback=True)

    native_sr = None
    rec = None
    last_err = None

    print("Loopback via python-soundcard")
    print(f"Output device: {speaker.name}")
    print("Ctrl+C to exit\n")

    for sr in SR_CANDIDATES:
        for ch in CHANNELS_CANDIDATES:
            try:
                test = loopback_mic.recorder(samplerate=sr, channels=ch)
                test.__enter__()
                test.record(int(sr * 0.05))
                test.__exit__(None, None, None)
                rec = loopback_mic.recorder(samplerate=sr, channels=ch)
                rec.__enter__()
                native_sr, channels = sr, ch
                print(f"✔ Recorder works: samplerate={sr}, channels={ch}\n")
                break
            except Exception as e:
                last_err = e
                continue
        if rec is not None:
            break

    if rec is None:
        print(f"Unable to open loopback: {last_err}")
        return

    model = WhisperModel(model_size, device=computing_device, compute_type="int8")

    audio_q: queue.Queue[np.ndarray] = queue.Queue(maxsize=50)
    chunk_q: queue.Queue[Chunk] = queue.Queue(maxsize=10)
    stop_event = threading.Event()
    samples_per_chunk_16k = int(ASR_TARGET_SR * CHUNK_SEC)
    samples_overlap_16k = int(ASR_TARGET_SR * OVERLAP_SEC)
    build_buf_16k = np.zeros(0, dtype=np.float32)
    last_chunk_ts = time.time()

    frames_per_block = int(native_sr * BLOCK_SEC)

    def reader():
        try:
            while not stop_event.is_set():
                data = rec.record(frames_per_block)
                if data.ndim == 2 and data.shape[1] > 1:
                    data = data.mean(axis=1)
                else:
                    data = data.reshape(-1)
                y = linear_resample(data, native_sr, ASR_TARGET_SR).astype(np.float32)
                if y.size:
                    audio_q.put(y)
        finally:
            try:
                rec.__exit__(None, None, None)
            except Exception:
                pass

    def chunker():
        nonlocal build_buf_16k, last_chunk_ts
        while not stop_event.is_set():
            try:
                y = audio_q.get(timeout=0.2)
            except queue.Empty:
                continue
            build_buf_16k = np.concatenate([build_buf_16k, y])
            while build_buf_16k.size >= samples_per_chunk_16k:
                part = build_buf_16k[:samples_per_chunk_16k]
                keep = build_buf_16k[samples_per_chunk_16k - samples_overlap_16k:] if samples_overlap_16k > 0 else build_buf_16k[samples_per_chunk_16k:]
                build_buf_16k = keep.copy()
                chunk_q.put(Chunk(pcm_f32_16k=part, start_ts=last_chunk_ts))
                last_chunk_ts = time.time()

    printed_so_far = ""

    def process_text_and_print(new_text: str, src_lang_whisper: str):
        nonlocal printed_so_far
        pref = 0
        m = min(len(printed_so_far), len(new_text))
        while pref < m and printed_so_far[pref] == new_text[pref]:
            pref += 1
        suffix = new_text[pref:].strip()
        if not suffix:
            return
        print(f"[ASR] {suffix}")
        if ARGOS_OK:
            src = (src_lang_whisper or "auto").split("-")[0]
            tgt = from_language.split("-")[0]
            if src != tgt and ensure_argos_pair(src, tgt):
                try:
                    installed = argos_tr.get_installed_languages()
                    src_obj = next((l for l in installed if l.code == src), None)
                    if src_obj:
                        tr = next((t for t in src_obj.translations if t.code == tgt), None)
                        if tr:
                            print(f"   → [{tgt}] {tr.translate(suffix)}")
                except Exception:
                    pass
        printed_so_far = (printed_so_far + " " + suffix).strip()

    def asr_worker():
        while not stop_event.is_set():
            try:
                ch = chunk_q.get(timeout=0.2)
            except queue.Empty:
                continue
            segments, info = model.transcribe(
                ch.pcm_f32_16k,
                language=to_language if to_language else 'en',
                beam_size=5,
                vad_filter=VAD_FILTER,
                vad_parameters=dict(min_silence_duration_ms=500),
                condition_on_previous_text=True,
                temperature=0.0,
                word_timestamps=False,
            )
            text = " ".join(seg.text.strip() for seg in segments if seg.text)
            if PRINT_PARTIALS and text:
                process_text_and_print(text, info.language)

    t_reader = threading.Thread(target=reader, daemon=True)
    t_chunk = threading.Thread(target=chunker, daemon=True)
    t_asr = threading.Thread(target=asr_worker, daemon=True)
    t_reader.start()
    t_chunk.start()
    t_asr.start()

    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nEnding...")
    finally:
        stop_event.set()
        time.sleep(0.5)


if __name__ == "__main__":
    main()
