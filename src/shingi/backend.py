"""One process owns the native model and serializes access to its context."""
import json
import selectors
import subprocess
import threading

from .gpu import gpu_free_mib, gpu_profile

CONTEXT_TOKENS = 16384


class NativeReadout:
    def __init__(self, executable, model, projector=None):
        self.lock = threading.Lock()
        _, preload, self.headroom = gpu_profile()
        if gpu_free_mib() < preload:
            raise RuntimeError(f"Shingi 27B requires at least {preload} MiB free on the GPU before loading")
        # The optional third argument is the vision projector; without it the readout is text only.
        command = [str(executable), str(model), str(CONTEXT_TOKENS)] + ([str(projector)] if projector else [])
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
        try:
            self.info = self._read(300)
            if not self.info.get("ready"):
                raise RuntimeError("native model failed to initialize")
            if bool(projector) != bool(self.info.get("vision")):
                raise RuntimeError("native readout vision state does not match the requested projector")
            if gpu_free_mib() < self.headroom:
                raise RuntimeError("insufficient GPU headroom after model load")
        except BaseException:
            self.close()
            raise

    def _read(self, timeout):
        with selectors.DefaultSelector() as selector:
            selector.register(self.process.stdout, selectors.EVENT_READ)
            if not selector.select(timeout):
                self.close()
                raise TimeoutError("native readout timed out; process stopped")
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError("native readout exited")
        try:
            result = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError("native readout returned malformed JSON") from exc
        if "error" in result:
            error = ValueError if result.get("error_kind") == "input" else RuntimeError
            raise error(result["error"])
        return result

    @property
    def vision(self):
        return bool(self.info.get("vision"))

    def infer(self, prompt, labels, images=None):
        with self.lock:
            if gpu_free_mib() < self.headroom:
                self.close()
                raise RuntimeError("GPU headroom fell below profile floor; native model stopped")
            if self.process.poll() is not None:
                raise RuntimeError("native readout is not running")
            request = {"prompt": prompt, "labels": labels}
            if images:
                request["images"] = images
            self.process.stdin.write(json.dumps(request) + "\n")
            self.process.stdin.flush()
            return self._read(300)

    def close(self):
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
