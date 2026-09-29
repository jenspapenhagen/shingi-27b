import argparse
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, HTTPException
import uvicorn

from .backend import CONTEXT_TOKENS, NativeReadout
from .decision import Calibration, DecisionEngine, MODEL_ID
from .model import (CALIBRATION_FILE, CALIBRATION_SHA256, MODEL_FILE, MODEL_SHA256, RUNTIME,
                    fetch, load_calibration, sha256, verify)
from .schema import Request

ALIASES = (MODEL_ID, "shingi", "shingi-latest", "jev-latest", "openjev")


def create_app(engine=None, *, executable=None, model=None, identity=None,
               calibration=Calibration(), calibration_sha256=None):
    identity = identity or {"model": MODEL_ID, "model_sha256": None, "verified": False}

    @asynccontextmanager
    async def lifespan(app):
        backend = None
        try:
            if engine is None:
                backend = NativeReadout(executable, model)
                app.state.engine = DecisionEngine(backend, calibration, model_id=identity["model"])
            yield
        finally:
            if backend is not None:
                backend.close()

    app = FastAPI(title="Shingi 27B", lifespan=lifespan)
    app.state.engine = engine

    @app.get("/health")
    def health():
        current = app.state.engine
        process = getattr(getattr(current, "backend", None), "process", None)
        if current is None or (process is not None and process.poll() is not None):
            raise HTTPException(503, "native model unavailable")
        return {"status": "ok"}

    @app.get("/v1/version")
    def version():
        return {**identity, "calibration": asdict(app.state.engine.calibration),
                "calibration_sha256": calibration_sha256, "runtime": RUNTIME,
                "context_tokens": CONTEXT_TOKENS, "kv_cache": "q8_0", "choice_order": "sorted",
                "single_pass_options": 52, "choice_limit": 255, "image_input": False}

    @app.get("/v1/models")
    def models():
        return {"models": [{"name": identity["model"], "description": "Shingi 27B local decision model; text only.",
                            "release_date": "2026-09-29"}]}

    @app.post("/v1/systemone")
    def system_one(body: Request):
        if body.model not in ALIASES:
            raise HTTPException(422, "unknown model alias")
        try:
            response, _ = app.state.engine.evaluate(body.model_dump(exclude_none=True))
            return response
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        except (RuntimeError, TimeoutError, BrokenPipeError) as exc:
            raise HTTPException(503, "native readout unavailable") from exc

    return app


def prepare(model=None, calibration=None, skip_verify=False):
    """Fetch omitted files from the Hub, verify pinned hashes, and return the serving identity."""
    calibration_path = calibration or fetch(CALIBRATION_FILE)
    parameters, calibration_sha256 = load_calibration(calibration_path)
    if not skip_verify:
        verify("Calibration", calibration_sha256, CALIBRATION_SHA256)
    model = model or fetch(MODEL_FILE)
    model_sha256 = None
    if not skip_verify:
        print(f"Verifying {model} ...", flush=True)
        model_sha256 = sha256(model)
        verify("Model", model_sha256, MODEL_SHA256)
    identity = {"model": MODEL_ID, "model_sha256": model_sha256, "verified": not skip_verify}
    return model, identity, parameters, calibration_sha256


def main():
    parser = argparse.ArgumentParser(prog="shingi-27b", description="Serve the Shingi 27B decision API.")
    parser.add_argument("--executable", required=True, type=Path,
                        help="native readout built against the pinned Prism runtime")
    parser.add_argument("--model", type=Path, help=f"{MODEL_FILE}; downloaded from Hugging Face when omitted")
    parser.add_argument("--calibration", type=Path,
                        help=f"{CALIBRATION_FILE}; downloaded from Hugging Face when omitted")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--skip-verify", action="store_true",
                        help="do not check the pinned SHA-256 of the model and calibration")
    args = parser.parse_args()
    if not args.executable.is_file():
        parser.error(f"readout executable not found: {args.executable}")
    model, identity, calibration, calibration_sha256 = prepare(args.model, args.calibration, args.skip_verify)
    # The API has no authentication. Bind beyond localhost only behind your own access control.
    uvicorn.run(create_app(executable=args.executable, model=model, identity=identity,
                           calibration=calibration, calibration_sha256=calibration_sha256),
                host=args.host, port=args.port)


if __name__ == "__main__":
    main()
