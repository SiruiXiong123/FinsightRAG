import argparse
import os
import time
import urllib.error
import urllib.request
from pathlib import Path


WORKSPACE_DIR = Path(os.getenv("OCR_WORKSPACE_DIR", "/workspace/work"))
DEFAULT_DATA_DIR = WORKSPACE_DIR / "data"
DEFAULT_OUTPUT_DIR = WORKSPACE_DIR / "data" / "output"
SUPPORTED_PATTERNS = ("*.pdf", "*.png", "*.jpg", "*.jpeg")
DEFAULT_DEVICE = os.getenv("OCR_DEVICE", "cpu")
DEFAULT_VLM_BACKEND = os.getenv("OCR_VLM_BACKEND", "vllm-server")
DEFAULT_VLM_SERVER_URL = os.getenv(
    "OCR_VLM_SERVER_URL",
    "http://host.docker.internal:8118/v1",
)


def parse_bool(value):
    if isinstance(value, bool):
        return value
    lowered = value.strip().lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def build_parser():
    parser = argparse.ArgumentParser(
        description="Batch process PDFs/images with PaddleOCR-VL."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--file", type=Path, action="append", dest="files")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--small-first", type=parse_bool, default=True)
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    parser.add_argument("--pipeline-version", default="v1.6")
    parser.add_argument("--precision", default="fp16")
    parser.add_argument("--use-queues", type=parse_bool, default=False)
    parser.add_argument("--orientation", type=parse_bool, default=False)
    parser.add_argument("--unwarping", type=parse_bool, default=False)
    parser.add_argument("--layout", type=parse_bool, default=True)
    parser.add_argument("--chart", type=parse_bool, default=False)
    parser.add_argument("--seal", type=parse_bool, default=False)
    parser.add_argument("--ocr-image-block", type=parse_bool, default=False)
    parser.add_argument("--format-block-content", type=parse_bool, default=False)
    parser.add_argument("--merge-layout-blocks", type=parse_bool, default=True)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--min-pixels", type=int, default=None)
    parser.add_argument("--max-pixels", type=int, default=None)
    parser.add_argument("--restructure", type=parse_bool, default=True)
    parser.add_argument("--merge-tables", type=parse_bool, default=True)
    parser.add_argument("--relevel-titles", type=parse_bool, default=True)
    parser.add_argument(
        "--vl-rec-backend",
        default=DEFAULT_VLM_BACKEND,
        help=(
            "VLM service backend, for example vllm-server or sglang-server. "
            "Short aliases vllm and sglang are also accepted. Use none to "
            "run without a VLM service."
        ),
    )
    parser.add_argument("--vl-rec-server-url", default=DEFAULT_VLM_SERVER_URL)
    parser.add_argument("--vl-rec-max-concurrency", type=int, default=1)
    parser.add_argument("--vl-rec-api-model-name", default=None)
    parser.add_argument("--vl-rec-api-key", default=None)
    parser.add_argument("--skip-vlm-check", type=parse_bool, default=False)
    return parser


def collect_files(args):
    if args.files:
        files = [p if p.is_absolute() else (Path.cwd() / p) for p in args.files]
    else:
        files = []
        for pattern in SUPPORTED_PATTERNS:
            files.extend(args.data_dir.glob(pattern))

    files = [p.resolve() for p in files if p.exists()]
    if args.small_first:
        files.sort(key=lambda p: (p.stat().st_size, p.name.lower()))
    else:
        files.sort(key=lambda p: p.name.lower())
    if args.limit is not None:
        files = files[: args.limit]
    return files


def normalize_backend(backend):
    if backend in (None, ""):
        return None
    backend = backend.strip()
    if backend.lower() in {"0", "false", "local", "none", "null", "off"}:
        return None
    aliases = {
        "vllm": "vllm-server",
        "sglang": "sglang-server",
        "fastdeploy": "fastdeploy-server",
    }
    normalized = aliases.get(backend, backend)
    allowed = {
        "vllm-server",
        "sglang-server",
        "fastdeploy-server",
        "mlx-vlm-server",
        "llama-cpp-server",
    }
    if normalized not in allowed:
        raise ValueError(f"Unsupported VLM backend: {backend}")
    return normalized


def check_vlm_server(server_url):
    models_url = f"{server_url.rstrip('/')}/models"
    try:
        with urllib.request.urlopen(models_url, timeout=15) as response:
            if response.status != 200:
                raise RuntimeError(f"unexpected HTTP status {response.status}")
    except (OSError, urllib.error.URLError, RuntimeError) as exc:
        raise RuntimeError(
            "Cannot reach the PaddleOCR-VL vLLM service. Start it first with "
            "scripts\\start_vllm_server.ps1, then retry. "
            f"Checked URL: {models_url}. Error: {exc}"
        ) from exc


def save_result(res, path):
    res.save_to_json(save_path=path.with_suffix(".json"))
    res.save_to_markdown(save_path=path.with_suffix(".md"))


def load_paddleocr_runtime():
    import paddle
    from paddleocr import PaddleOCRVL

    return paddle, PaddleOCRVL


def build_pipeline_kwargs(args):
    pipeline_kwargs = {
        "pipeline_version": args.pipeline_version,
        "device": args.device,
        "precision": args.precision,
        "use_doc_orientation_classify": args.orientation,
        "use_doc_unwarping": args.unwarping,
        "use_layout_detection": args.layout,
        "use_chart_recognition": args.chart,
        "use_seal_recognition": args.seal,
        "use_ocr_for_image_block": args.ocr_image_block,
        "format_block_content": args.format_block_content,
        "merge_layout_blocks": args.merge_layout_blocks,
        "use_queues": args.use_queues,
    }
    if args.vl_rec_backend:
        pipeline_kwargs.update(
            {
                "vl_rec_backend": args.vl_rec_backend,
                "vl_rec_server_url": args.vl_rec_server_url,
                "vl_rec_max_concurrency": args.vl_rec_max_concurrency,
                "vl_rec_api_model_name": args.vl_rec_api_model_name,
                "vl_rec_api_key": args.vl_rec_api_key,
            }
        )
    return pipeline_kwargs


def build_predict_kwargs(args):
    predict_kwargs = {
        "use_doc_orientation_classify": args.orientation,
        "use_doc_unwarping": args.unwarping,
        "use_layout_detection": args.layout,
        "use_chart_recognition": args.chart,
        "use_seal_recognition": args.seal,
        "use_ocr_for_image_block": args.ocr_image_block,
        "format_block_content": args.format_block_content,
        "merge_layout_blocks": args.merge_layout_blocks,
        "use_queues": args.use_queues,
        "max_new_tokens": args.max_new_tokens,
    }
    if args.min_pixels is not None:
        predict_kwargs["min_pixels"] = args.min_pixels
    if args.max_pixels is not None:
        predict_kwargs["max_pixels"] = args.max_pixels
    return predict_kwargs


def run_batch(args):
    args.data_dir = args.data_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    args.vl_rec_backend = normalize_backend(args.vl_rec_backend)
    if args.vl_rec_backend and not args.vl_rec_server_url:
        args.vl_rec_server_url = DEFAULT_VLM_SERVER_URL
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.vl_rec_backend and not args.skip_vlm_check:
        print(f"Checking VLM service: {args.vl_rec_server_url}", flush=True)
        check_vlm_server(args.vl_rec_server_url)

    files = collect_files(args)
    if not files:
        print(f"No supported files found in {args.data_dir}", flush=True)
        return 1

    paddle, PaddleOCRVL = load_paddleocr_runtime()

    print(f"Paddle: {paddle.__version__}", flush=True)
    print(f"CUDA compiled: {paddle.device.is_compiled_with_cuda()}", flush=True)
    print(f"Device requested: {args.device}", flush=True)
    if args.vl_rec_backend:
        print(
            f"VLM service: {args.vl_rec_backend} at {args.vl_rec_server_url}",
            flush=True,
        )
    print(f"Input files: {len(files)}", flush=True)
    for file_path in files:
        size_mb = file_path.stat().st_size / 1024 / 1024
        print(f"  - {file_path.name} ({size_mb:.2f} MB)", flush=True)

    print("Loading PaddleOCR-VL pipeline...", flush=True)
    load_started = time.perf_counter()
    pipeline = PaddleOCRVL(**build_pipeline_kwargs(args))
    print(f"Pipeline loaded in {time.perf_counter() - load_started:.1f}s", flush=True)

    predict_kwargs = build_predict_kwargs(args)
    failures = 0
    for index, file_path in enumerate(files, start=1):
        print(f"\n[{index}/{len(files)}] Processing {file_path.name}", flush=True)
        started = time.perf_counter()
        pages_res = []
        page_output_dir = args.output_dir / file_path.stem
        page_output_dir.mkdir(parents=True, exist_ok=True)

        try:
            for page_index, res in enumerate(
                pipeline.predict(input=str(file_path), **predict_kwargs),
                start=1,
            ):
                pages_res.append(res)
                page_base = page_output_dir / f"page_{page_index:04d}"
                save_result(res, page_base)
                elapsed = time.perf_counter() - started
                print(
                    f"  page {page_index} saved after {elapsed:.1f}s",
                    flush=True,
                )

            if args.restructure and pages_res:
                print("  restructuring pages...", flush=True)
                restructured = pipeline.restructure_pages(
                    pages_res,
                    merge_tables=args.merge_tables,
                    relevel_titles=args.relevel_titles,
                    concatenate_pages=True,
                )
                for res_index, res in enumerate(restructured, start=1):
                    suffix = "" if res_index == 1 else f"_{res_index}"
                    save_result(res, args.output_dir / f"{file_path.stem}{suffix}")

            elapsed = time.perf_counter() - started
            print(f"Done {file_path.name} in {elapsed:.1f}s", flush=True)
        except Exception as exc:
            failures += 1
            print(f"Failed {file_path.name}: {exc}", flush=True)

    print(f"\nOutput directory: {args.output_dir}", flush=True)
    return 1 if failures else 0


def main(argv=None):
    args = build_parser().parse_args(argv)
    return run_batch(args)


if __name__ == "__main__":
    raise SystemExit(main())
