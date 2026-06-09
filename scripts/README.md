# Script Entry Points

The only supported Python entry point is the unified CLI:

```powershell
finsightrag <subcommand> ...
```

This directory keeps PowerShell scripts for environment orchestration, such as
Docker-based OCR and PaddleOCR-VL vLLM startup. Python pipeline commands should
be run through `finsightrag`.

## Runtime Setup

| Task | Script |
| --- | --- |
| Start the PaddleOCR-VL vLLM service | `start_vllm_server.ps1` |
| Run PaddleOCR-VL OCR | `run_ocr.ps1` |

## Document Pipeline

| Stage | CLI command |
| --- | --- |
| Generate text-only Markdown | `finsightrag generate-text-md` |
| Chunk document text | `finsightrag chunk-text` |
| Crop table and image assets | `finsightrag extract-assets` |
| Enrich cropped assets | `finsightrag enrich-assets` |
| Build FAISS indexes | `finsightrag build-indexes` |

## Query Tools

| Task | CLI command |
| --- | --- |
| Search indexes | `finsightrag search-indexes` |
| Build evidence package | `finsightrag build-evidence-package` |
| Generate final answer | `finsightrag generate-answer` |
| Run retrieval-to-answer pipeline | `finsightrag run-query-pipeline` |

## Helpers

`paddleocr_model_volume.ps1` contains shared PowerShell functions used by
`start_vllm_server.ps1` and `run_ocr.ps1`; it is not a standalone entry point.
