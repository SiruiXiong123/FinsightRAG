import glob
import logging
import os
from getpass import getpass
from pathlib import Path
from typing import List, Optional

from dotenv import find_dotenv, load_dotenv
from llama_index.core import Document, VectorStoreIndex, get_response_synthesizer
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.core.postprocessor import LLMRerank
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.schema import TextNode
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.llms.gemini import Gemini

try:
    from paddleocr_adapter import OpenAICompatibleVisionSummarizer, PaddleOCRPreprocessor
    from rag_config import RagConfig
except ImportError:  # Allows package-style imports in tests.
    from .paddleocr_adapter import OpenAICompatibleVisionSummarizer, PaddleOCRPreprocessor
    from .rag_config import RagConfig


class DocProcessor:
    """
    Processes PDFs through PaddleOCR-VL outputs before RAG indexing.

    Text is read from the OCR Markdown file and split into semantic nodes. Tables,
    images, seals, charts, and similar non-text blocks are read from the OCR JSON
    plus Markdown, summarized by the configured multimodal model, and converted to
    TextNode objects.
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        ocr_output_dir: Optional[str] = None,
        paddleocr_command: Optional[str] = None,
    ) -> None:
        self.config = RagConfig.load(config_path)
        self.retrieve_or_prompt_api_key()

        self.embed_model_name = "models/text-embedding-004"
        self.llm_name = "models/gemini-1.5-flash"

        self.embed_model = GeminiEmbedding(model_name=self.embed_model_name)
        self.semantic_splitter = SemanticSplitterNodeParser(
            buffer_size=1,
            embed_model=self.embed_model,
        )
        self.llm = Gemini(
            model=self.llm_name,
            generate_kwargs={"timeout": 2000},
        )

        self.supported_extensions = [".pdf"]
        self.logger = Logger(log_file="../RAG_log.log", logger_name=__name__).get_logger()
        self.ocr_preprocessor = PaddleOCRPreprocessor(
            config=self.config,
            output_dir=ocr_output_dir,
            command=paddleocr_command,
            logger=self.logger,
        )
        self.multimodal_summarizer = OpenAICompatibleVisionSummarizer(
            config=self.config,
            logger=self.logger,
        )

    def retrieve_or_prompt_api_key(self) -> None:
        """
        Load API keys needed by the unchanged retrieval/generation stack.

        PaddleOCR-VL multimodal summaries use config.yaml or environment variables
        through OpenAICompatibleVisionSummarizer. The existing Gemini embedding and
        final answer generation still need GOOGLE_API_KEY.
        """
        load_dotenv(find_dotenv())
        if "GOOGLE_API_KEY" not in os.environ:
            os.environ["GOOGLE_API_KEY"] = getpass("Please provide your GOOGLE_API_KEY:")

    def list_supported_files(self, input_path: str) -> List[str]:
        """
        Accept a single PDF, a directory, or a glob pattern, and return PDFs only.
        """
        if not input_path:
            return []

        candidate = Path(os.path.expandvars(input_path)).expanduser()
        files: List[Path] = []

        if candidate.exists():
            if candidate.is_file():
                files = [candidate]
            elif candidate.is_dir():
                files = sorted(candidate.rglob("*"))
        else:
            files = [Path(path) for path in glob.glob(input_path, recursive=True)]

        supported = []
        for file_path in files:
            if file_path.is_file() and file_path.suffix.lower() in self.supported_extensions:
                supported.append(str(file_path.resolve()))
        return supported

    def get_semantic_nodes(self, files_to_process: List[str]):
        """
        Build semantic text nodes from PaddleOCR-VL Markdown.
        """
        documents = []
        for file_path in files_to_process:
            ocr_result = self.ocr_preprocessor.load_result(file_path)
            text = self.ocr_preprocessor.clean_markdown_for_text(ocr_result.markdown)
            if not text:
                self.logger.warning("No OCR Markdown text found for %s", file_path)
                continue
            documents.append(
                Document(
                    text=text,
                    metadata={
                        "source_file": Path(file_path).stem,
                        "parser": "paddleocr_vl",
                        "ocr_json_path": str(ocr_result.json_path),
                        "ocr_markdown_path": str(ocr_result.md_path),
                    },
                )
            )

        if not documents:
            return []
        return self.semantic_splitter.get_nodes_from_documents(documents)

    def get_multimodal_nodes(self, files_to_process: List[str]) -> List[TextNode]:
        """
        Summarize PaddleOCR-VL non-text blocks and turn them into text nodes.
        """
        nodes: List[TextNode] = []
        for file_path in files_to_process:
            ocr_result = self.ocr_preprocessor.load_result(file_path)
            elements = self.ocr_preprocessor.extract_multimodal_elements(ocr_result)
            self.logger.info(
                "Found %s PaddleOCR-VL non-text blocks in %s",
                len(elements),
                file_path,
            )

            for element in elements:
                summary = self.multimodal_summarizer.summarize(element)
                node = TextNode(
                    text=summary,
                    metadata={
                        "source_file": element.source_file,
                        "modality": "multimodal_summary",
                        "element_type": element.kind,
                        "page_number": element.page_number,
                        "block_id": element.block_id,
                        "global_block_id": element.global_block_id,
                        "bbox": element.bbox,
                        "image_path": str(element.image_path) if element.image_path else None,
                        "ocr_json_path": str(ocr_result.json_path),
                        "ocr_markdown_path": str(ocr_result.md_path),
                    },
                )
                nodes.append(node)

        return nodes


class QueryEngine:
    """
    Builds the existing retrieval/generation engine over OCR text and multimodal summaries.
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        ocr_output_dir: Optional[str] = None,
        paddleocr_command: Optional[str] = None,
    ) -> None:
        self.doc_processor = DocProcessor(
            config_path=config_path,
            ocr_output_dir=ocr_output_dir,
            paddleocr_command=paddleocr_command,
        )
        self.logger = Logger(log_file="../RAG_log.log", logger_name=__name__).get_logger()

    def build_recursive_retriever(self, files_to_process, top_k=10):
        """
        Builds the current LlamaIndex retriever/query engine.
        """
        semantic_nodes = self.doc_processor.get_semantic_nodes(files_to_process)
        self.logger.info("PaddleOCR-VL Markdown semantic nodes are extracted")

        multimodal_nodes = self.doc_processor.get_multimodal_nodes(files_to_process)
        self.logger.info("PaddleOCR-VL multimodal summary nodes are extracted")

        all_nodes = semantic_nodes + multimodal_nodes
        if not all_nodes:
            raise RuntimeError("No RAG nodes were created from the provided files.")

        vector_index = VectorStoreIndex(
            all_nodes,
            embed_model=self.doc_processor.embed_model,
        )
        vector_retriever = vector_index.as_retriever(similarity_top_k=top_k)

        response_synthesizer = get_response_synthesizer(
            llm=self.doc_processor.llm,
            response_mode="compact",
        )

        return RetrieverQueryEngine(
            retriever=vector_retriever,
            node_postprocessors=[
                LLMRerank(
                    llm=self.doc_processor.llm,
                    choice_batch_size=5,
                    top_n=3,
                )
            ],
            response_synthesizer=response_synthesizer,
        )


class Logger:
    def __init__(
        self,
        log_file: str,
        logger_name: str,
        log_level: int = logging.INFO,
    ):
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(log_level)

        if not self.logger.handlers:
            fhandler = logging.FileHandler(filename=log_file, mode="w", encoding="utf-8")
            formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            fhandler.setFormatter(formatter)
            self.logger.addHandler(fhandler)

    def get_logger(self):
        return self.logger
