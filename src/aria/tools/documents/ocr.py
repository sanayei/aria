"""OCR tool using Surya OCR for text extraction from scanned documents."""

from pathlib import Path

from pydantic import BaseModel, Field
from pdf2image import convert_from_path
from PIL import Image

from aria.tools import BaseTool, ToolResult, RiskLevel
from aria.tools.documents.models import OCRResult


class OCRParams(BaseModel):
    """Parameters for OCR extraction."""

    file_path: str = Field(description="Path to image or PDF file")


class OCRTool(BaseTool[OCRParams]):
    """Extract text from scanned documents using Surya OCR models."""

    name = "ocr_extract"
    description = "Extract text from scanned documents (PDF or images) using OCR"
    risk_level = RiskLevel.LOW
    parameters_schema = OCRParams

    def __init__(self) -> None:
        """Initialize the OCR tool with Surya models."""
        super().__init__()
        self._models_loaded = False
        self._foundation_predictor = None
        self._detector = None
        self._recognizer = None
        self._task_name = None

    def _load_models(self) -> None:
        """Lazy load Surya OCR models (expensive operation)."""
        if self._models_loaded:
            return

        try:
            from surya.common.surya.schema import TaskNames
            from surya.detection import DetectionPredictor
            from surya.foundation import FoundationPredictor
            from surya.recognition import RecognitionPredictor

            foundation_predictor = FoundationPredictor()
            det_predictor = DetectionPredictor()
            rec_predictor = RecognitionPredictor(foundation_predictor)

            self._foundation_predictor = foundation_predictor
            self._detector = det_predictor
            self._recognizer = rec_predictor
            self._task_name = TaskNames.ocr_with_boxes

            self._models_loaded = True
        except Exception as e:
            raise RuntimeError(f"Failed to load Surya OCR models: {e}") from e

    async def execute(self, params: OCRParams) -> ToolResult:
        """Extract text from document using OCR."""
        try:
            path = Path(params.file_path).expanduser().resolve()

            if not path.exists():
                return ToolResult.error_result(f"File does not exist: {path}")

            if not path.is_file():
                return ToolResult.error_result(f"Path is not a file: {path}")

            # Load models if not already loaded
            self._load_models()

            # Convert PDF to images or load single image
            if path.suffix.lower() == ".pdf":
                images = convert_from_path(str(path))
            else:
                # Single image file
                try:
                    images = [Image.open(path)]
                except Exception as e:
                    return ToolResult.error_result(f"Failed to open image: {e}")

            # Run OCR on all pages/images
            try:
                predictions = self._recognizer(
                    images,
                    task_names=[self._task_name] * len(images),
                    det_predictor=self._detector,
                )
            except Exception as e:
                return ToolResult.error_result(f"OCR processing failed: {e}")

            # Extract text and calculate average confidence
            all_text: list[str] = []
            all_confidences: list[float] = []

            for page_pred in predictions:
                for text_line in page_pred.text_lines:
                    all_text.append(text_line.text)
                    all_confidences.append(text_line.confidence)

            extracted_text = "\n".join(all_text)
            avg_confidence = (
                sum(all_confidences) / len(all_confidences) if all_confidences else 0.0
            )

            ocr_result = OCRResult(
                text=extracted_text,
                confidence=avg_confidence,
                page_count=len(images),
            )

            return ToolResult.success_result(
                data={
                    "file_path": str(path),
                    "text": ocr_result.text,
                    "confidence": round(ocr_result.confidence, 2),
                    "page_count": ocr_result.page_count,
                    "char_count": len(extracted_text),
                }
            )

        except Exception as e:
            return ToolResult.error_result(f"OCR failed: {e}")

    def get_confirmation_message(self, params: OCRParams) -> str:
        """Get confirmation message (not needed for low-risk OCR)."""
        return f"Extract text from: {params.file_path}"
