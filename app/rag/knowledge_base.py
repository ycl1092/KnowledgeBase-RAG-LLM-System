"""
知识库管理

文档加载、分块、MD5 去重、写入向量库。
支持 TXT / MD / PDF / DOCX / JSON / 图片（OCR 占位）。
"""

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.config import settings
from app.core.logger import logger
from app.rag.vector_store import vector_store


class KnowledgeBase:
    """知识库：文档录入与去重"""

    # 支持的文件类型
    SUPPORTED_TYPES = {
        ".txt": "纯文本",
        ".md": "Markdown",
        ".json": "JSON 数据",
        ".pdf": "PDF 文档",
        ".docx": "Word 文档",
        ".png": "图片（OCR）",
        ".jpg": "图片（OCR）",
        ".jpeg": "图片（OCR）",
    }

    def __init__(self):
        md5_path = settings.get("chroma.md5_store", "data/md5.txt")
        self.md5_path = Path(settings.ROOT_DIR) / md5_path
        self.md5_path.parent.mkdir(parents=True, exist_ok=True)

        chunk_size = settings.get("chunk.size", 500)
        chunk_overlap = settings.get("chunk.overlap", 50)
        separators = settings.get("chunk.separators", ["\n\n", "\n", "。", " ", ""])

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators,
            length_function=len,
        )

        if not self.md5_path.exists():
            self.md5_path.touch()

    # ═══════════════════════════════════════
    # 公开接口
    # ═══════════════════════════════════════

    def upload_bytes(self, content: bytes, filename: str) -> str:
        """上传文件二进制内容到知识库"""
        text = self._extract_text(content, filename)
        return self.upload_text(text, source=filename)

    def upload_text(self, text: str, source: str = "manual") -> str:
        """上传一段文本到知识库"""
        if not text or not text.strip():
            return "skip: 内容为空"

        md5 = self._md5(text)

        if self._exists(md5):
            logger.info(f"[去重] 内容已存在: {md5[:8]}... ({source})")
            return "skip: 内容已存在知识库"

        # 长文本分块，短文本直接入库
        if len(text) > settings.get("chunk.size", 500):
            chunks = self.splitter.split_text(text)
        else:
            chunks = [text]

        metadatas = [{"source": source, "md5": md5} for _ in chunks]
        ids = vector_store.add_texts(chunks, metadatas)

        self._save_md5(md5)
        logger.info(f"[入库] {len(ids)} 个文档块，来源: {source}")
        return f"success: 已入库 {len(ids)} 个文档块"

    def upload_file(self, filepath: str, source: Optional[str] = None) -> str:
        """上传文件到知识库（从文件系统路径）"""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {filepath}")

        source = source or path.name
        content = path.read_bytes()
        return self.upload_bytes(content, source)

    def ingest_directory(self, dir_path: Optional[str] = None):
        """批量导入目录下所有支持的文件"""
        base = Path(dir_path) if dir_path else settings.ROOT_DIR / "data" / "raw"
        if not base.exists():
            logger.warning(f"目录不存在: {base}")
            return

        results = []
        allowed = tuple(self.SUPPORTED_TYPES.keys())
        for f in base.iterdir():
            if f.suffix.lower() in allowed:
                try:
                    result = self.upload_file(str(f))
                    results.append((f.name, result))
                    logger.info(f"  {f.name}: {result}")
                except Exception as e:
                    logger.error(f"  {f.name}: 失败 - {e}")
                    results.append((f.name, f"error: {e}"))
        return results

    # ═══════════════════════════════════════
    # 文件解析调度器
    # ═══════════════════════════════════════

    def _extract_text(self, content: bytes, filename: str) -> str:
        """根据文件后缀选择解析器"""
        suffix = Path(filename).suffix.lower()

        if suffix in (".txt", ".md"):
            return content.decode("utf-8")

        if suffix == ".json":
            return self._extract_json(content)

        if suffix == ".pdf":
            return self._extract_pdf(content, filename)

        if suffix == ".docx":
            return self._extract_docx(content)

        if suffix in (".png", ".jpg", ".jpeg", ".bmp", ".tiff"):
            return self._extract_image(content, filename)

        raise ValueError(
            f"不支持的文件格式: {suffix}。支持: {', '.join(self.SUPPORTED_TYPES.keys())}"
        )

    # ═══════════════════════════════════════
    # JSON 解析
    # ═══════════════════════════════════════

    def _extract_json(self, content: bytes) -> str:
        """从 JSON 中提取所有文本内容"""
        data = json.loads(content)

        if isinstance(data, list):
            parts = []
            for item in data:
                if isinstance(item, dict):
                    parts.append(
                        f"名称: {item.get('product_name', item.get('name', '未知'))}\n"
                        f"{item.get('manual', item.get('description', json.dumps(item, ensure_ascii=False)))}"
                    )
                else:
                    parts.append(str(item))
            return "\n\n---\n\n".join(parts)

        if isinstance(data, dict):
            return json.dumps(data, ensure_ascii=False, indent=2)

        return str(data)

    # ═══════════════════════════════════════
    # PDF 解析：pypdf → OCR → 多模态
    # ═══════════════════════════════════════

    def _extract_pdf(self, content: bytes, filename: str) -> str:
        """提取 PDF 文本。先试 pypdf，文字太少则尝试 OCR"""
        # 写临时文件（pypdf 需要文件路径或流）
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        try:
            from pypdf import PdfReader
            reader = PdfReader(tmp_path)
            texts = []
            for page in reader.pages:
                t = page.extract_text()
                if t:
                    texts.append(t.strip())

            text = "\n\n".join(texts)

            # 如果提取的文字太少（< 50 字符），可能是扫描件 → 尝试 OCR
            if len(text.strip()) < 50:
                logger.info(f"[PDF] pypdf 提取文本过短 ({len(text.strip())} 字符)，尝试 OCR")
                text = self._ocr_pdf(tmp_path) or text

            return text.strip()

        finally:
            try:
                os.unlink(tmp_path)
            except PermissionError:
                pass

    def _ocr_pdf(self, pdf_path: str) -> Optional[str]:
        """OCR 扫描版 PDF（需要 pytesseract + Tesseract 系统安装 + pdf2image）"""
        try:
            from pdf2image import convert_from_path
            from PIL import Image
            import pytesseract
        except ImportError:
            logger.warning(
                "[OCR] 缺少依赖: pdf2image / pytesseract / Pillow。"
                "如需 OCR 功能请安装: pip install pdf2image pytesseract Pillow"
            )
            return self._multimodal_fallback(pdf_path, "PDF 扫描件")

        try:
            images = convert_from_path(pdf_path, dpi=300)
            texts = []
            for i, img in enumerate(images):
                text = pytesseract.image_to_string(img, lang="chi_sim+eng")
                if text.strip():
                    texts.append(f"[第{i+1}页]\n{text.strip()}")

            return "\n\n".join(texts) if texts else None
        except Exception as e:
            logger.warning(f"[OCR] PDF OCR 失败: {e}")
            return None

    # ═══════════════════════════════════════
    # DOCX 解析
    # ═══════════════════════════════════════

    def _extract_docx(self, content: bytes) -> str:
        """提取 Word 文档文本"""
        with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        try:
            from docx import Document
            doc = Document(tmp_path)
            paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
            return "\n\n".join(paragraphs)
        finally:
            try:
                os.unlink(tmp_path)
            except PermissionError:
                pass

    # ═══════════════════════════════════════
    # 图片 OCR 及多模态占位
    # ═══════════════════════════════════════

    def _extract_image(self, content: bytes, filename: str) -> str:
        """图片 OCR 提取文字"""
        try:
            from PIL import Image
            import pytesseract
            import io

            image = Image.open(io.BytesIO(content))
            text = pytesseract.image_to_string(image, lang="chi_sim+eng")

            if text.strip():
                return f"[OCR 识别结果 - {filename}]\n{text.strip()}"

            logger.info(f"[图片] OCR 未提取到文字，尝试多模态: {filename}")
            return self._multimodal_fallback(content, filename)

        except ImportError:
            logger.warning(
                "[OCR] 缺少 pytesseract，无法识别图片文字。"
                "安装: pip install pytesseract，并安装 Tesseract-OCR 系统软件"
            )
            return self._multimodal_fallback(content, filename)

    def _multimodal_fallback(
        self, content_or_path, filename: str
    ) -> str:
        """多模态模型占位 —— 后续可接入 GPT-4o Vision / Qwen-VL

        当 OCR 不可用或失败时调用。
        接入后：将图片/PDF 页面发送给多模态 LLM，返回文字描述。
        """
        fname = filename if isinstance(filename, str) else "未知文件"
        logger.info(f"[多模态占位] 未启用多模态模型，跳过解析: {fname}")
        return (
            f"[未解析内容] 文件 '{fname}' 包含非文本内容（图片/扫描件），"
            f"当前未启用多模态模型解析。\n"
            f"[多模态占位] 后续可接入 GPT-4o Vision 或 Qwen-VL 等模型，"
            f"自动识别图片和扫描件内容。"
        )

    # ═══════════════════════════════════════
    # MD5 去重
    # ═══════════════════════════════════════

    def _md5(self, text: str) -> str:
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    def _exists(self, md5: str) -> bool:
        if not self.md5_path.exists():
            return False
        with open(self.md5_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip() == md5:
                    return True
        return False

    def _save_md5(self, md5: str):
        with open(self.md5_path, "a", encoding="utf-8") as f:
            f.write(md5 + "\n")


knowledge_base = KnowledgeBase()
