from __future__ import annotations

import ast
from pathlib import Path
import zipfile
from xml.etree import ElementTree


ROOT = Path(__file__).resolve().parents[1]
BUILDER = ROOT / "scripts" / "build_trkh_research_process_docx.py"
REPORT = ROOT / "docs" / "TRKH_5CLASS_RESEARCH_PROCESS_STATUS.docx"
WORD_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def test_hyperlink_builder_uses_ooxml_schema_order() -> None:
    module = ast.parse(BUILDER.read_text(encoding="utf-8"))
    function = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == "_add_hyperlink"
    )
    extend_call = next(
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "extend"
    )
    assert isinstance(extend_call.args[0], ast.List)
    assert [
        element.id
        for element in extend_call.args[0].elts
        if isinstance(element, ast.Name)
    ] == [
        "fonts",
        "color",
        "underline",
    ]


def test_generated_hyperlinks_follow_ooxml_schema_order() -> None:
    with zipfile.ZipFile(REPORT) as archive:
        root = ElementTree.fromstring(archive.read("word/document.xml"))
    run_properties = root.findall(
        f".//{{{WORD_NS}}}hyperlink/{{{WORD_NS}}}r/{{{WORD_NS}}}rPr"
    )
    assert run_properties
    for properties in run_properties:
        assert [_local_name(child.tag) for child in properties] == [
            "rFonts",
            "color",
            "u",
        ]


def test_generated_source_list_keeps_compact_spacing_after_numbering() -> None:
    with zipfile.ZipFile(REPORT) as archive:
        root = ElementTree.fromstring(archive.read("word/document.xml"))
    source_paragraphs = [
        paragraph
        for paragraph in root.findall(f".//{{{WORD_NS}}}p")
        if paragraph.find(f"{{{WORD_NS}}}hyperlink") is not None
    ]
    assert len(source_paragraphs) == 33
    for paragraph in source_paragraphs:
        spacing = paragraph.find(
            f"{{{WORD_NS}}}pPr/{{{WORD_NS}}}spacing"
        )
        assert spacing is not None
        assert spacing.get(f"{{{WORD_NS}}}after") == "20"
        assert spacing.get(f"{{{WORD_NS}}}line") == "252"
        assert spacing.get(f"{{{WORD_NS}}}lineRule") == "auto"
