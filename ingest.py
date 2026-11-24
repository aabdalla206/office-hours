"""Turn a folder of course materials into (1) a shared course prefix and (2) a retrieval index.

Layout expected under course/:
    course/core/      syllabus, policies, schedule  -> always in the prompt (the shared prefix)
    course/notes/     lecture notes, readings       -> chunked and retrieved per question

Supports .pdf, .md, .txt. Run: python ingest.py course/ --out index/
"""
import argparse
import json
import pathlib

CHUNK_CHARS = 1200
OVERLAP = 200


def read_file(path: pathlib.Path) -> str:
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader
        return "\n".join((p.extract_text() or "") for p in PdfReader(str(path)).pages)
    return path.read_text(errors="ignore")


def read_dir(d: pathlib.Path):
    exts = {".pdf", ".md", ".txt"}
    return [(p, read_file(p)) for p in sorted(d.rglob("*")) if p.suffix.lower() in exts]


def chunk(text: str, source: str):
    text = " ".join(text.split())
    out, i = [], 0
    while i < len(text):
        out.append({"source": source, "text": text[i:i + CHUNK_CHARS]})
        i += CHUNK_CHARS - OVERLAP
    return out


def unpack_export(export_path, course_dir):
    """Write a canvas_export.js file into course/core and course/notes, same split as canvas_ingest.py."""
    from canvas_ingest import CORE_HINTS, html_to_text, slug

    data = json.loads(pathlib.Path(export_path).read_text())
    core, notes = pathlib.Path(course_dir, "core"), pathlib.Path(course_dir, "notes")
    core.mkdir(parents=True, exist_ok=True)
    notes.mkdir(parents=True, exist_ok=True)
    (core / "00-syllabus.md").write_text(f"# {data['course']} syllabus\n\n{html_to_text(data['syllabus_html'])}")
    for p in data["pages"]:
        dest = core if CORE_HINTS.search(p["title"]) else notes
        (dest / f"{slug(p['title'])}.md").write_text(f"# {p['title']}\n\n{html_to_text(p['html'])}")
    outline = "\n".join(
        f"- {m['name']}: " + "; ".join(i["title"] for i in m["items"] if "Recorded" not in i["title"])
        for m in data["modules"]
    )
    (core / "01-course-outline.md").write_text(f"# Weekly course outline (modules)\n\n{outline}")
    print(f"unpacked {data['course']}: syllabus + {len(data['pages'])} pages + outline -> {course_dir}/")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("course_dir", nargs="?", default="course")
    ap.add_argument("--from-export", help="course_export.json made by canvas_export.js; unpacked into course_dir first")
    ap.add_argument("--out", default="index")
    ap.add_argument("--prefix-chars", type=int, default=48000,
                    help="cap on the shared prefix (~4 chars per token; keep it under vLLM's --max-model-len)")
    args = ap.parse_args()

    if args.from_export:
        unpack_export(args.from_export, args.course_dir)

    root, out = pathlib.Path(args.course_dir), pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    core = read_dir(root / "core")
    prefix = "\n\n".join(f"### {p.name}\n{t.strip()}" for p, t in core)[: args.prefix_chars]
    (out / "prefix.txt").write_text(prefix)

    chunks = [c for p, t in read_dir(root / "notes") for c in chunk(t, p.name)]
    (out / "chunks.json").write_text(json.dumps(chunks))

    if chunks:
        import faiss
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer("all-MiniLM-L6-v2")
        emb = model.encode([c["text"] for c in chunks], normalize_embeddings=True, show_progress_bar=True)
        index = faiss.IndexFlatIP(emb.shape[1])
        index.add(emb)
        faiss.write_index(index, str(out / "notes.faiss"))

    print(f"prefix: {len(prefix)} chars from {len(core)} core files; {len(chunks)} note chunks")


if __name__ == "__main__":
    main()
