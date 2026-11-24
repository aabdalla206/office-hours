"""Pull a Canvas course's materials into course/ using YOUR OWN Canvas access token.

Create a token yourself in Canvas: Account > Settings > "+ New Access Token". Keep it secret:
put it in an environment variable, never in code or git.

    export CANVAS_TOKEN=...            # or in Colab: os.environ["CANVAS_TOKEN"] = getpass()
    python canvas_ingest.py --base https://canvas.uw.edu --course 1918696

Writes:
    course/core/   syllabus + any page whose title mentions policy/course information  (shared prefix)
    course/notes/  every other wiki page, plus PDFs attached to modules                (retrieved)

Only fetches what your account can already see. Course materials belong to the instructor:
keep course/ out of git (see .gitignore) and get permission before sharing a deployment.
"""
import argparse
import os
import pathlib
import re

CORE_HINTS = re.compile(r"policy|course information|syllabus", re.I)


def html_to_text(html: str) -> str:
    html = re.sub(r"(?i)</(p|li|h\d|tr|div)>|<br\s*/?>", "\n", html or "")
    text = re.sub(r"<[^>]+>", " ", html)
    text = re.sub(r"&nbsp;", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n\s*\n+", "\n\n", text).strip()


def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")[:80] or "untitled"


class Canvas:
    def __init__(self, base, token):
        import httpx

        self.c = httpx.Client(base_url=base.rstrip("/") + "/api/v1",
                              headers={"Authorization": f"Bearer {token}"}, timeout=60, follow_redirects=True)

    def get_all(self, path, **params):
        params.setdefault("per_page", 100)
        url, out = path, []
        while url:
            r = self.c.get(url, params=params if url == path else None)
            r.raise_for_status()
            data = r.json()
            out.extend(data if isinstance(data, list) else [data])
            url = r.links.get("next", {}).get("url")
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="https://canvas.uw.edu")
    ap.add_argument("--course", required=True, help="numeric course id from the course URL")
    ap.add_argument("--out", default="course")
    args = ap.parse_args()

    token = os.environ.get("CANVAS_TOKEN")
    if not token:
        raise SystemExit("Set CANVAS_TOKEN to your own Canvas access token first.")
    cv = Canvas(args.base, token)
    core, notes = pathlib.Path(args.out, "core"), pathlib.Path(args.out, "notes")
    core.mkdir(parents=True, exist_ok=True)
    notes.mkdir(parents=True, exist_ok=True)

    course = cv.get_all(f"/courses/{args.course}", **{"include[]": "syllabus_body"})[0]
    (core / "00-syllabus.md").write_text(f"# {course['name']} syllabus\n\n{html_to_text(course.get('syllabus_body'))}")

    # Many courses hide the Pages list from students, so reach pages through module items.
    modules = cv.get_all(f"/courses/{args.course}/modules", **{"include[]": "items"})
    n_pages, seen = 0, set()
    for m in modules:
        for it in m.get("items", []):
            if it.get("type") != "Page" or not it.get("page_url") or it["page_url"] in seen:
                continue
            seen.add(it["page_url"])
            page = cv.get_all(f"/courses/{args.course}/pages/{it['page_url']}")[0]
            body = html_to_text(page.get("body"))
            if not body:
                continue
            dest = core if CORE_HINTS.search(page["title"]) else notes
            (dest / f"{slug(page['title'])}.md").write_text(f"# {page['title']}\n\n{body}")
            n_pages += 1

    n_pdfs = 0
    for m in modules:
        for it in m.get("items", []):
            if it.get("type") != "File" or not it.get("url"):
                continue
            meta = cv.c.get(it["url"]).json()
            if not meta.get("display_name", "").lower().endswith(".pdf"):
                continue
            (notes / meta["display_name"]).write_bytes(cv.c.get(meta["url"]).content)
            n_pdfs += 1

    print(f"syllabus + {n_pages} pages + {n_pdfs} PDFs -> {args.out}/ (core: {len(list(core.iterdir()))} files)")


if __name__ == "__main__":
    main()
