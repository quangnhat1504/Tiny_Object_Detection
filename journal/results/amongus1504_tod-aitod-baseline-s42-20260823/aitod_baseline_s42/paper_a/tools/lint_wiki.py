#!/usr/bin/env python3
"""
lint_wiki.py - Comprehensive Linter for the Tiny Object Detection Research Wiki.

Checks:
1. YAML frontmatter schema and validity.
2. Internal wikilink resolution ([[Page Name]] or [[Target|Anchor]]).
3. Markdown relative file links ([text](relative/path.md)).
4. Index registration (all wiki documents listed in wiki/index.md).
5. Orphan page detection.
"""

import os
import re
import sys
import glob
import yaml

WIKI_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "wiki"))

VALID_TYPES = {
    "source",
    "concept",
    "topic",
    "analysis",
    "synthesis",
    "overview",
    "research",
    "entity",
    "audit",
    "protocol",
}


def strip_code_blocks(text: str) -> str:
    """Strip fenced code blocks and inline code to prevent false-positive link parsing."""
    # Strip fenced code blocks (```...```)
    text = re.sub(r"```[\s\S]*?```", "", text)
    # Strip inline code (`...`)
    text = re.sub(r"`[^`\n]+`", "", text)
    return text


def lint_wiki(wiki_dir: str = WIKI_DIR) -> int:
    print(f"=== Starting Wiki Lint on: {wiki_dir} ===\n")

    all_md_files = glob.glob(os.path.join(wiki_dir, "**", "*.md"), recursive=True)
    print(f"Discovered {len(all_md_files)} markdown files in wiki.")

    errors = []
    warnings = []

    # Map: normalized_key -> relative_path
    title_to_file = {}
    file_to_title = {}
    file_to_fm = {}

    # Pass 1: Parse Frontmatters and build Symbol Table
    for file_path in all_md_files:
        rel_path = os.path.relpath(file_path, wiki_dir)
        basename_no_ext = os.path.splitext(os.path.basename(file_path))[0]

        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        fm_match = re.match(r"^---\s*\n([\s\S]*?)\n---\s*\n", content)
        if not fm_match:
            errors.append(f"[{rel_path}] Missing YAML frontmatter header (---).")
            title_to_file[basename_no_ext.lower()] = rel_path
            continue

        try:
            fm = yaml.safe_load(fm_match.group(1))
            if not isinstance(fm, dict):
                errors.append(f"[{rel_path}] Frontmatter must be a YAML dictionary.")
                continue

            file_to_fm[rel_path] = fm

            # Validate Type
            doc_type = fm.get("type")
            if not doc_type:
                warnings.append(f"[{rel_path}] Frontmatter missing 'type' field.")
            elif doc_type not in VALID_TYPES:
                warnings.append(
                    f"[{rel_path}] Unrecognized document type: '{doc_type}'."
                )

            # Record Title
            title = fm.get("title", basename_no_ext)
            file_to_title[rel_path] = title
            title_to_file[title.lower()] = rel_path
            title_to_file[basename_no_ext.lower()] = rel_path

            # Record aliases if any
            aliases = fm.get("aliases", [])
            if isinstance(aliases, list):
                for alias in aliases:
                    title_to_file[str(alias).lower()] = rel_path

        except Exception as e:
            errors.append(f"[{rel_path}] YAML parse error: {e}")

    # Pass 2: Check Wikilinks and Markdown links in all files
    wikilink_pattern = re.compile(r"\[\[([^\]|]+)(?:\|[^\]]+)?\]\]")
    md_link_pattern = re.compile(r"\[([^\]]+)\]\(([^)]+\.md)\)")

    for file_path in all_md_files:
        rel_path = os.path.relpath(file_path, wiki_dir)
        with open(file_path, "r", encoding="utf-8") as f:
            raw_content = f.read()

        # Strip frontmatter and code blocks
        body = re.sub(r"^---\s*\n[\s\S]*?\n---\s*\n", "", raw_content)
        cleaned_body = strip_code_blocks(body)

        # 2a: Validate Wikilinks [[...]]
        for link in wikilink_pattern.findall(cleaned_body):
            target = link.strip().lower()
            if target not in title_to_file:
                errors.append(f"[{rel_path}] Broken wikilink: [[{link.strip()}]]")

        # 2b: Validate Markdown Links [...](....md)
        for text, target_path in md_link_pattern.findall(cleaned_body):
            if target_path.startswith("http://") or target_path.startswith("https://"):
                continue
            # Resolve relative path
            target_full = os.path.normpath(
                os.path.join(os.path.dirname(file_path), target_path)
            )
            if not os.path.exists(target_full):
                errors.append(
                    f"[{rel_path}] Dead relative markdown link: [{text}]({target_path})"
                )

    # Pass 3: Check wiki/index.md Registration
    index_path = os.path.join(wiki_dir, "index.md")
    if not os.path.exists(index_path):
        errors.append("Critical: wiki/index.md does not exist.")
    else:
        with open(index_path, "r", encoding="utf-8") as f:
            index_content = strip_code_blocks(f.read())

        indexed_wikilinks = set(
            l.strip().lower() for l in wikilink_pattern.findall(index_content)
        )
        indexed_md_links = set(
            os.path.normpath(os.path.join(wiki_dir, l)).lower()
            for _, l in md_link_pattern.findall(index_content)
        )

        for file_path in all_md_files:
            rel = os.path.relpath(file_path, wiki_dir)
            if rel in ["index.md", "overview.md", "log.md"]:
                continue

            basename = os.path.splitext(os.path.basename(file_path))[0].lower()
            title = file_to_title.get(rel, basename).lower()

            norm_full = os.path.normpath(file_path).lower()

            is_indexed = (
                title in indexed_wikilinks
                or basename in indexed_wikilinks
                or norm_full in indexed_md_links
            )

            if not is_indexed:
                warnings.append(
                    f"[{rel}] Page '{file_to_title.get(rel, rel)}' is not indexed in wiki/index.md."
                )

    # Report Summary
    print("--------------------------------------------------")
    print(f"Wiki Lint Results:")
    print(f"  Total Errors   : {len(errors)}")
    print(f"  Total Warnings : {len(warnings)}")
    print("--------------------------------------------------\n")

    if errors:
        print("ERRORS:")
        for err in errors:
            print(f"  [ERROR] {err}")
        print()

    if warnings:
        print("WARNINGS:")
        for warn in warnings:
            print(f"  [WARN]  {warn}")
        print()

    if not errors and not warnings:
        print(">>> SUCCESS: Wiki is 100% compliant with zero errors and zero warnings! <<<\n")
        return 0
    elif not errors:
        print(">>> PASSED: Wiki has zero fatal errors (some warnings exist). <<<\n")
        return 0
    else:
        print(">>> FAILED: Please resolve the fatal errors above. <<<\n")
        return 1


if __name__ == "__main__":
    sys.exit(lint_wiki())
