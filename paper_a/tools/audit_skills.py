"""
Skill Integrity and Schema Linter.
Validates all global and workspace SKILL.md files against official standards.
"""
from __future__ import annotations
import os
import re
from pathlib import Path

skill_roots = [
    Path(r"C:\Users\ADMIN\.gemini\config\skills"),
    Path(r"c:\Users\ADMIN\_Project\tiny-object-detection\.agents\skills"),
]

errors = []
warnings = []
total_skills = 0
passed_skills = []

for root in skill_roots:
    if not root.exists():
        continue
    for skill_file in root.glob("*/SKILL.md"):
        total_skills += 1
        content = skill_file.read_text(encoding="utf-8")
        
        if not content.startswith("---"):
            errors.append(f"{skill_file}: Missing leading '---' in YAML frontmatter")
            continue
        
        parts = content.split("---", 2)
        if len(parts) < 3:
            errors.append(f"{skill_file}: Malformed YAML frontmatter (missing closing '---')")
            continue
            
        frontmatter = parts[1]
        name_match = re.search(r"^name:\s*(.+)$", frontmatter, re.MULTILINE)
        desc_match = re.search(r"^description:\s*(.+)$", frontmatter, re.MULTILINE)
        
        if not name_match:
            errors.append(f"{skill_file}: Missing 'name:' field in YAML frontmatter")
        if not desc_match:
            errors.append(f"{skill_file}: Missing 'description:' field in YAML frontmatter")
            
        folder_name = skill_file.parent.name
        if name_match:
            declared_name = name_match.group(1).strip().strip('"').strip("'")
            if folder_name != declared_name:
                warnings.append(f"{skill_file}: Directory name '{folder_name}' differs from frontmatter name '{declared_name}'")
        
        if not errors:
            passed_skills.append(folder_name)

print("==================================================")
print("             SKILL AUDIT REPORT                   ")
print("==================================================")
print(f"Total Skills Inspected : {total_skills}")
print(f"Total Errors Found     : {len(errors)}")
print(f"Total Warnings Found   : {len(warnings)}")
print("--------------------------------------------------")

if errors:
    print("\n[!] Errors:")
    for e in errors:
        print(f"  ❌ {e}")

if warnings:
    print("\n[!] Warnings:")
    for w in warnings:
        print(f"  ⚠️ {w}")

if not errors and not warnings:
    print("\n>>> SUCCESS: All skills are 100% compliant with zero errors and zero warnings! <<<\n")
    print(f"Verified Skills ({len(passed_skills)}):")
    for s in sorted(set(passed_skills)):
        print(f"  [OK] {s}")
