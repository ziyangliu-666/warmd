#!/usr/bin/env bash
# Flag common machine-writing habits in docs (see docs/STYLE.md). Prints file:line:match.
cd "$(dirname "$0")/.."
files=$(ls README.md CONTRIBUTING.md SECURITY.md CHANGELOG.md docs/*.md 2>/dev/null | grep -v STYLE.md)
words='seamless|robust|powerful|leverag|crucial|pivotal|delve|unlock|empower|effortless|cutting-edge|game-chang|best-in-class|blazing|comprehensive|ensuring|showcas|underscor|testament|tapestry|meticulous'
grep -nE "—" $files | sed 's/^/em-dash: /'
grep -niE "\b($words)" $files | sed 's/^/word: /'
grep -nE "^[-*] \*\*[^*]+\*\*:?" $files | sed 's/^/bold-label list: /'
grep -nE "^#+ ([A-Z][a-z]+ ){1,}[A-Z][a-z]+" $files | grep -vE "^#+ [A-Z][a-z]+$" | sed 's/^/title case?: /'
grep -nE "✅|❌|🚀|✨|⚡" $files | sed 's/^/emoji: /'
exit 0
