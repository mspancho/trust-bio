#!/usr/bin/env bash
# Build the TRUST-BIO paper with Tectonic (NOT pdflatex -- system TeX Live is
# incomplete and tlmgr needs sudo; tectonic fetches its own packages).
# First run may 429 on the font CDN; retry. Mirrors pisces-rct/docs/v2paper/build.sh.
set -uo pipefail
cd "$(dirname "$0")"
export PATH="$HOME/.local/bin:$PATH"
tectonic -X compile main.tex || exit 1

# Undefined CITATIONS do not say "undefined" in the log (natbib reports them
# differently); grep the RENDERED PDF for '(?)'.
n=$(pdftotext main.pdf - 2>/dev/null | grep -c '(?)')
[ "$n" -ne 0 ] && { echo "FAIL: $n unresolved '(?)' in the PDF"; exit 1; }

# Placeholders must be gone before camera-ready.
if grep -n '\\placeholder{' main.tex | grep -v '^\s*%' | grep -v newcommand; then
  echo "WARNING: \placeholder still present (listed above)"
fi
pages=$(pdfinfo main.pdf | awk '/^Pages/{print $2}')
refpage=$(pdftotext main.pdf - | awk 'BEGIN{p=1}/\f/{p++}/References/{print p; exit}')
echo "OK: ${pages} pages; References begin on page ${refpage} (content limit: 8)"
