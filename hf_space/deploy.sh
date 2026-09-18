#!/usr/bin/env bash
# ============================================================
# deploy.sh - HF Space-ku push pannurathu
# ============================================================
# HF Space oru thani git repo. Anga model code irukkanum, aana
# adhai repo-la rendu thadava vachaa (inga + hf_space/la) rendum
# vera vera aagi poidum - onnu maathina innonu pazhaysa nikkum.
#
# So hf_space/-la app.py mattum. Meedhi code push panra podhu
# inga irundhu copy aagum.
#
# Odurathu:  bash hf_space/deploy.sh
# ============================================================
set -e

SPACE_URL="git@hf.co:spaces/dheepakkaran/multi-camera-bev"

# ROOT-ai MUNNADI kandupidikkanum - cd panna apram intha relative
# path velai seiyaadhu
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK=$(mktemp -d)

echo "cloning Space..."
git clone -q "$SPACE_URL" "$WORK"
cd "$WORK" && git lfs install --local >/dev/null

echo "copying app + assets..."
cp "$ROOT/hf_space/app.py" "$ROOT/hf_space/requirements.txt" \
   "$ROOT/hf_space/README.md" .
mkdir -p assets && cp "$ROOT/hf_space/assets/"* assets/

echo "copying model code from the main repo..."
for pkg in models training data visualization; do
    mkdir -p "$pkg"
    rsync -a --include='*/' --include='*.py' --exclude='*' \
          "$ROOT/$pkg/" "$pkg/"
done

cat > .gitattributes <<'LFS'
*.pth filter=lfs diff=lfs merge=lfs -text
*.npz filter=lfs diff=lfs merge=lfs -text
LFS
printf '__pycache__/\n*.pyc\n' > .gitignore
find . -name __pycache__ -type d -not -path './.git/*' -exec rm -rf {} + 2>/dev/null || true

git add -A
if git diff --cached --quiet; then
    echo "maatram illa"
else
    git commit -qm "${1:-Update from main repo}"
    git push -q origin main
    echo "pushed to $SPACE_URL"
fi
rm -rf "$WORK"
