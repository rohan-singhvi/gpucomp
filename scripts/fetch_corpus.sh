#!/usr/bin/env bash
# Downloads the Silesia and Canterbury corpora into testdata/corpus/ (git-ignored).
# Usage: scripts/fetch_corpus.sh [silesia] [canterbury]   (default: both)
#   then: gpucomp bench --corpus testdata/corpus/silesia
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
dest="$root/testdata/corpus"
mkdir -p "$dest"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

fetch() { # name url archive-type
  local name=$1 url=$2 kind=$3
  if [ -d "$dest/$name" ] && [ -n "$(ls -A "$dest/$name")" ]; then
    echo "$name: already present"; return
  fi
  echo "$name: downloading $url"
  curl -fL --retry 3 -o "$tmp/$name.$kind" "$url"
  mkdir -p "$dest/$name"
  case $kind in
    zip) unzip -q -o "$tmp/$name.$kind" -d "$dest/$name" ;;
    tar.gz) tar -xzf "$tmp/$name.$kind" -C "$dest/$name" ;;
  esac
  echo "$name: $(ls "$dest/$name" | wc -l | tr -d ' ') files"
}

for name in "${@:-silesia canterbury}"; do
  for n in $name; do
    case $n in
      silesia) fetch silesia https://sun.aei.polsl.pl/~sdeor/corpus/silesia.zip zip ;;
      canterbury) fetch canterbury https://corpus.canterbury.ac.nz/resources/cantrbry.tar.gz tar.gz ;;
      *) echo "unknown corpus: $n" >&2; exit 1 ;;
    esac
  done
done
