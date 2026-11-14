#!/usr/bin/env bash
# Download one part of the full nuScenes dataset, keeping only the front camera.
#
#   bash fetch_part.sh 1        # part 1 of 10
#
# A part is about 30 GB because it holds every sensor. We need about a
# tenth of that. So the download is unpacked as it arrives and everything
# except the two front camera folders is thrown away on the fly. The 30 GB
# archive itself never touches the disk. Expect about 3.5 GB of images.
#
# The catch: a download that is unpacked on the fly cannot be resumed. If
# the connection drops or hangs, this script notices and starts the part
# over, up to three times. Files that already arrived are simply written
# again.
#
# When it finishes, run:   python process_scenes.py

set -euo pipefail

if [ $# -ne 1 ] || ! [[ "$1" =~ ^([1-9]|10)$ ]]; then
  echo "usage: bash fetch_part.sh <part number 1 to 10>"
  exit 1
fi

PART=$(printf "%02d" "$1")
URL="https://motional-nuscenes.s3.amazonaws.com/public/v1.0/v1.0-trainval${PART}_blobs.tgz"
# FETCH_URL and FETCH_NEED_GB let a test point this script at a small local
# file and skip the free-space requirement.
URL="${FETCH_URL:-$URL}"
DEST="data/nuscenes"
NEED_GB="${FETCH_NEED_GB:-6}"

mkdir -p "$DEST"
FREE_GB=$(df -Pk "$DEST" | awk 'NR==2 {print int($4/1024/1024)}')
if [ "$FREE_GB" -lt "$NEED_GB" ]; then
  echo "Only ${FREE_GB} GB free. Need at least ${NEED_GB} GB. Free some space first."
  exit 1
fi

count_images() {
  # The folders do not exist before the first download, hence "|| true".
  { find "$DEST/samples/CAM_FRONT" "$DEST/sweeps/CAM_FRONT" -name '*.jpg' 2>/dev/null || true; } \
    | wc -l | tr -d ' '
}
BEFORE=$(count_images)

echo "Part ${PART}: ${FREE_GB} GB free, ${BEFORE} front camera images on disk."
echo "Downloading and unpacking front camera only. This takes a while..."

# macOS ships bsdtar and Linux ships GNU tar. They spell "only these files"
# differently. The leading * lets the pattern match whether or not the
# paths inside the archive start with "./".
if tar --version 2>/dev/null | grep -qi bsdtar; then
  KEEP=(--include '*samples/CAM_FRONT/*' --include '*sweeps/CAM_FRONT/*')
else
  KEEP=(--wildcards '*samples/CAM_FRONT/*' '*sweeps/CAM_FRONT/*')
fi

# --speed-limit and --speed-time make curl give up if less than 10 KB per
# second arrives for a full minute. Without them a connection that goes
# silent would hang forever with no error, which happened once on part 10.
fetch_once() {
  curl -f -L --progress-bar --speed-limit 10240 --speed-time 60 "$URL" \
    | tar -xzf - -C "$DEST" "${KEEP[@]}"
}

ATTEMPT=1
until fetch_once; do
  if [ "$ATTEMPT" -ge 3 ]; then
    echo "Download failed 3 times. Check the connection and run this again."
    exit 1
  fi
  ATTEMPT=$((ATTEMPT + 1))
  echo "Download was cut off. Starting part ${PART} over (attempt ${ATTEMPT} of 3)..."
  sleep 5
done

AFTER=$(count_images)
echo
echo "Done. Front camera images on disk: ${BEFORE} before, ${AFTER} now."
echo "Disk used by front camera images:"
du -sh "$DEST/samples/CAM_FRONT" "$DEST/sweeps/CAM_FRONT"
if [ "$AFTER" -eq "$BEFORE" ]; then
  echo "WARNING: no new images arrived. Check the output above for errors."
  exit 1
fi
echo "Next:  python process_scenes.py"
