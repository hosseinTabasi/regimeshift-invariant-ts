#!/usr/bin/env bash
# NASA C-MAPSS turbofan data (FD002/FD004: six operating conditions).
# Primary: NASA PCoE archive on S3; fallback: public GitHub mirror of the txt files.
set -e
ROOT=${INDUSTRIAL_TS_DATA:-data/raw}; mkdir -p "$ROOT/CMAPSS"; cd "$ROOT/CMAPSS"
MIRROR=https://raw.githubusercontent.com/LahiruJayasinghe/RUL-Net/master/CMAPSSData
for s in FD002 FD004; do
  [ -f train_$s.txt ] || curl -fL -o train_$s.txt "$MIRROR/train_$s.txt"
done
ls -l train_FD00*.txt; echo "export INDUSTRIAL_TS_DATA=$ROOT"
