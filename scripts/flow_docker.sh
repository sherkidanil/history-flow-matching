#!/bin/sh
set -eu

image="${OPM_DOCKER_IMAGE:-openporousmedia/opmreleases:latest}"
workdir=$(pwd -P)

if [ "${1:-}" = "--version" ]; then
  exec docker run --rm "$image" flow --version
fi

if [ "${OPM_KEEP_SIMULATOR_FILES:-0}" = "1" ]; then
  exec docker run --rm \
    --user "$(id -u):$(id -g)" \
    --volume "$workdir:/shared_host" \
    --workdir /shared_host \
    "$image" flow "$@"
fi

if [ "$#" -lt 1 ]; then
  echo "usage: flow_docker.sh DECK.DATA [flow options]" >&2
  exit 64
fi

deck_name=$(basename "$1")
case_name=${deck_name%.*}
container=$(docker create "$image" sleep infinity)
container_workdir=/tmp/fmgeo
cleanup() {
  docker rm --force "$container" >/dev/null 2>&1 || true
}
trap cleanup EXIT HUP INT TERM

docker start "$container" >/dev/null
docker exec "$container" mkdir -p "$container_workdir"
docker cp "$workdir/." "$container:$container_workdir/"

set +e
docker exec --workdir "$container_workdir" "$container" flow "$@"
status=$?
set -e

for extension in SMSPEC UNSMRY RSM; do
  output="$container_workdir/$case_name.$extension"
  if docker exec "$container" test -f "$output"; then
    docker cp "$container:$output" "$workdir/" >/dev/null
  fi
done

exit "$status"
