"""
Process every scene whose images are on disk and save the results.

    python process_scenes.py

Goes through all indexed scenes. For each one that has its front camera
images downloaded and is not yet in cache/, it runs the camera pipeline
(motion plus image-health measurements) and saves the result. Scenes that
are already cached are skipped, so it is safe to stop and rerun at any
time.

Afterwards it prints how many usable scenes there are per condition and
lists any scene rejected for bad CAN data.

Takes roughly ten seconds per scene.
"""

import time
from collections import Counter

from nuscenes.can_bus.can_bus_api import NuScenesCanBus

from camera_cache import camera_motion_for_scene, is_cached
from scene_io import DATA_ROOT, all_scene_names, images_on_disk, load_scene
from scene_sets import can_problem, condition_of


def main():
    names = all_scene_names()
    todo = [n for n in names if not is_cached(n) and images_on_disk(load_scene(n))]
    print(f"{len(names)} scenes indexed, {len(todo)} to process")

    started = time.time()
    for i, name in enumerate(todo, start=1):
        camera_motion_for_scene(name)
        per_scene = (time.time() - started) / i
        left = per_scene * (len(todo) - i) / 60
        print(f"  [{i}/{len(todo)}] {name}  ({per_scene:.0f} s per scene, about {left:.0f} min left)",
              flush=True)

    nusc_can = NuScenesCanBus(dataroot=DATA_ROOT)
    usable, rejected = Counter(), []
    for name in names:
        if not is_cached(name):
            continue
        scene = load_scene(name)
        problem = can_problem(nusc_can, scene)
        if problem is None:
            usable[condition_of(scene["description"])] += 1
        else:
            rejected.append((name, problem))

    print(f"\nusable scenes in cache: {sum(usable.values())}")
    for label in ("clear", "rain", "night", "night+rain", "wet"):
        print(f"  {label:11} {usable[label]:4d}")
    if rejected:
        print(f"\nrejected for CAN problems: {len(rejected)}")
        for name, problem in rejected:
            print(f"  {name}: {problem}")


if __name__ == "__main__":
    main()
