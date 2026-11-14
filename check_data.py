# Quick sanity check that the dataset and CAN expansion are readable.
from nuscenes.nuscenes import NuScenes
from nuscenes.can_bus.can_bus_api import NuScenesCanBus

ROOT = "data/nuscenes"

nusc = NuScenes(version="v1.0-mini", dataroot=ROOT, verbose=True)
nusc.list_scenes()

can = NuScenesCanBus(dataroot=ROOT)
name = nusc.scene[0]["name"]
pose = can.get_messages(name, "pose")

print(f"\n{name}: {len(pose)} pose messages")
print("fields:", list(pose[0].keys()))
speeds = [m["vel"][0] for m in pose]
print(f"forward speed range: {min(speeds):.1f} to {max(speeds):.1f} m/s")