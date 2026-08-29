# ============================================================
# constants.py
# ============================================================
#
# ETHUKU CREATE PANNINOM? (Why does this file exist?)
# ━━━━━━━━━━━━━━━━━━━━━━
# Project full-la ellarum use panra "magic numbers" oru edathula
# irukkanum. Image size, camera names, BEV grid size, class names.
# Ithu illaina ovvoru file-layum 224, 400 nu type panni, oru naal
# oru edathula mattum maathina bug varum.
#
# MUNADI FILE ODA CONNECTION: (How does it connect to other files?)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━
# First file. Ithu yaaraiyum import pannala.
# Ithuku aprom varra ellathum (camera_loader, dataset, models) ithai
# import pannum.
#
# INNER OPERATIONS: (What happens inside?)
# ━━━━━━━━━━━━━━━━
# Vera onnum illa - just constant values define pannurom.
#
# INPUT / OUTPUT:
# ━━━━━━━━━━━━━━
# Input: illa. Output: module-level constants.
#
# EPADI USE AAGUM: (How is it used in the bigger system?)
# ━━━━━━━━━━━━━━━
# from data.scripts.constants import TARGET_W, TARGET_H, CAMERAS
#
# ============================================================

# --- Image size ---
# nuScenes original photo = 1600 x 900 pixels (romba periyathu).
# Athai 400 x 224 ku suruki-kirom -> 16x less pixels -> fast training.
# 224 & 400 ellam 32-la divide aagum (CNN downsample 32x pannum,
# so remainder illama irukkanum).
ORIGINAL_W = 1600
ORIGINAL_H = 900
TARGET_W = 400
TARGET_H = 224

# Resize scale factor. Camera K matrix-yum ithe scale-la maathanum,
# illaina 3D math thappa poidum.
SCALE_W = TARGET_W / ORIGINAL_W   # 0.25
SCALE_H = TARGET_H / ORIGINAL_H   # 0.2489

# --- 6 cameras (order MUKIYAM, always same order) ---
# Car mela 6 camera. Order fix panniten - model ithe order-la
# ethirpaakkum.
CAMERAS = [
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_FRONT_LEFT",
]
N_CAMERAS = len(CAMERAS)   # 6

# --- ImageNet normalization ---
# EfficientNet-B0 ImageNet photos-la pretrain aagirukku.
# Anga use panna mean/std ithu. Same normalization pannina thaan
# pretrained weights correct-a velai seiyum.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# --- BEV grid (top-down map) ---
# Car center-la nikkuthu. Suthi 100m x 100m area-va
# 200 x 200 cells-a pirikirom. Oru cell = 0.5m x 0.5m.
BEV_H = 200
BEV_W = 200
BEV_RESOLUTION = 0.5          # metres per cell
X_RANGE = (-50.0, 50.0)       # ego x = pinnadi(-)/munnadi(+), metres
Y_RANGE = (-50.0, 50.0)       # ego y = valathu(-)/idathu(+), metres
Z_RANGE = (-5.0, 3.0)         # ego z = keezha(-)/mela(+), metres
# nuScenes ego frame: x munnadi, y idathu pakkam, z mela. Ithu
# LSS-um dataset target-um ORE frame use pannurathunala thaan
# box position-um camera projection-um match aaguthu.

# --- LSS depth bins ---
# Camera-la depth theriyathu. So "2m to 50m varaikum 64 guesses"
# nu vachi, ovvoru guess-kum probability predict pannuvom.
D_MIN = 2.0
D_MAX = 50.0
N_DEPTHS = 64

# --- Channels ---
BACKBONE_OUT_CHANNELS = 64    # camera feature channels
BEV_OUT_CHANNELS = 128        # BEV encoder output channels

# --- 10 nuScenes detection classes ---
CLASSES = [
    "car",
    "truck",
    "bus",
    "trailer",
    "construction_vehicle",
    "pedestrian",
    "motorcycle",
    "bicycle",
    "traffic_cone",
    "barrier",
]
N_CLASSES = len(CLASSES)   # 10

# nuScenes-la category name romba long ("vehicle.car").
# Athai namma 10 class-ku map pannurom.
NUSCENES_NAME_MAP = {
    "vehicle.car": "car",
    "vehicle.truck": "truck",
    "vehicle.bus.bendy": "bus",
    "vehicle.bus.rigid": "bus",
    "vehicle.trailer": "trailer",
    "vehicle.construction": "construction_vehicle",
    "human.pedestrian.adult": "pedestrian",
    "human.pedestrian.child": "pedestrian",
    "human.pedestrian.construction_worker": "pedestrian",
    "human.pedestrian.police_officer": "pedestrian",
    "vehicle.motorcycle": "motorcycle",
    "vehicle.bicycle": "bicycle",
    "movable_object.trafficcone": "traffic_cone",
    "movable_object.barrier": "barrier",
}
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASSES)}

# --- Dataset paths / split ---
DATA_ROOT = "data/nuscenes-mini"
VERSION = "v1.0-mini"

# nuScenes OFFICIAL mini split use pannurom: 8 train scene, 2 val scene.
# Yaen official? Official NDS evaluation "mini_val" scene list-ai
# ethirpaakkum. Namma sonthama split panna, official metric run panna
# mudiyaathu (apo resume-la NDS number podave mudiyaathu).
# Same scene train+val la irukka koodathu (illaina model mugam
# paathurum = cheating) - official split athai already kavanichirukku.
VAL_SCENES = ["scene-0103", "scene-0916"]
