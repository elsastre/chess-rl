"""Diagnóstico de cámaras: prueba índices, backends y formatos e informa qué funciona.
Uso, desde la carpeta del repo: python chesscam/camtest.py"""
import os
os.environ.setdefault("OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS", "0")
import cv2

print("OpenCV", cv2.__version__)
info = cv2.getBuildInformation()
for line in info.splitlines():
    if any(k in line for k in ("DirectShow", "Media Foundation", "Video I/O")):
        print("  ", line.strip())

import sys
if sys.platform.startswith("win"):
    backends = [("dshow", cv2.CAP_DSHOW), ("msmf", cv2.CAP_MSMF)]
elif sys.platform == "darwin":
    backends = [("avf", cv2.CAP_AVFOUNDATION), ("any", cv2.CAP_ANY)]
else:
    backends = [("v4l2", cv2.CAP_V4L2), ("any", cv2.CAP_ANY)]
found = []
for idx in range(4):
    for bname, b in backends:
        for mjpg in (False, True):
            tag = f"--cam {idx} --backend {bname}" + (" --mjpg" if mjpg else "")
            print(f"Probando {tag} ...", end=" ", flush=True)
            cap = cv2.VideoCapture(idx, b)
            if not cap.isOpened():
                print("no abre")
                cap.release()
                continue
            if mjpg:
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            ok = False
            for _ in range(5):            # algunas cámaras fallan el primer frame
                ok, frame = cap.read()
                if ok:
                    break
            cap.release()
            if ok:
                print(f"OK {frame.shape[1]}x{frame.shape[0]}")
                found.append(tag)
            else:
                print("abre pero sin imagen")

print("\nFunciona:" if found else "\nNinguna combinación entregó imagen.")
for t in found:
    print("   python chesscam/chesscam.py", t)
