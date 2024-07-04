import cv2
import numpy as np
import itertools
import kalman

def match(ref_kpts, ref_descs, frame_kpts, frame_descs):
  ## match descriptors and sort them in the order of their distance
  matches = bf.match(ref_descs, frame_descs)
  dmatches = sorted(matches, key = lambda x:x.distance)

  ## extract the matched keypoints
  src_pts  = np.float32([ref_kpts[m.queryIdx].pt for m in dmatches]).reshape(-1,1,2)
  dst_pts  = np.float32([frame_kpts[m.trainIdx].pt for m in dmatches]).reshape(-1,1,2)

  ## find homography matrix and do perspective transform
  M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,10.0)

  u, _, vh = np.linalg.svd(M[0:2, 0:2])
  R = u @ vh
  return math.atan2(R[1,0], R[0,0])

import argparse
arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('path')
args = arg_parser.parse_args()

## Create ORB object and BF object(using HAMMING)
orb = cv2.ORB_create()
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

cap = cv2.VideoCapture(args.path)
if not cap.isOpened():
  print("Cannot open video")
  exit(1)

import math

fil = kalman.LinearKalmanFilter(
  F = np.array([
    [1, math.nan],
    [0, math.nan]
  ]),
  H = np.array([
    [1, 0]
  ]),
  Q = np.array([
    [1e-7, 0],
    [0, 1]
  ]),
  R = np.array([
    [1]
  ]),
  x0 = np.array([
    [0],
    [0]
  ]),
  P0 = np.array([
    [1e-8, 0],
    [0, 1e-8]
  ])
)
def signed_angle_dist(tgt, src):
  return (tgt - src + math.pi) % (2*math.pi) - math.pi

def set_deltat(fil: kalman.LinearKalmanFilter, dt: float):
  fil.F[0,1] = dt
  fil.F[1, 1] = 0.3 ** dt

roi = None
ref_kpts = None
ref_descs = None

cap.set(cv2.CAP_PROP_POS_FRAMES, 1200)
old_t = None

clahe = cv2.createCLAHE(2.0, (8,8))

with open('log.csv', 'w') as f:
  try:
    while cv2.pollKey() != ord('q'):
      ret, frame = cap.read()
      if not ret:
        break

      frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
      frame = cv2.GaussianBlur(frame, (0,0), 5)

      if roi is None:
        roi = cv2.selectROI('frame', frame)

      frame = frame[roi[1]:(roi[1]+roi[3]), roi[0]:(roi[0]+roi[2])]
      frame = clahe.apply(frame)

      if ref_kpts is None or ref_descs is None:
        ref_kpts, ref_descs = orb.detectAndCompute(frame, None)
        cv2.imshow('frame', cv2.drawKeypoints(frame, ref_kpts, None))
        cv2.waitKey()

      try:
        frame_kpts, frame_descs = orb.detectAndCompute(frame, None)
        angle = match(ref_kpts, ref_descs, frame_kpts, frame_descs)
      except Exception as e:
        print(e)
        continue
      
      t = cap.get(cv2.CAP_PROP_POS_MSEC)
      if old_t is not None:
        delta_t = (t - old_t) / 1000
        set_deltat(fil, delta_t)
        fil.predict() 
        fil.update(np.array([angle]), signed_angle_dist)
      old_t = t

      angle = fil.get_state()[0,0]
      print(angle, "\n", fil.get_covariance())

      f.write(f"{cap.get(cv2.CAP_PROP_POS_MSEC)}, {angle}\n")
      
      frame = cv2.drawKeypoints(frame, frame_kpts, None)
      rows, cols, _ = frame.shape
      M = cv2.getRotationMatrix2D(((cols-1) / 2.0, (rows-1) / 2.0), math.degrees(angle), 1)
      frame = cv2.warpAffine(frame, M, (cols, rows))

      cv2.imshow('frame', frame)
  finally:
    cap.release()