import cv2
import numpy as np

def match(ref_kpts, ref_descs, frame_kpts, frame_descs):
  ## match descriptors and sort them in the order of their distance
  matches = bf.match(ref_descs, frame_descs)
  dmatches = sorted(matches, key = lambda x:x.distance)

  ## extract the matched keypoints
  src_pts  = np.float32([ref_kpts[m.queryIdx].pt for m in dmatches]).reshape(-1,1,2)
  dst_pts  = np.float32([frame_kpts[m.trainIdx].pt for m in dmatches]).reshape(-1,1,2)

  ## find homography matrix and do perspective transform
  M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)

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

ang = 0
samples = []

roi = None
ref_kpts = None
ref_descs = None

cap.set(cv2.CAP_PROP_POS_FRAMES, 1200)

try:
  while cv2.pollKey() != ord('q'):
    ret, frame = cap.read()
    if not ret:
      break

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.GaussianBlur(frame, (0,0), 3)
    # ret, frame = cv2.threshold(frame, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)


    if roi is None:
      roi = cv2.selectROI('frame', frame)

    frame = frame[roi[1]:(roi[1]+roi[3]), roi[0]:(roi[0]+roi[2])]

    if ref_kpts is None or ref_descs is None:
      print("ref")
      ref_kpts, ref_descs = orb.detectAndCompute(frame,None)
      cv2.imshow('frame', cv2.drawKeypoints(frame, ref_kpts, None))
      cv2.waitKey()


    rows, cols = frame.shape

    frame_kpts, frame_descs = orb.detectAndCompute(frame, None)

    wait = False
    try:
      angle = match(ref_kpts, ref_descs, frame_kpts, frame_descs)
    except:
      print("Frame skipped!")
      continue

    frame = cv2.drawKeypoints(frame, frame_kpts, None)

    ang = angle

    M = cv2.getRotationMatrix2D(((cols-1) / 2.0, (rows-1) / 2.0), math.degrees(ang), 1)
    frame = cv2.warpAffine(frame, M, (cols, rows))

    cv2.imshow('frame', frame)
finally:
  cap.release()