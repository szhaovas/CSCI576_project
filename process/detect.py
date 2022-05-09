from cv2 import INTER_AREA
import numpy as np
import cv2 as cv
import os

import time

def rec_ntimes(func, source, ntimes):
    if ntimes == 0:
        return source
    else:
        return rec_ntimes(func, func(source), ntimes-1)

class Detect():

  def __init__(self, media, logo_downscale=1):
    self.media = media
    # self.raw_data = media.getRGBData()
    # self.total_frames = int(self.raw_data.shape[0] / media.frame_size)
    self.frame_size = media.frame_size
    self.channel_size = media.channel_size
    self.logos = media.logos
    self.logo_downscale = logo_downscale
    self.orb = cv.ORB_create(nfeatures=1000)

  # def detect_logos_in_scene(self, scene_data, scene_length):
  def detect_logos_in_scene(self, scene_data, out_f):
    # Logo detection only for relevant scenes used in EDIT (under replaceAds)
    # scene_data > raw RGB video of scene
    # scene_length > # of frames in scene
    outputData = np.copy(scene_data)
    ad_detected = False
    ad_index = 0

    # Loop for every logo
    start_byte = out_f.tell()
    frame_modified = [False] * len(scene_data)
    for index, logo_file in enumerate(self.logos):
      self.cc = index
      # Read RGB logo
      print(f'      Searching for logo: {os.path.basename(logo_file)}')
      rgb_file = open(logo_file, 'rb')
      raw_data = np.fromfile(rgb_file, dtype=np.uint8, count=-1)
      logo = self._convert_bgr(raw_data)
      logo = rec_ntimes(cv.pyrDown, cv.cvtColor(logo, cv.COLOR_BGR2GRAY), \
          self.logo_downscale)

      last_logo = (index == len(self.logos)-1)

      # Uncomment to verify that image converts to BGR correctly
      # cv.imshow("test", logo)
      # cv.waitKey()

      # kp1, des1 = self.getFeatures(logo)
      kp1, des1 = self.orb.detectAndCompute(logo, None)

      for ifr, frame in enumerate(scene_data):
        bgr_frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
        processed_frame = self.process_frame(des1, kp1, bgr_frame, logo)
        if processed_frame is not None:
          ad_detected = True
          ad_index = index
          out_f.write(self._convert_rgb(processed_frame))
          frame_modified[ifr] = True
        # avoid copying multiple times for each logo
        # don't copy if frame already marked bounding box
        elif last_logo and not frame_modified[ifr]:
          out_f.write(self._convert_rgb(bgr_frame))

      # for frame in range(scene_length):
      #   start = frame * self.frame_size
      #   end = start + self.frame_size
      #   bgr_data = self._convert_bgr(outputData[start:end])
      #   processed_frame = self.process_frame(des1, kp1, bgr_data, logo)
      #   if not np.array_equal(processed_frame, bgr_data):
      #     # If there is anything drawn, ad is detected and reported to the editor
      #     # to edit in the corresponding ad
      #     ad_detected = True
      #     ad_index = index
      #   outputData[start:end] = self._convert_rgb(processed_frame)

      # done scanning 1 logo, return byte pointer to scan next logo
      if not last_logo:
          out_f.seek(start_byte)
    # return bytes(bytearray(outputData.tolist())), ad_detected, ad_index
    return ad_detected, ad_index

  def detect_logos(self):
    # Handle overall logo detection flow

    # Ouput MP4 video for now, easier debugging; TODO: replace and integrate w/ rgb
    cwd = os.getcwd()
    test = cv.imread(f'{cwd}\dataset\Brand Images\starbtest5.jpg')
    logo = cv.imread(f'{cwd}\dataset\Brand Images\hrc_logo.bmp')
    test_output = f'{self.media.outputPath}/logo.mp4'
    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    video = cv.VideoWriter(
      test_output,
      fourcc,
      self.media.fps,
      (self.media.resolution_width, self.media.resolution_height),
      True
    )

    kp1, des1 = self.getFeatures(logo)
    #bgr_data2 = self._convert_bgr(cv.cvtColor())

    #processed_frame = self.process_frame(orb, des1, kp1, test, logo)
    #cv.imshow("preview", processed_frame)
    #cv.waitKey()
    print('Starting processing video for logos')

    #self.process_frame(test, logo)
    for frame in range(self.total_frames):
      start = frame * self.frame_size
      end = start + self.frame_size
      bgr_data = self._convert_bgr(self.raw_data[start:end])
      #region = self.processFrame2(bgr_data, train_features)

      processed_frame = self.process_frame(des1, kp1, bgr_data, logo)
      video.write(processed_frame)
      #cv.imshow("preview", bgr_data)
      #cv.waitKey()

    print('  Finished processing and writing out video for logos')
    video.release()

  def process_frame(self, des1, kp1, frame, logo):
    # Takes in BGR data and processes it (detect logo and draw green bounding box)

    # HUGE TODO list:
    # 1. Logo detection
    # 2. Green bounding box
    # 3. Replace logo?

    # output_frame = frame.copy()
    gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # scale = 5
    # upscaled_frame = cv.resize(
    #   output_frame,
    #   (self.media.resolution_width * scale, self.media.resolution_height * scale),
    #   interpolation=cv.INTER_AREA
    # )
    # kp2, des2 = self.getFeatures(upscaled_frame)
    kp2, des2 = self.orb.detectAndCompute(gray_frame, None)

    if not kp2:
        return None

    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(des1, des2, k=2)
    #matches = sorted(matches, key = lambda x:x.distance)

    good_matches = []

    for m, n in matches:
      if m.distance < 0.8 * n.distance:
        good_matches.append(m)

    if len(good_matches) < 0.05 * len(kp1) or len(good_matches) < 8:
      #print(len(kp1))
      #print(len(good_matches))
      #cv.imshow("preview", frame)
      #cv.waitKey()
      return None

    # good_matches = sorted(good_matches, key = lambda x:x.distance)[:16]
    #good_matches = matches[:10]
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good_matches ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good_matches ]).reshape(-1,1,2)
    M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC,5.0)

    # M, _ = cv.estimateAffinePartial2D(src_pts, dst_pts)
    # M = np.vstack([M, [0,0,1]])
    # matchesMask = mask.ravel().tolist()
    #
    # if sum(matchesMask) < 8:
    #     return None

    # h,w = logo.shape[:2]
    # h,w = self.media.resolution_height * scale, self.media.resolution_width * scale
    h,w = int(self.media.resolution_height / (2**self.logo_downscale)), int(self.media.resolution_width / (2**self.logo_downscale))
    # img3 = cv.drawMatches(logo, kp1, gray_frame, kp2, good_matches, None, flags=2)
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    if M is not None:
      dst = cv.perspectiveTransform(pts,M)

      rect = cv.minAreaRect(dst)

      #img3 = cv.drawMatches(logo,kp1,frame,kp2,good_matches, None,**draw_params)
      #img3 = cv.polylines(frame2, [np.int32(dst)], True, (0,0,255),3, cv.LINE_AA)
      if rect[1][0] < self.media.resolution_width // 12 or rect[1][1] < self.media.resolution_height // 12:
          return None
      box = cv.boxPoints(rect)
      box = np.int0(box)
      cv.drawContours(frame, [box], 0, (0, 255, 0), 2)

      # dst += (w, 0)
      # box = np.int0(cv.boxPoints(cv.minAreaRect(dst)))
      # cv.drawContours(img3, [box], 0, (0, 255, 0), 2)

    # cv.imshow(f'{self.cc}', img3); cv.waitKey(1)
    # time.sleep(0.1)
    # return cv.polylines(frame, [np.int32(dst)], True, (0,0,255),3, cv.LINE_AA)
    return frame

    # scaled_dst = dst / scale / 2
    #
    # rect = cv.minAreaRect(scaled_dst)
    # #img3 = cv.drawMatches(logo,kp1,frame,kp2,good_matches, None,**draw_params)
    # #img3 = cv.polylines(frame2, [np.int32(dst)], True, (0,0,255),3, cv.LINE_AA)
    #
    # box = cv.boxPoints(rect)
    # box = np.int0(box)
    # cv.drawContours(output_frame, [box], 0, (0, 255, 0), 2)
    # #print(len(kp1))
    # #print(len(good_matches))
    #
    # return output_frame

  def _convert_bgr(self, rgb_data):
    r_data = rgb_data[0:self.channel_size].reshape(
      (self.media.resolution_height, self.media.resolution_width)
    )
    g_data = rgb_data[self.channel_size:self.channel_size * 2].reshape(
      (self.media.resolution_height, self.media.resolution_width)
    )
    b_data = rgb_data[self.channel_size*2:self.channel_size*3].reshape(
      (self.media.resolution_height, self.media.resolution_width)
    )
    return np.stack((b_data, g_data, r_data), axis=-1)

  def _convert_rgb(self, bgr_data):
    b_data = bgr_data[:,:,:1].flatten()
    g_data = np.roll(bgr_data, -1)[:,:,:1].flatten()
    r_data = np.roll(bgr_data, -2)[:,:,:1].flatten()
    return np.concatenate((r_data, g_data, b_data))

  def getFeatures(self, img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    kps, descs = self.orb.detectAndCompute(gray, None)
    return kps, descs
