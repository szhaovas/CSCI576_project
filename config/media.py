import os
import cv2
import numpy as np

class Media():
  # Holds all the configuration data, paths, attributes, etc

  def __init__(self, pathToRGB, pathToWAV, pathToOutputRGB, pathToOutputWAV, ad_scenes=[], logos=[], outputDir='output', debug=False):
    self.pathToRGB = pathToRGB
    self.pathToWAV =  pathToWAV
    self.pathToOutputRGB = pathToOutputRGB
    self.pathToOutputWAV = pathToOutputWAV
    self.ad_scenes = ad_scenes
    self.logos = logos

    # Video Attributes (logo images uses same resolution and format)
    self.fps = 30 # Frame rate
    self.format = 'rawvideo' # Input format
    self.pix_fmt = 'rgb24' # ffmpeg -pix_fmts
    self.resolution = '480x270' # Resolution
    self.resolution_width = 480
    self.resolution_height = 270
    self.frame_size = self.resolution_width * self.resolution_height * 3
    self.channel_size = self.resolution_width * self.resolution_height

    # Audio Attributes
    self.audio_bitrate = '768k' # 16 bits/sample * 48000Hz
    self.sampling_rate = 48000 # Sampling Rate 48000Hz
    self.audio_channels = 1

    # Verify output directory exists
    self.outputPath = os.path.dirname(pathToOutputRGB)
    if not os.path.exists(self.outputPath):
      os.makedirs(self.outputPath)

    # Remove output files if exists
    if os.path.exists(self.pathToOutputWAV):
      os.remove(self.pathToOutputWAV)
    if os.path.exists(self.pathToOutputRGB):
      os.remove(self.pathToOutputRGB)

    self.enable_debug = debug
    self.logo_output = f'{self.outputPath}/logo.mp4'
    self.edit_output = f'{self.outputPath}/edit.mp4'

  def getRGBData(self):
    rgb_file = open(self.pathToRGB, 'rb')
    return np.fromfile(rgb_file, dtype=np.uint8, count=-1)

  def outputMP4(self, rgb_file, output_file=''):
    if output_file:
      output = f'{self.outputPath}/{output_file}'
    else:  
      output = self.edit_output
    if os.path.exists(output):
      os.remove(output)
    rgb_file = open(rgb_file, 'rb')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # Calculate total frames
    rgb_file.seek(0, 2)
    total_frames = int(rgb_file.tell() / self.frame_size)
    rgb_file.seek(0, 0)

    video = cv2.VideoWriter(output, fourcc, self.fps, (self.resolution_width, self.resolution_height), True)
    for _ in range(total_frames):
      raw_data = np.fromfile(rgb_file, dtype=np.uint8, count=self.frame_size)
      r_data = raw_data[0:self.channel_size].reshape((self.resolution_height, self.resolution_width))
      g_data = raw_data[self.channel_size:self.channel_size * 2].reshape((self.resolution_height, self.resolution_width))
      b_data = raw_data[self.channel_size * 2:self.channel_size * 3].reshape((self.resolution_height, self.resolution_width))
      bgr_data = np.stack((b_data, g_data, r_data), axis=-1)
      # shaped_data = np.reshape(raw_data, (self.resolution_height, self.resolution_width, 3), order='C')
      video.write(bgr_data)
    video.release()
    rgb_file.close()
    return output
