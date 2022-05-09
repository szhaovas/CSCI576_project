from scenedetect import VideoManager
from scenedetect import SceneManager
from scenedetect.detectors import ContentDetector, AdaptiveDetector

import numpy as np
import librosa
import wave
import cv2
from statistics import mean, stdev

from config import media

class Scenes():

  def __init__(self, media, threshold=3.0, min_scene_len=10):
    # Default min_scene_len = 10 frames
    # Default threshold = 30.0
    # ^: 8bit intensity value for each pixel must be <= to trigger scene change
    self.min_scene_len = min_scene_len
    self.threshold = threshold
    self.luma_only = False

    self.media = media
    # self.video_path = media.pathToRGB

    mp4_path = media.outputMP4(media.pathToRGB, 'original.mp4')

    self.cv2_video = cv2.VideoCapture(mp4_path)

    # Create our video & scene managers, then add the detector.
    video_manager = VideoManager([mp4_path])
    scene_manager = SceneManager()
    scene_manager.add_detector(
        AdaptiveDetector(
          video_manager,
          adaptive_threshold=self.threshold,
          min_scene_len=self.min_scene_len,
          luma_only=self.luma_only,
          min_delta_hsv=15.0,
          window_width=2
        ))

    # Improve processing speed by downscaling before processing.
    video_manager.set_downscale_factor()

    # Start the video manager and perform the scene detection.
    video_manager.start()
    print('Detecting scenes...')
    scene_manager.detect_scenes(frame_source=video_manager)

    # Each returned scene is a tuple of the (start, end) timecode.
    self.scenes = scene_manager.get_scene_list()
    # self.cuts = scene_manager.get_cut_list()

    print('List of scenes obtained:')
    for i, scene in enumerate(self.scenes):
      print(
        'Scene %2d: Start %s / Frame %d, End %s / Frame %d' % (
          i+1,
          scene[0].get_timecode(), scene[0].get_frames(),
          scene[1].get_timecode(), scene[1].get_frames(),))

    wav_f = wave.open(self.media.pathToWAV)
    wav_nframes = wav_f.getnframes()
    self.wav_buf = np.fromstring(wav_f.readframes(wav_nframes), np.int16)
    self.total_frames = self.scenes[-1][1].get_frames()
    self.wav_nframes_per_rgb = wav_nframes / self.total_frames

  def generate_scenes_info(self):
    scenes_info = []

    # Initialize data from scenes
    print('Checking which scenes are ads')
    for i, scene in enumerate(self.scenes):
      scene_length = scene[1].get_seconds() - scene[0].get_seconds()
      motion_characteristic = self._avg_motion(scene[0].get_frames(), scene[1].get_frames())
      audio_levels = self._spectral_contrast(scene[0].get_seconds(), scene[1].get_seconds())

      audio_entropy = self._spectral_entropy(scene[0].get_frames(), scene[1].get_frames())

      scene_info = {
        'scene_id': i,
        'length_seconds': scene_length,
        'motion': motion_characteristic,
        'audio_level': audio_levels,
        'audio_entropy': audio_entropy,
        'start_time': scene[0].get_seconds(),
        'end_time': scene[1].get_seconds(),
        'start_timestamp': scene[0].get_timecode(precision=2),
        'end_timestamp': scene[1].get_timecode(precision=2),
        'start_frame': scene[0].get_frames(),
        'end_frame': scene[1].get_frames(),
        'len_score': 0,
        'audio_score': 0,
        'motion_score': 0,
        'is_ad': False
      }
      scenes_info.append(scene_info)


    # Determine if scene is an Ad
    threshold = 1 # Number of standard deviations to detect outliers
    # Detect outliers in length (Shorter length correlated to ads)
    print(' Scoring scene length')
    len_min = 10.0 # Any scene len > 10 not counted as ad
    len_mean = mean(scene['length_seconds'] for scene in scenes_info)
    len_stdev = stdev(scene['length_seconds'] for scene in scenes_info)
    for scene in scenes_info:
      z_score = (scene['length_seconds'] - len_mean) / len_stdev
      if np.abs(z_score) <= threshold and scene['length_seconds'] < len_min:
        scene['len_score'] += 1
    # Detect outliers in audio level (Less noise correlated to ads)
    print(' Scoring audio')
    # TODO: Verify additional edge cases for audio
    # audio_min = 500 # Only include audio greater than
    audio_mean = mean(scene['audio_level'] for scene in scenes_info)
    audio_stdev = stdev(scene['audio_level'] for scene in scenes_info)
    for scene in scenes_info:
      z_score = (scene['audio_level'] - audio_mean) / audio_stdev
      if np.abs(z_score) <= threshold:
        scene['audio_score'] += 1

    print(' Scoring audio entropy')
    norm_scene_nframes = np.array(
        [scene['end_frame'] - scene['start_frame'] for scene in scenes_info]) \
        / self.total_frames
    audio_ent_mean = np.average(
        [scene['audio_entropy'] for scene in scenes_info], \
        weights = norm_scene_nframes
    )
    audio_ent_stdev = np.sqrt(
        np.cov(
            [scene['audio_entropy'] for scene in scenes_info], \
            aweights = norm_scene_nframes
        )
    )
    for scene in scenes_info:
      z_score = (audio_ent_mean - scene['audio_entropy']) / audio_ent_stdev
      if np.abs(z_score) > 2:
        scene['audio_score'] += 1

    # TODO: Motion detection
    print(' Scoring scene motion')
    # Either motion is implemented wrong or motion may be a bad metric to determine ads
    motion_mean = mean(scene['motion'] for scene in scenes_info)
    motion_stdev = stdev(scene['motion'] for scene in scenes_info)
    for scene in scenes_info:
      z_score = (scene['motion'] - motion_mean) / motion_stdev
      if np.abs(z_score) <= threshold:
        scene['motion_score'] += 0 # 0 -> disabled

    ad_threshold = 2 # Threshold score to determine if scene is an ad
    for scene in scenes_info:
      total = scene['len_score'] + scene['audio_score'] + scene['motion_score']
      if total >= ad_threshold:
        scene['is_ad'] = True

    return scenes_info

  def _zero_crossing_rate(self, start, end):
    # Calculate zero crossing rate for the audio
    # Higher value w/ percussive sounds/music
    duration = end - start
    audio_waveform, _ = librosa.load(self.media.pathToWAV, offset=start, duration=duration)
    return sum(librosa.zero_crossings(audio_waveform)) / duration

  def _spectral_contrast(self, start, end):
    # Calculate spectral constrast
    # Valleys - correlate to noise
    # Peaks - harmonic components
    duration = end - start
    audio_waveform, sr = librosa.load(self.media.pathToWAV, offset=start, duration=duration)
    S = np.abs(librosa.stft(audio_waveform))
    spectral_contrast = librosa.feature.spectral_contrast(S=S, sr=sr)
    # Works well to identify video but can't tell if it's just due to length of scene
    # Length of subarrays (same across the list) indicate different octave-based freqs (I think?)
    return len(spectral_contrast[0])

  '''
  Normal scenes typically have entropy >10
  Ads typically have entropy <9
  '''
  def _spectral_entropy(self, start, end, sampling_rate=48000):
    scene_audio = self.wav_buf[
        round(start*self.wav_nframes_per_rgb):round(end*self.wav_nframes_per_rgb)
    ]
    nframes = scene_audio.size

    fhat = np.fft.fft(scene_audio, nframes)
    PSD = fhat * np.conj(fhat) / nframes
    PDF = PSD / np.sum(PSD)

    return -np.sum(PDF * np.log(PDF))

  def _avg_motion(self, startFrame, endFrame):
    # Calculate average motion during scene
    totalDiffPixels = 0
    duration = endFrame - startFrame
    self.cv2_video.set(cv2.CAP_PROP_POS_FRAMES, startFrame)
    # Get starting frame
    _, prevFrame = self.cv2_video.read()
    # Convert to grayscale
    # prevFrame = cv2.cvtColor(prevFrame, cv2.COLOR_BGR2GRAY)
    for i in range(duration-1):
      _, frame = self.cv2_video.read()
      diff = cv2.absdiff(prevFrame, frame)
      gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
      blur = cv2.GaussianBlur(gray, (5, 5), 0)
      _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
      dilated = cv2.dilate(thresh, None, iterations=3)
      totalDiffPixels += np.sum(dilated == 255)
      prevFrame = frame.copy()
    return totalDiffPixels / duration
