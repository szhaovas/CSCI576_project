import numpy as np
import pprint
from pydub import AudioSegment
from .detect import Detect

class Edit():

  def __init__(self, media, scenes):
    self.media = media
    self.scenes = scenes

    # Set tuple for ads (RGB, WAV)
    self.ad_replacements = []
    for ad in media.ad_scenes:
      self.ad_replacements.append(
        (f'{ad}.rgb', f'{ad}.wav')
      )

    # Initialize WAV files
    self.audio_ads = []
    for ad in self.ad_replacements:
      self.audio_ads.append(AudioSegment.from_file(
          ad[1],
          format='wav',
          sample_width=1,
          channels=self.media.audio_channels,
          frame_rate=self.media.sampling_rate
        )
      )
    self.audio_original = AudioSegment.from_file(
      media.pathToWAV,
      format='wav',
      sample_width=1,
      channels=self.media.audio_channels,
      frame_rate=self.media.sampling_rate
    )

    # Debugging scene info
    self.pp = pprint.PrettyPrinter(indent=4)
    # self.pp.pprint(scenes)
    # self.pp.pprint([scene for scene in scenes if scene['is_ad']])

  def read_rgb(self, filename, start_frame, end_frame):
      nframes = end_frame - start_frame
      data = np.zeros(
        (nframes, \
        self.media.resolution_height, \
        self.media.resolution_width, 3), \
        dtype='uint8')

      f = open(filename, 'rb')
      start_bytes = start_frame*self.media.frame_size
      f.seek(start_bytes)
      for ifr in range(nframes):
          for ic in range(3):
              buf = np.fromfile(f, dtype='uint8', \
                  count=self.media.resolution_height*self.media.resolution_width)
              if buf.size != 0:
                  data[ifr,:,:,ic] = buf.reshape(
                    (self.media.resolution_height, \
                    self.media.resolution_width)
                )

      f.close()

      return data


  def replaceAds(self):
    # Replace Ads
    print('Replacing ad segments in video')

    # Create lists of kept/removed scenes
    kept_segments_s = []
    kept_segments_frames = []
    removed_segments = [] # Not used
    ad_flag = False
    # [ Start scene, End Scene, Is Scene before an ad ]
    for scene in self.scenes:
      if not scene['is_ad']:
        kept_segments_s.append([scene['start_time'], scene['end_time'], False])
        kept_segments_frames.append([scene['start_frame'], scene['end_frame'], False])
        ad_flag = False
      elif scene['is_ad'] and not ad_flag:
        removed_segments.append([scene['start_time'], scene['end_time']])
        if kept_segments_s:
          kept_segments_s[-1][2] = True
        if kept_segments_frames:
          kept_segments_frames[-1][2] = True
        ad_flag = True
      elif scene['is_ad'] and ad_flag:
        removed_segments[-1][1] = scene['end_time']

    # Create output audio
    # print(' Creating output WAV file')
    # ad_count = 0
    # final_audio = AudioSegment.empty()
    # for segment in kept_segments_s:
    #   start_ms = segment[0] * 1000
    #   end_ms = segment[1] * 1000
    #   final_audio += self.audio_original[start_ms:end_ms]
    #   if segment[2] and ad_count < len(self.ad_replacements):
    #     final_audio += self.audio_ads[ad_count]
    #     ad_count += 1
    # final_audio.export(
    #   self.media.pathToOutputWAV,
    #   format='wav',
    #   bitrate=self.media.audio_bitrate
    # )

    # Initialize Video ads files
    video_ad_files = [open(ad[0]) for ad in self.ad_replacements]
    video_ad_data = [np.fromfile(ad, dtype=np.uint8, count=-1) for ad in video_ad_files]

    # Create output RGB
    print(f' Creating output files with {len(kept_segments_frames)} scenes')
    # ad_count = 0
    final_audio = AudioSegment.empty()
    rgb_file = open(self.media.pathToRGB, 'rb')
    # original_data = np.fromfile(rgb_file, dtype=np.uint8, count=-1)
    detect = Detect(self.media)
    ad_flag = False
    ad_index = 0
    with open(self.media.pathToOutputRGB, 'wb') as outputFile:
      count = 1
      for segment_frame, segment_s in zip(kept_segments_frames, kept_segments_s):
        # start = segment_frame[0] * self.media.frame_size
        # end = segment_frame[1] * self.media.frame_size
        # length = segment_frame[1] - segment_frame[0]
        # data = bytes(bytearray(original_data[start:end].tolist()))
        # data = original_data[start:end]
        data = self.read_rgb(self.media.pathToRGB, segment_frame[0], segment_frame[1])

        start_ms = segment_s[0] * 1000
        end_ms = segment_s[1] * 1000
        final_audio += self.audio_original[start_ms:end_ms]

        # Logo detection
        print(f'    Detecting logos for scene {count}')
        # final_out, ad_detected_in_scene, index = detect.detect_logos_in_scene(data, length)
        ad_detected_in_scene, index = detect.detect_logos_in_scene(data, outputFile)
        if ad_detected_in_scene:
          print(f'      !!! Detected ad: {self.media.ad_scenes[index]}')
          ad_flag = True
          ad_index = index
        print(f'    Finished logo detection for scene {count}')

        # outputFile.write(final_out)

        if segment_frame[2] and ad_flag:
          print(f'      Inserting ad {self.media.ad_scenes[ad_index]}')
          ad_data = bytes(bytearray(video_ad_data[ad_index].tolist()))
          outputFile.write(ad_data)
          final_audio += self.audio_ads[ad_index]
          ad_flag = False

        count += 1

    print(f'  Exporting final audio')
    final_audio.export(
      self.media.pathToOutputWAV,
      format='wav',
      bitrate=self.media.audio_bitrate
    )

    # Close files
    for ad in video_ad_files:
      ad.close()
    rgb_file.close()

    return (self.media.pathToOutputRGB, self.media.pathToOutputWAV)
