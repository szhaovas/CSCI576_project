import sys
import os
import argparse
import numpy as np
from PyQt5.QtWidgets import QApplication

from config import Media, Player
from process import Scenes, Edit, Detect

def main():
  cwd = os.getcwd()
  parser = argparse.ArgumentParser(description = \
    'CSCI 576 Multimedia project')
  parser.add_argument('rgb', metavar='rgb', help='Path to .rgb file')
  parser.add_argument('wav', metavar='wav', help='Path to .wav file')
  parser.add_argument('-rgbout', metavar='rgbout', help='Path to the output .rgb file', default=f'{cwd}/output/output.rgb')
  parser.add_argument('-wavout', metavar='wavout', help='Path to the output .wav file', default=f'{cwd}/output/output.wav')
  parser.add_argument('--ad-scenes', type=str, nargs='+', metavar='ad_scenes', default=[],
    help='Path(s) (w/o file suffix) to ad segments to fill in. Expecting both .rgb and .wav files. MUST correlate with --logos.'
  )
  parser.add_argument('--logos', type=str, nargs='+', metavar='logos', default=[],
    help='Path(s) to image files containing logos to detect. MUST correlate with --ad-scenes.'
  )
  args = parser.parse_args()
  print(args)

  if len(args.ad_scenes) != len(args.logos):
    sys.exit('Input arguments for logos and ad scenes are not equal! They must match each other.')

  media = Media(
    args.rgb,
    args.wav,
    args.rgbout,
    args.wavout,
    ad_scenes=args.ad_scenes,
    logos=args.logos,
    debug=True
  )

  # Logo testing
  # detect = Detect(media)
  # original_rgb_file = open(media.pathToRGB, 'rb')
  # original_data = np.fromfile(original_rgb_file, dtype=np.uint8, count=-1)

  # with open(f'{media.outputPath}/logo.rgb', 'wb') as outputFile:
  #   frame_start = 5300
  #   frame_end = 5330
  #   start = frame_start * media.frame_size
  #   end = frame_end * media.frame_size
  #   length = frame_end - frame_start
  #   data = original_data[start:end]
  #   final_out, ad_detected_in_scene, index = detect.detect_logos_in_scene(data, length)
  #   print(f'Ad detected: {ad_detected_in_scene}')
  #   print(f'ad: {media.ad_scenes[index]}')
  #   outputFile.write(final_out)
  #   media.outputMP4(f'{media.outputPath}/logo.rgb', 'logo.mp4')

  scenes = Scenes(media)
  edit = Edit(media, scenes.generate_scenes_info())

  app = QApplication(sys.argv)

  original_player = Player(media.pathToRGB, media.pathToWAV)
  modified_player = Player(*edit.replaceAds())

  if media.enable_debug:
    media.outputMP4(media.pathToOutputRGB)
    print(f'Verify ad removal and logo identification at {media.edit_output}')

  print('Launching RGB player')

  app.exec_()
  original_player.rgb_f.close()
  modified_player.rgb_f.close()

if __name__ == "__main__":
    main()
