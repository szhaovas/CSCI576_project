USC CSCI576 multimedia system final team project
Original repo at https://github.com/sunshuu/multimedia-project
This is a cleaned-up version of the memory branch from the source

Player:
- python3 player.py <path_to_rgb_file> <path_to_wav_file>
- python3 player.py -h to get details

Ads detection, replacement, and bounding box:
- python3 main.py <path_to_rgb_file> <path_to_wav_file> --logos <path_to_logo1_rgb_file> <path_to_logo2_rgb_file> --ad-scenes <path_to_replacement_ad1> <path_to_replacement_ad2>
- do not include file extensions such as .txt or .wav in <path_to_replacement_ad1>; place the <replacement_ad1.rgb> and <replacement_ad1.wav> in the same directory and only input <replacement_ad1>; main.py will load both .rgb and .wav
- outputs generated in in output folder
- python3 main.py -h to get details

Please find requirements on input data format in input_data_format.txt
