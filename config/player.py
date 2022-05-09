import wave
import numpy as np
import sounddevice as sd
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import QMainWindow, QWidget, QLabel, QApplication, \
    QVBoxLayout, QPushButton, QStyle
from PyQt5.QtCore import QTimer, Qt, pyqtRemoveInputHook

# import pdb; pyqtRemoveInputHook(); pdb.set_trace()

class Player(QMainWindow):
    def __init__(self, rgb_filepath, wav_filepath, window_width=480, window_height=270, \
        fps=30, buf_len=300, wav_sample_rate=48000):
        super().__init__()
        self.setGeometry(0, 0, window_width, window_height)
        self.init_ui()

        self.width, self.height = window_width, window_height

        self.rgb_f = open(rgb_filepath, 'rb')
        self.rgb_f.seek(0, 2)
        self.total_frames = self.rgb_f.tell() / (window_width*window_height*3)
        self.rgb_f.seek(0, 0)
        self.buf_len = buf_len
        self.rgb_buf = np.zeros((buf_len, window_height, window_width, 3), dtype='uint8')
        self.frame_counter = 0

        sd.default.samplerate = wav_sample_rate
        wav_f = wave.open(wav_filepath)
        wav_nframes = wav_f.getnframes()
        self.wav_buf_orig = np.fromstring(wav_f.readframes(wav_nframes), np.int16)
        self.wav_buf = np.copy(self.wav_buf_orig)
        self.wav_nframes_per_rgb = wav_nframes / self.total_frames

        self.playing = False
        self.update_buf()

        image = QImage(self.rgb_buf[self.frame_counter % self.buf_len, ...], self.width, \
            self.height, 3*self.width, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(image)
        self.video_label.setPixmap(pixmap)
        self.frame_counter += 1

        self.timer = QTimer()
        self.timer.setTimerType(Qt.PreciseTimer)
        self.timer.timeout.connect(self.next_frame)
        self.timer.start(int(1000.0 / fps))
        self.timer.stop()

        self.show()
        self.setFocus()

    def init_ui(self):
        widget = QWidget(self)
        self.setCentralWidget(widget)

        self.video_label = QLabel()

        self.play_btn = QPushButton()
        self.play_btn.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.play_btn.clicked.connect(self.play_pause)

        self.stop_btn = QPushButton()
        self.stop_btn.setIcon(self.style().standardIcon(QStyle.SP_MediaStop))
        self.stop_btn.clicked.connect(self.replay)
        self.stop_btn.setEnabled(False)

        vboxLayout = QVBoxLayout()
        vboxLayout.addWidget(self.video_label)
        vboxLayout.addWidget(self.play_btn)
        vboxLayout.addWidget(self.stop_btn)

        widget.setLayout(vboxLayout)

    def update_buf(self):
        sd.stop()
        self.wav_buf = self.wav_buf_orig[round(self.frame_counter*self.wav_nframes_per_rgb):]
        if self.playing:
            sd.play(self.wav_buf)
        for ifr in range(self.buf_len):
            for ic in range(3):
                buf = np.fromfile(self.rgb_f,  dtype='uint8', \
                    count=self.width*self.height)
                if buf.size != 0:
                    self.rgb_buf[ifr, :,:,ic] = buf.reshape((self.height, self.width))

    def next_frame(self):
        image = QImage(self.rgb_buf[self.frame_counter % self.buf_len, ...], self.width, \
            self.height, 3*self.width, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(image)
        self.video_label.setPixmap(pixmap)
        self.frame_counter += 1

        if self.frame_counter >= self.total_frames:
            self.timer.stop()
            self.play_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
            self.playing = False

        if self.frame_counter % self.buf_len == 0:
            self.update_buf()

    def replay(self):
        self.play_btn.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.play_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)

        self.wav_buf = self.wav_buf_orig
        self.rgb_f.seek(0, 0)
        self.frame_counter = 0
        self.update_buf()

        image = QImage(self.rgb_buf[self.frame_counter % self.buf_len, ...], self.width, \
            self.height, 3*self.width, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(image)
        self.video_label.setPixmap(pixmap)
        self.frame_counter += 1

        self.playing = False
        self.timer.stop()

        self.play_btn.setEnabled(True)

    def play_pause(self):
        self.playing = not self.playing
        if self.playing:
            self.play_btn.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))
            self.timer.start()
            sd.play(self.wav_buf)
            self.stop_btn.setEnabled(False)
        else:
            self.play_btn.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
            self.timer.stop()
            sd.stop()
            self.wav_buf = self.wav_buf_orig[round(self.frame_counter*self.wav_nframes_per_rgb):]
            self.stop_btn.setEnabled(True)
