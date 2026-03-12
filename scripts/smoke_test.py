import os, sys, numpy as np, soundfile as sf
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from processing.audio_ops import standard_convert_to_mp3, fwht_transform_and_mp3

def main():
    out_dir = os.path.join(os.getcwd(), 'output')
    os.makedirs(out_dir, exist_ok=True)
    wav = os.path.join(out_dir, 'test_sine.wav')
    sr=44100
    x = 0.2*np.sin(2*np.pi*440*np.arange(sr)/sr).astype('float32')
    sf.write(wav, x, sr)
    print('WAV written:', wav)
    std_mp3, t_std = standard_convert_to_mp3(wav, out_dir)
    print('STD:', std_mp3, t_std)
    fwht_mp3, t_fwht = fwht_transform_and_mp3(wav, out_dir, block_size=2048, select_mode='none')
    print('FWHT:', fwht_mp3, t_fwht)

if __name__ == '__main__':
    main()
