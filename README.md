
# Whisper Audio Transcription

このプロジェクトは、MP4ファイルから音声を抽出し、Whisperモデルを使用して文字起こしを行うPythonスクリプトです。

## 必要条件

- Python 3.x
- ffmpeg
- Whisper

## インストール

1. ffmpegをインストールし、パスを通します。
   - [ffmpegのダウンロードページ](https://ffmpeg.org/download.html)からダウンロードできます。
   - `C:/ffmpeg/bin`にインストールし、環境変数に追加します。

2. 必要なPythonパッケージをインストールします。

```bash
conda create -n whisper-env python=3.11
conda activate whisper-env
conda install numpy=1.26 -y
conda install libuv -y
python -m pip install torch==2.5.1+cxx11.abi torchvision==0.20.1+cxx11.abi torchaudio==2.5.1+cxx11.abi intel-extension-for-pytorch==2.5.10+xpu --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/mtl/us/
pip install openai-whisper
```
Intel XE GPU を使いたいので、IPEX を使えるようにする必要がある。
https://pytorch-extension.intel.com/installation?platform=gpu&version=v2.5.10%2Bxpu&os=windows&package=pip

が、今のところ使えてない。IPEX 自体が実験的な存在。

## 使い方

1. スクリプトを実行します。
   ```bash
   python convert_and_transcribe.py input.mp4
   ```

2. スクリプトは以下の手順で動作します。
   - MP4ファイルから音声を抽出し、MP3ファイルを生成します。
   - Whisperモデルを使用してMP3ファイルを文字起こしし、テキストファイルを生成します。

