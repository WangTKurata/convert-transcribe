import os
import subprocess
import sys
from datetime import datetime

import intel_extension_for_pytorch as ipex
import torch

import whisper

os.environ["PYTORCH_XPU_ALLOCATOR"] = "1"
# ipex.debug(True)  # デバッグモードを有効化

# Intel XPU（GPU）を使用する
device = "xpu" if torch.xpu.is_available() else "cpu"
print(f"Intel XPU（GPU）使用可否: {device}")


def set_ffmpeg_path():
    """FFmpegのパスを環境変数に追加"""
    ffmpeg_path = "C:/ffmpeg/bin"
    os.environ["PATH"] += os.pathsep + ffmpeg_path


def extract_audio(mp4_path, mp3_path):
    """FFmpegを使用してMP4からMP3を抽出"""
    print(f"音声抽出開始: {mp3_path}")
    command = [
        "ffmpeg",
        "-y",  # 強制上書き
        "-i",
        mp4_path,  # 入力MP4
        "-q:a",
        "5",  # 中音質
        "-ac",
        "1",  # モノラル
        "-vn",  # 映像なし
        mp3_path,
    ]
    subprocess.run(command, check=True)
    print(f"✅ 音声抽出完了: {mp3_path}")


def transcribe_audio(mp3_path, output_txt):
    """Whisperを使用してMP3を文字起こし"""
    try:
        model = whisper.load_model("medium")  # モデル読み込み
        model.eval()  # 推論モードに変更
        # IPEX を適用して最適化
        model = ipex.optimize(model)

        # メモリ使用量の確認（ロード後）
        print(
            "ロード後のメモリ使用量:", torch.xpu.memory_allocated(0) / 1024 / 1024, "MB"
        )
        result = model.transcribe(
            mp3_path, language="ja", verbose=True, initial_prompt="です。ます。でした。"
        )
        # メモリ使用量の確認（推論後）
        print(
            "推論後のメモリ使用量:", torch.xpu.memory_allocated(0) / 1024 / 1024, "MB"
        )

        with open(output_txt, "w", encoding="utf-8") as f:
            f.write(result["text"])

        print(f"✅ 文字起こし完了: {output_txt}")
        print("文字起こし結果:")
        print(result["text"])
    except Exception as e:
        print(f"❌ 文字起こし中にエラーが発生しました: {e}")


def main(mp4_path):
    start_time = datetime.now()  # 開始時間を記録
    if not os.path.exists(mp4_path):
        print(f"❌ ファイルが見つかりません: {mp4_path}")
        return

    # 出力ファイルのパス
    base_name = os.path.splitext(mp4_path)[0]
    mp3_path = f"{base_name}.mp3"
    output_txt = f"{base_name}.txt"

    # MP3抽出 & 文字起こし
    extract_audio(mp4_path, mp3_path)
    transcribe_audio(mp3_path, output_txt)

    end_time = datetime.now()  # 終了時間を記録
    elapsed_time = end_time - start_time  # 経過時間を計算
    print(f"🎉 全処理完了！実行時間: {elapsed_time}")


if __name__ == "__main__":
    set_ffmpeg_path()  # FFmpegのパスを設定
    if len(sys.argv) < 2:
        print("❌ MP4ファイルを指定してください")
        print("使い方: python convert_and_transcribe.py input.mp4")
        sys.exit(1)

    mp4_path = sys.argv[1]
    main(mp4_path)
