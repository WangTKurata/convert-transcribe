import os
import sys
import subprocess
import whisper

def set_ffmpeg_path():
    """FFmpegのパスを環境変数に追加"""
    ffmpeg_path = "C:/ffmpeg/bin"
    os.environ["PATH"] += os.pathsep + ffmpeg_path

def extract_audio(mp4_path, mp3_path):
    """FFmpegを使用してMP4からMP3を抽出"""
    print(f"音声抽出開始: {mp3_path}")
    command = [
        "ffmpeg",
        "-i", mp4_path,      # 入力MP4
        "-q:a", "0",         # 高音質
        "-ac", "1",          # モノラル
        "-vn",               # 映像なし
        mp3_path
    ]
    subprocess.run(command, check=True)
    print(f"✅ 音声抽出完了: {mp3_path}")

def transcribe_audio(mp3_path, output_txt):
    """Whisperを使用してMP3を文字起こし"""
    try:
        model = whisper.load_model("medium")  # モデル読み込み
        result = model.transcribe(mp3_path, language="ja", verbose=True)  # 日本語指定

        with open(output_txt, "w", encoding="utf-8") as f:
            f.write(result["text"])
        
        print(f"✅ 文字起こし完了: {output_txt}")
        print("文字起こし結果:")
        print(result["text"])
    except Exception as e:
        print(f"❌ 文字起こし中にエラーが発生しました: {e}")

def main(mp4_path):
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

    print("🎉 全処理完了！")

if __name__ == "__main__":
    set_ffmpeg_path()  # FFmpegのパスを設定
    if len(sys.argv) < 2:
        print("❌ MP4ファイルを指定してください")
        print("使い方: python convert_and_transcribe.py input.mp4")
        sys.exit(1)
    
    mp4_path = sys.argv[1]
    main(mp4_path)
