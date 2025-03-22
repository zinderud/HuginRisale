import os
import torch
import datetime
import warnings
from transformers import pipeline
from huggingface_hub import HfApi
from pydub import AudioSegment
import json
import re
import yt_dlp
import shutil

# Uyarıları bastır
warnings.filterwarnings("ignore", category=FutureWarning)

# Hugging Face kimlik bilgileri
HF_TOKEN = ""  # Kendi geçerli token'ınızla değiştirin
HF_USERNAME = "zinderud"
REPO_NAME = "risale-sohbet-turkish"

def setup_huggingface_repo():
    """Hugging Face repo ayarları"""
    api = HfApi(token=HF_TOKEN)
    
    try:
        api.repo_info(f"{HF_USERNAME}/{REPO_NAME}", repo_type="dataset")
        print(f"Repo {REPO_NAME} zaten mevcut.")
    except Exception as e:
        if "401" in str(e) or "404" in str(e):
            repo_url = api.create_repo(
                repo_id=f"{HF_USERNAME}/{REPO_NAME}",
                repo_type="dataset",
                private=False,
                exist_ok=True
            )
            print(f"Yeni repo oluşturuldu: {repo_url}")
            
            readme_content = f"""# YouTube Transkripsiyon Veri Seti

## Veri Yapısı
- `audio/`: MP3 dosyaları
- `transcripts/`: Metin transkripsiyonları
- `srt/`: Altyazı dosyaları
- `metadata/`: Video bilgileri
- `database.json`: Tüm videoların indeksi

## Güncelleme Tarihi
{datetime.datetime.now().strftime('%Y-%m-%d')}
"""
            api.upload_file(
                path_or_fileobj=readme_content.encode(),
                path_in_repo="README.md",
                repo_id=f"{HF_USERNAME}/{REPO_NAME}",
                repo_type="dataset"
            )
            print("README yüklendi")
        else:
            print(f"Repo kontrolü sırasında beklenmeyen hata: {e}")
            return None
    
    local_dir = REPO_NAME
    os.makedirs(os.path.join(local_dir, "audio"), exist_ok=True)
    os.makedirs(os.path.join(local_dir, "transcripts"), exist_ok=True)
    os.makedirs(os.path.join(local_dir, "srt"), exist_ok=True)
    os.makedirs(os.path.join(local_dir, "metadata"), exist_ok=True)
    
    return local_dir

def download_youtube_audio(url, output_dir):
    """YouTube'dan ses indirme (yt-dlp ile)"""
    try:
        ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
            'outtmpl': os.path.join(output_dir, '%(title)s.%(ext)s'),
            'quiet': True,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            filename = ydl.prepare_filename(info).replace('.webm', '.mp3').replace('.m4a', '.mp3')
            filepath = os.path.join(output_dir, f"{clean_filename(info['title'])}.mp3")
            if os.path.exists(filename) and filename != filepath:
                os.rename(filename, filepath)
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"Dosya indirilemedi: {filepath}")
            
            print(f"Dosya indirildi: {filepath}")
            return filepath, {
                "title": info['title'],
                "author": info.get('uploader', 'Bilinmiyor'),
                "duration": info.get('duration', 0),
                "views": info.get('view_count', 0),
                "publish_date": info.get('upload_date', 'Bilinmiyor'),
                "url": url
            }
    except Exception as e:
        print(f"İndirme hatası: {str(e)}")
        return None, None

def clean_filename(text):
    """Güvenli dosya adı oluşturma"""
    text = re.sub(r'[^\w\s-]', '', text)
    text = re.sub(r'\s+', '_', text)
    return text[:50]

def transcribe_audio(filepath, language="tr"):
    """Whisper ile transkripsiyon"""
    try:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Transkripsiyon için dosya bulunamadı: {filepath}")
        
        print(f"Transkripsiyon yapılıyor: {filepath}")
        transcriber = pipeline(
            "automatic-speech-recognition",
            model="openai/whisper-large-v2",
            device=0 if torch.cuda.is_available() else -1
        )
        result = transcriber(
            filepath,
            chunk_length_s=30,
            generate_kwargs={"language": language}
        )
        print(f"Transkripsiyon tamamlandı: {len(result['text'])} karakter")
        return result["text"], result.get("chunks", [])
    except Exception as e:
        print(f"Transkripsiyon hatası: {e}")
        return None, None

def create_srt(chunks, filename):
    """SRT dosyası oluşturma"""
    print(f"SRT dosyası oluşturuluyor: {filename}")
    srt_content = ""
    for i, chunk in enumerate(chunks):
        start = chunk["timestamp"][0]
        end = chunk["timestamp"][1]
        text = chunk["text"].strip()
        
        srt_content += f"{i+1}\n"
        srt_content += f"{format_time(start)} --> {format_time(end)}\n"
        srt_content += f"{text}\n\n"
    
    with open(filename, "w", encoding="utf-8") as f:
        f.write(srt_content)
    return filename

def format_time(seconds):
    """Saniyeyi SRT formatına çevirme"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    milliseconds = int((secs - int(secs)) * 1000)
    
    return (
        f"{hours:02d}:"
        f"{minutes:02d}:"
        f"{int(secs):02d},"
        f"{milliseconds:03d}"
    )

def save_metadata(metadata, filename):
    """Metadata kaydetme"""
    print(f"Metadata kaydediliyor: {filename}")
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    return filename

def update_database(local_dir, metadata):
    """Veritabanını güncelleme"""
    db_file = os.path.join(local_dir, "database.json")
    print(f"Veritabanı güncelleniyor: {db_file}")
    
    if os.path.exists(db_file):
        with open(db_file, "r", encoding="utf-8") as f:
            db = json.load(f)
    else:
        db = {"videos": []}
    
    existing_urls = {v["url"] for v in db["videos"]}
    if metadata["url"] not in existing_urls:
        db["videos"].append(metadata)
        db["last_updated"] = datetime.datetime.now().isoformat()
        
        with open(db_file, "w", encoding="utf-8") as f:
            json.dump(db, f, ensure_ascii=False, indent=2)
    
    return db

def process_video(url, local_repo):
    """Ana işlem akışı"""
    try:
        print(f"\nİşleniyor: {url}")
        
        audio_path, metadata = download_youtube_audio(url, os.path.join(local_repo, "audio"))
        if not audio_path:
            return False
        
        audio_path = os.path.abspath(audio_path)
        print(f"İndirilen dosya yolu (mutlak): {audio_path}")
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"İndirilen dosya bulunamadı: {audio_path}")
        
        transcript, chunks = transcribe_audio(audio_path)
        if not transcript:
            return False
        
        base_name = clean_filename(metadata["title"])
        srt_path = os.path.join(local_repo, "srt", f"{base_name}.srt")
        txt_path = os.path.join(local_repo, "transcripts", f"{base_name}.txt")
        metadata_path = os.path.join(local_repo, "metadata", f"{base_name}.json")
        
        create_srt(chunks, srt_path)
        with open(txt_path, "w", encoding="utf-8") as f:
            print(f"Transkript kaydediliyor: {txt_path}")
            f.write(transcript)
        save_metadata(metadata, metadata_path)
        update_database(local_repo, metadata)
        
        # HfApi ile dosyaları yükle
        api = HfApi(token=HF_TOKEN)
        for file_path in [audio_path, srt_path, txt_path, metadata_path, os.path.join(local_repo, "database.json")]:
            if os.path.exists(file_path):
                api.upload_file(
                    path_or_fileobj=file_path,
                    path_in_repo=file_path[len(local_repo)+1:],  # 'risale-sohbet-turkish/' kısmını çıkar
                    repo_id=f"{HF_USERNAME}/{REPO_NAME}",
                    repo_type="dataset",
                    commit_message=f"Add: {metadata['title']}"
                )
                print(f"Dosya yüklendi: {file_path}")
        
        print("Başarıyla tamamlandı!")
        return True
    
    except Exception as e:
        print(f"Hata oluştu: {str(e)}")
        return False

def main():
    api = HfApi(token=HF_TOKEN)
    try:
        whoami = api.whoami()
        print(f"Kimlik doğrulama başarılı, kullanıcı: {whoami['name']}")
    except Exception as e:
        print(f"Kimlik doğrulama başarısız: {e}")
        return
    
    if os.path.exists(REPO_NAME):
        shutil.rmtree(REPO_NAME)
        print(f"Eski {REPO_NAME} dizini temizlendi.")
    
    local_repo = setup_huggingface_repo()
    if local_repo is None:
        return
    
    urls = [
        "https://www.youtube.com/watch?v=RP3eibpPSs8",
  "https://www.youtube.com/watch?v=X9caEnhBUXU",
  "https://www.youtube.com/watch?v=YSzDVdnYrs4",
  "https://www.youtube.com/watch?v=IcBlmlr6nuw",
  "https://www.youtube.com/watch?v=uIv6QDro4lM",
  "https://www.youtube.com/watch?v=M6mFPJwPVdM",
 
    ]
    
    for url in urls:
        process_video(url, local_repo)
    
    print("\nTüm işlemler tamamlandı!")

if __name__ == "__main__":
    main()