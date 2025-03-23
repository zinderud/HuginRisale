import os
import torch
import datetime
import warnings
from transformers import pipeline, AutoTokenizer
from huggingface_hub import HfApi
import json
import re
import yt_dlp
from collections import Counter
from keybert import KeyBERT

# Uyarıları bastır
warnings.filterwarnings("ignore", category=FutureWarning)

# Hugging Face kimlik bilgileri
HF_TOKEN = ""
HF_USERNAME = "zinderud"
REPO_NAME = "risale-sohbet-turkish"

# Cihaz belirleme
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Kullanılan cihaz: {device}")

# Tokenizer ve KeyBERT modeli
tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-turkish-cased")
kw_model = KeyBERT(model="dbmdz/bert-base-turkish-cased")

def setup_huggingface_repo():
    """Hugging Face repo ayarları"""
    api = HfApi(token=HF_TOKEN)
    local_dir = REPO_NAME
    os.makedirs(os.path.join(local_dir, "audio"), exist_ok=True)
    os.makedirs(os.path.join(local_dir, "transcripts"), exist_ok=True)
    os.makedirs(os.path.join(local_dir, "srt"), exist_ok=True)
    os.makedirs(os.path.join(local_dir, "metadata"), exist_ok=True)
    return local_dir

def download_youtube_audio(url, output_dir):
    """YouTube'dan ses indirme"""
    try:
        ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{'key': 'FFmpegExtractAudio', 'preferredcodec': 'mp3', 'preferredquality': '192'}],
            'outtmpl': os.path.join(output_dir, '%(title)s.%(ext)s'),
            'quiet': True,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            filename = ydl.prepare_filename(info).replace('.webm', '.mp3').replace('.m4a', '.mp3')
            filepath = os.path.join(output_dir, f"{clean_filename(info['title'])}.mp3")
            if os.path.exists(filename) and filename != filepath:
                os.rename(filename, filepath)
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
    """Whisper ile transkripsiyon (GPU optimize)"""
    try:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Transkripsiyon için dosya bulunamadı: {filepath}")
        
        transcriber = pipeline(
            "automatic-speech-recognition",
            model="openai/whisper-large-v2",
            device=device,
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        result = transcriber(
            filepath,
            chunk_length_s=30,
            generate_kwargs={"language": language, "task": "transcribe", "return_timestamps": True},
            batch_size=8
        )
        return result["text"], result.get("chunks", [])
    except Exception as e:
        print(f"Transkripsiyon hatası: {e}")
        return None, None

def tokenize_text(transcript):
    """BERT tokenizer ile tokenizasyon"""
    return tokenizer.tokenize(transcript)

def generate_keywords(transcript, num_keywords=5):
    """Anahtar kelime çıkarma (KeyBERT + yedek frekans analizi)"""
    try:
        keywords = kw_model.extract_keywords(
            transcript,
            keyphrase_ngram_range=(1, 2),
            stop_words=None,
            top_n=num_keywords
        )
        keyword_list = [kw[0] for kw in keywords]
        if not keyword_list:
            raise ValueError("KeyBERT anahtar kelime çıkaramadı")
        return keyword_list
    except Exception as e:
        words = re.findall(r'\w+', transcript.lower())
        stop_words = {'bir', 've', 'bu', 'ile', 'da', 'de', 'mi', 'mu', 'ki', 'ne', 'için'}
        filtered_words = [word for word in words if word not in stop_words and len(word) > 2]
        word_counts = Counter(filtered_words)
        return [word for word, _ in word_counts.most_common(num_keywords)]

def create_srt(chunks, filename, full_text=None):
    """SRT dosyası oluşturma"""
    srt_content = ""
    if not chunks and full_text:
        srt_content = "1\n00:00:00,000 --> 00:00:30,000\n" + full_text.strip() + "\n\n"
    else:
        for i, chunk in enumerate(chunks):
            start = chunk["timestamp"][0]
            end = chunk["timestamp"][1]
            text = chunk["text"].strip()
            srt_content += f"{i+1}\n{format_time(start)} --> {format_time(end)}\n{text}\n\n"
    
    with open(filename, "w", encoding="utf-8") as f:
        f.write(srt_content)
    return filename

def format_time(seconds):
    """Saniyeyi SRT formatına çevirme"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    milliseconds = int((secs - int(secs)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{int(secs):02d},{milliseconds:03d}"

def save_metadata(metadata, filename):
    """Metadata kaydetme"""
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    return filename

def download_dataset_json(local_dir):
    """dataset.json'ı Hugging Face Hub'dan indir"""
    api = HfApi(token=HF_TOKEN)
    db_file = os.path.join(local_dir, "dataset.json")
    try:
        api.hf_hub_download(repo_id=f"{HF_USERNAME}/{REPO_NAME}", filename="dataset.json", local_dir=local_dir, repo_type="dataset")
        with open(db_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        return []

def update_dataset_json(local_dir, title, transcript, audio_path, metadata):
    """dataset.json'ı güncelle"""
    db_file = os.path.join(local_dir, "dataset.json")
    dataset = download_dataset_json(local_dir)
    
    tokens = tokenize_text(transcript)
    keywords = generate_keywords(transcript)
    
    new_entry = {
        "title": title,
        "transcripts": transcript,
        "tokens": tokens,
        "audio_path": audio_path[len(local_dir)+1:],
        "anahtar_kelimeler": keywords,
        "metadata": metadata
    }
    
    existing_urls = {entry["metadata"]["url"] for entry in dataset}
    if metadata["url"] not in existing_urls:
        dataset.append(new_entry)
    else:
        print(f"Video zaten mevcut: {metadata['url']}")
    
    with open(db_file, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    return dataset

def process_video(url, local_repo):
    """Ana işlem akışı"""
    try:
        print(f"\nİşleniyor: {url}")
        audio_path, metadata = download_youtube_audio(url, os.path.join(local_repo, "audio"))
        if not audio_path:
            return False
        
        audio_path = os.path.abspath(audio_path)
        transcript, chunks = transcribe_audio(audio_path)
        if not transcript:
            return False
        
        base_name = clean_filename(metadata["title"])
        srt_path = os.path.join(local_repo, "srt", f"{base_name}.srt")
        txt_path = os.path.join(local_repo, "transcripts", f"{base_name}.txt")
        metadata_path = os.path.join(local_repo, "metadata", f"{base_name}.json")
        
        create_srt(chunks, srt_path, full_text=transcript)
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(transcript)
        save_metadata(metadata, metadata_path)
        update_dataset_json(local_repo, metadata["title"], transcript, audio_path, metadata)
        
        api = HfApi(token=HF_TOKEN)
        for file_path in [audio_path, srt_path, txt_path, metadata_path, os.path.join(local_repo, "dataset.json")]:
            if os.path.exists(file_path):
                api.upload_file(
                    path_or_fileobj=file_path,
                    path_in_repo=file_path[len(local_repo)+1:],
                    repo_id=f"{HF_USERNAME}/{REPO_NAME}",
                    repo_type="dataset",
                    commit_message=f"Add: {metadata['title']}"
                )
        
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
    
    local_repo = setup_huggingface_repo()
    if local_repo is None:
        return
    
    urls = [
        "https://www.youtube.com/watch?v=s4ujNtaKKEc",
        "https://www.youtube.com/watch?v=XFRkvFk58jc",
    ]
    
    for url in urls:
        process_video(url, local_repo)
    
    print("\nTüm işlemler tamamlandı!")

if __name__ == "__main__":
    main()