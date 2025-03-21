import os
import json
import argparse
import hashlib
from pathlib import Path
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

def create_pipeline(model_name, device_map="auto", torch_dtype=torch.float16):
    """Text generation pipeline oluşturur - hafıza optimizasyonu için."""
    print(f"Model yükleniyor: {model_name}")
    
    # Tokenizer'ı yükle
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Model yüklemek için pipeline kullan - daha iyi hafıza optimizasyonu
    pipe = pipeline(
        "text-generation",
        model=model_name,
        tokenizer=tokenizer,
        torch_dtype=torch_dtype,
        device_map=device_map,
        max_new_tokens=500,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
    )
    
    return pipe

def generate_text_with_pipeline(pipe, prompt, max_new_tokens=500):
    """Pipeline kullanarak metin üretir."""
    result = pipe(prompt, max_new_tokens=max_new_tokens)
    generated_text = result[0]['generated_text']
    
    # Prompt'u çıktıdan çıkar (prompt kendi başına dönerse)
    if generated_text.startswith(prompt):
        return generated_text[len(prompt):].strip()
    
    return generated_text.strip()

def extract_metadata_with_llm(pipe, content, file_path):
    """LLM kullanarak metinden metadata çıkarır - chunk'lar halinde."""
    # İçeriği daha küçük parçalara bölerek işleme
    content_preview = content[:3000]  # Özet için ilk 3000 karakter yeterli
    
    # Başlık extraction
    title_prompt = f"""
    Aşağıdaki metinden uygun bir başlık çıkar. Başlık kısa ve öz olmalı.
    
    Metin:
    {content_preview[:1000]}
    
    Başlık:
    """
    title = generate_text_with_pipeline(pipe, title_prompt, max_new_tokens=50)
    
    # Anahtar kelimeler extraction
    keywords_prompt = f"""
    Aşağıdaki metinden 5-10 anahtar kelime çıkar. Anahtar kelimeler virgülle ayrılmış olmalı.
    
    Metin:
    {content_preview[:2000]}
    
    Anahtar Kelimeler:
    """
    keywords_str = generate_text_with_pipeline(pipe, keywords_prompt, max_new_tokens=100)
    keywords = [k.strip() for k in keywords_str.split(",") if k.strip()]
    
    # Özet extraction
    summary_prompt = f"""
    Aşağıdaki metni 3-5 cümleyle özetle.
    
    Metin:
    {content_preview}
    
    Özet:
    """
    summary = generate_text_with_pipeline(pipe, summary_prompt, max_new_tokens=200)
    
    # Konu/Kategori extraction
    category_prompt = f"""
    Aşağıdaki metnin hangi kategoriye ait olduğunu belirle. Örneğin: Teoloji, Felsefe, Etik, İman, İbadet, vb.
    
    Metin:
    {content_preview[:2000]}
    
    Kategori:
    """
    category = generate_text_with_pipeline(pipe, category_prompt, max_new_tokens=50)
    
    # Zorluk seviyesi extraction
    difficulty_prompt = f"""
    Aşağıdaki metnin okuyucu için zorluk seviyesini belirle: Temel, Orta, İleri.
    
    Metin:
    {content_preview[:1500]}
    
    Zorluk Seviyesi:
    """
    difficulty = generate_text_with_pipeline(pipe, difficulty_prompt, max_new_tokens=30)
    
    # Soru-cevap çiftleri oluşturma - her bir soru için ayrı istek
    qa_prompts = [
        f"""
        Aşağıdaki metne dayalı olarak bir soru ve detaylı cevap hazırla. Önce soruyu yaz, sonra cevabı yaz.
        
        Metin:
        {content_preview}
        
        Soru ve Cevap (Soru: ile başla, Cevap: ile devam et):
        """,
        f"""
        Aşağıdaki metne dayalı olarak bir başka soru ve detaylı cevap hazırla. İlk soru-cevaptan farklı bir konu seç.
        
        Metin:
        {content_preview}
        
        Soru ve Cevap (Soru: ile başla, Cevap: ile devam et):
        """
    ]
    
    qa_pairs_text = ""
    for prompt in qa_prompts:
        qa_result = generate_text_with_pipeline(pipe, prompt, max_new_tokens=300)
        qa_pairs_text += qa_result + "\n\n"
    
    # Dosya yolundan bölüm/kitap bilgisi çıkarma
    parts = file_path.split(os.sep)
    if len(parts) >= 3:
        chapter = parts[-3] if parts[-3] != "content" else parts[-2]
        section = parts[-2]
    elif len(parts) == 2:
        chapter = parts[-2]
        section = "Genel"
    else:
        chapter = "Ana Bölüm"
        section = "Genel"
    
    chapter = chapter.replace('_', ' ').replace('-', ' ').title()
    section = section.replace('_', ' ').replace('-', ' ').title()
    
    # Sonuçları döndür
    return {
        "title": title,
        "keywords": keywords,
        "summary": summary,
        "category": category,
        "difficulty": difficulty,
        "qa_pairs": qa_pairs_text,
        "section": section,
        "chapter": chapter
    }

def parse_qa_pairs(qa_text):
    """Soru-cevap metni parçalar ve bir dizi soru-cevap çifti döndürür."""
    qa_pairs = []
    current_question = None
    current_answer = ""
    
    lines = qa_text.split("\n")
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        if line.startswith("Soru:"):
            # Yeni soru başladı, önceki soru-cevabı kaydet
            if current_question:
                qa_pairs.append({
                    "instruction": current_question,
                    "response": current_answer.strip()
                })
            current_question = line[5:].strip()
            current_answer = ""
        elif line.startswith("Cevap:"):
            current_answer = line[6:].strip()
        elif current_answer and line:
            current_answer += " " + line
    
    # Son soru-cevabı ekle
    if current_question:
        qa_pairs.append({
            "instruction": current_question,
            "response": current_answer.strip()
        })
    
    return qa_pairs

def process_file_with_llm(file_path, pipe, batch_size=1):
    """Dosyayı LLM ile işler ve zenginleştirilmiş veriler döndürür."""
    try:
        # Dosyayı oku
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # LLM ile metadata çıkar
        metadata = extract_metadata_with_llm(pipe, content, file_path)
        
        # Soru-cevap çiftlerini ayrıştır
        qa_pairs = parse_qa_pairs(metadata["qa_pairs"])
        
        # Eğer LLM soru-cevap çiftleri oluşturamadıysa, temel bir soru-cevap çifti ekle
        if not qa_pairs:
            qa_pairs = [{
                "instruction": f"{metadata['title']} konusu hakkında bilgi verir misiniz?",
                "response": content[:1000]  # İlk 1000 karakter
            }]
        
        # Sonuçları oluştur
        results = []
        
        # Her soru-cevap çifti için bir veri öğesi oluştur
        for i, qa_pair in enumerate(qa_pairs):
            # Benzersiz ID oluştur
            unique_id = hashlib.md5(f"{file_path}_{i}".encode()).hexdigest()
            
            # Veri öğesini oluştur
            data_item = {
                "id": unique_id,
                "title": metadata["title"],
                "text": content[:10000],  # Çok uzun metinleri kısalt
                "summary": metadata["summary"],
                "category": metadata["category"],
                "difficulty": metadata["difficulty"],
                "section": metadata["section"],
                "chapter": metadata["chapter"],
                "keywords": metadata["keywords"],
                "file_type": os.path.splitext(file_path)[1][1:],  # Uzantıyı al (.md -> md)
                "file_path": file_path,
                "instruction": qa_pair["instruction"],
                "response": qa_pair["response"]
            }
            
            results.append(data_item)
        
        return results
    except Exception as e:
        print(f"Hata ({file_path}): {str(e)}")
        return []

def process_directory_batch(dir_path, output_path, model_name, batch_size=10):
    """Klasördeki tüm dosyaları LLM ile batch olarak işler ve aralarda kaydeder."""
    # Pipeline oluştur
    pipe = create_pipeline(model_name)
    
    all_data = []
    file_paths = []
    
    # İşlenecek dosyaları topla
    for root, _, files in os.walk(dir_path):
        for file in files:
            if file.endswith(('.md', '.txt')):
                file_paths.append(os.path.join(root, file))
    
    # Batch olarak işle ve aralarda kaydet
    for i in tqdm(range(0, len(file_paths), batch_size), desc="Batch İşleniyor"):
        batch_files = file_paths[i:i+batch_size]
        batch_data = []
        
        for file_path in batch_files:
            try:
                file_data = process_file_with_llm(file_path, pipe)
                batch_data.extend(file_data)
                print(f"İşlendi: {file_path}")
            except Exception as e:
                print(f"Hata ({file_path}): {str(e)}")
        
        # Batch sonuçlarını toplam sonuçlara ekle
        all_data.extend(batch_data)
        
        # Ara kaydetme - her batch sonrası kaydet
        partial_output = os.path.join(os.path.dirname(output_path), f"partial_{i}.json")
        with open(partial_output, 'w', encoding='utf-8') as f:
            json.dump(batch_data, f, ensure_ascii=False, indent=2)
        
        print(f"Batch {i} kaydedildi: {partial_output}")
    
    # Tüm sonuçları birleştir ve kaydet
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2)
    
    print(f"Toplam {len(all_data)} veri öğesi oluşturuldu ve {output_path} dosyasına kaydedildi.")
    return all_data

def convert_to_huggingface_dataset(input_json, output_path):
    """JSON veri setini HuggingFace veri seti formatına dönüştürür."""
    # JSON dosyasını oku
    with open(input_json, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # HuggingFace formatına dönüştür (jsonl formatı)
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in data:
            # HuggingFace için basitleştirilmiş format
            hf_item = {
                "text": item["text"],
                "summary": item["summary"],
                "category": item["category"],
                "difficulty": item["difficulty"],
                "file_type": item["file_type"],
                "file_path": item["file_path"],
                "instruction": item["instruction"],
                "response": item["response"],
                "title": item["title"],
                "section": item["section"],
                "chapter": item["chapter"],
                "keywords": item["keywords"]
            }
            f.write(json.dumps(hf_item, ensure_ascii=False) + '\n')
    
    print(f"Veriler HuggingFace formatında {output_path} dosyasına kaydedildi.")

def combine_partial_files(output_dir, output_path):
    """Kısmi JSON dosyalarını birleştirir."""
    all_data = []
    
    # Tüm kısmi dosyaları bul
    partial_files = [f for f in os.listdir(output_dir) if f.startswith("partial_") and f.endswith(".json")]
    
    # Her dosyayı oku ve birleştir
    for file in partial_files:
        file_path = os.path.join(output_dir, file)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                all_data.extend(data)
            print(f"Birleştirildi: {file_path}")
        except Exception as e:
            print(f"Birleştirme hatası ({file_path}): {str(e)}")
    
    # Birleştirilmiş sonucu kaydet
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2)
    
    print(f"Toplam {len(all_data)} veri öğesi birleştirildi ve {output_path} dosyasına kaydedildi.")
    return all_data

def process_with_parameters(input_dir="content", 
                          output_dir="dataset", 
                          model_name="malhajar/Mistral-7B-Instruct-v0.2-turkish", 
                          batch_size=5,
                          combine_only=False):
    """Parametrelerle işlem yapan ana fonksiyon"""
    try:
        # Çıktı klasörü oluştur
        os.makedirs(output_dir, exist_ok=True)
        
        # Dosya yolları
        json_output = os.path.join(output_dir, 'risale_dataset_llm.json')
        huggingface_output = os.path.join(output_dir, 'risale_dataset_hf.jsonl')
        
        if combine_only:
            combine_partial_files(output_dir, json_output)
        else:
            process_directory_batch(input_dir, json_output, model_name, batch_size)
            
        convert_to_huggingface_dataset(json_output, huggingface_output)
        return True
        
    except Exception as e:
        print(f"Hata: {str(e)}")
        return False

def main():
    try:
        if 'google.colab' in str(get_ipython()):
            # Colab'da doğrudan process_with_parameters'ı kullan
            return process_with_parameters()
        else:
            # Normal Python ortamında argparse kullan
            parser = argparse.ArgumentParser(description='LLM kullanarak içerikleri zenginleştirir.')
            parser.add_argument('--input', type=str, default='content')
            parser.add_argument('--output', type=str, default='dataset')
            parser.add_argument('--model', type=str, default='malhajar/Mistral-7B-Instruct-v0.2-turkish')
            parser.add_argument('--batch_size', type=int, default=5)
            parser.add_argument('--combine_only', action='store_true')
            
            args = parser.parse_args()
            return process_with_parameters(
                args.input, 
                args.output, 
                args.model, 
                args.batch_size,
                args.combine_only
            )
            
    except Exception as e:
        print(f"Hata: {str(e)}")
        return False

if __name__ == "__main__":
    main()

# Kaggle için basitleştirilmiş çalıştırma fonksiyonu
def run_kaggle_process(input_dir="content", 
                      output_dir="dataset", 
                      model_name="malhajar/Mistral-7B-Instruct-v0.2-turkish", 
                      batch_size=5,
                      combine_only=False):
    """Kaggle için veri işleme süreci"""
    try:
        # Çıktı klasörü oluştur
        os.makedirs(output_dir, exist_ok=True)
        
        # Dosya yolları
        json_output = os.path.join(output_dir, 'risale_dataset_llm.json')
        huggingface_output = os.path.join(output_dir, 'risale_dataset_hf.jsonl')
        
        if combine_only:
            # Sadece kısmi dosyaları birleştir
            combine_partial_files(output_dir, json_output)
        else:
            # Tüm işlem akışını çalıştır
            process_directory_batch(input_dir, json_output, model_name, batch_size)
            
        # HuggingFace formatına dönüştür
        convert_to_huggingface_dataset(json_output, huggingface_output)
        
        print("İşlem başarıyla tamamlandı!")
        return True
        
    except Exception as e:
        print(f"Hata: {str(e)}")
        return False

# Eğer bu dosya doğrudan çalıştırılırsa
if __name__ == "__main__":
    # Argparse yerine basit bir yaklaşım
    # Bu kısmı Kaggle'da kullanmazsınız
    import sys
    
    if len(sys.argv) > 1:
        input_dir = sys.argv[1] if len(sys.argv) > 1 else "content"
        output_dir = sys.argv[2] if len(sys.argv) > 2 else "dataset"
        model_name = sys.argv[3] if len(sys.argv) > 3 else "malhajar/Mistral-7B-Instruct-v0.2-turkish"
        batch_size = int(sys.argv[4]) if len(sys.argv) > 4 else 5
        combine_only = True if len(sys.argv) > 5 and sys.argv[5].lower() == "true" else False
        
        run_kaggle_process(input_dir, output_dir, model_name, batch_size, combine_only)
    else:
        # Varsayılan değerlerle çalıştır
        run_kaggle_process()

# Kaggle notebook için kullanım:
"""
# GitHub'dan içeriği indir
!git clone https://github.com/zinderud/HuginRisale.git

# Gerekli kütüphaneleri yükle
!pip install transformers torch tqdm

# Koddan doğrudan process_directory_batch ve convert_to_huggingface_dataset fonksiyonlarını çağır
from llmuse1 import run_kaggle_process

# İşlemi başlat
run_kaggle_process(
    input_dir="HuginRisale/content", 
    output_dir="dataset", 
    model_name="malhajar/Mistral-7B-Instruct-v0.2-turkish", 
    batch_size=5
)
"""