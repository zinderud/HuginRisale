import os
import json
import argparse
import hashlib
from pathlib import Path
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_model_and_tokenizer(model_name):
    """LLM modeli ve tokenizer'ı yükler."""
    print(f"Model yükleniyor: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True
    )
    return model, tokenizer

def generate_text(model, tokenizer, prompt, max_length=512):
    """Belirtilen prompt için model çıktısı üretir."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
            max_length=max_length,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Prompt'u çıktıdan çıkar
    response = response[len(tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)):]
    return response.strip()

def extract_metadata_with_llm(model, tokenizer, content, file_path):
    """LLM kullanarak metinden metadata çıkarır."""
    # Başlık extraction
    title_prompt = f"""
    Aşağıdaki metinden uygun bir başlık çıkar. Başlık kısa ve öz olmalı.
    
    Metin:
    {content[:1000]}
    
    Başlık:
    """
    title = generate_text(model, tokenizer, title_prompt, max_length=50)
    
    # Anahtar kelimeler extraction
    keywords_prompt = f"""
    Aşağıdaki metinden 5-10 anahtar kelime çıkar. Anahtar kelimeler virgülle ayrılmış olmalı.
    
    Metin:
    {content[:2000]}
    
    Anahtar Kelimeler:
    """
    keywords_str = generate_text(model, tokenizer, keywords_prompt, max_length=100)
    keywords = [k.strip() for k in keywords_str.split(",") if k.strip()]
    
    # Özet extraction
    summary_prompt = f"""
    Aşağıdaki metni 3-5 cümleyle özetle.
    
    Metin:
    {content[:3000]}
    
    Özet:
    """
    summary = generate_text(model, tokenizer, summary_prompt, max_length=200)
    
    # Konu/Kategori extraction
    category_prompt = f"""
    Aşağıdaki metnin hangi kategoriye ait olduğunu belirle. Örneğin: Teoloji, Felsefe, Etik, İman, İbadet, vb.
    
    Metin:
    {content[:2000]}
    
    Kategori:
    """
    category = generate_text(model, tokenizer, category_prompt, max_length=50)
    
    # Zorluk seviyesi extraction
    difficulty_prompt = f"""
    Aşağıdaki metnin okuyucu için zorluk seviyesini belirle: Temel, Orta, İleri.
    
    Metin:
    {content[:1500]}
    
    Zorluk Seviyesi:
    """
    difficulty = generate_text(model, tokenizer, difficulty_prompt, max_length=30)
    
    # Soru-cevap çiftleri oluşturma
    qa_prompt = f"""
    Aşağıdaki metne dayalı olarak 2 soru ve detaylı cevaplar hazırla. Her soru ve cevap çifti "Soru:" ve "Cevap:" ile başlamalı.
    
    Metin:
    {content[:4000]}
    
    Soru-Cevap Çiftleri:
    """
    qa_pairs_text = generate_text(model, tokenizer, qa_prompt, max_length=500)
    
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

def process_file_with_llm(file_path, model, tokenizer):
    """Dosyayı LLM ile işler ve zenginleştirilmiş veriler döndürür."""
    # Dosyayı oku
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # LLM ile metadata çıkar
    metadata = extract_metadata_with_llm(model, tokenizer, content, file_path)
    
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
            "text": content,
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

def process_directory_with_llm(dir_path, output_path, model_name):
    """Klasördeki tüm dosyaları LLM ile işler ve veri seti dosyası oluşturur."""
    # Model ve tokenizer'ı yükle
    model, tokenizer = load_model_and_tokenizer(model_name)
    
    all_data = []
    
    # Klasördeki tüm dosyaları işle
    for root, _, files in os.walk(dir_path):
        for file in tqdm(files, desc="Dosyalar işleniyor"):
            # Sadece .md ve .txt dosyalarını işle
            if file.endswith(('.md', '.txt')):
                file_path = os.path.join(root, file)
                try:
                    file_data = process_file_with_llm(file_path, model, tokenizer)
                    all_data.extend(file_data)
                    print(f"İşlendi: {file_path}")
                except Exception as e:
                    print(f"Hata ({file_path}): {str(e)}")
    
    # Sonuçları JSON dosyasına yaz
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

def main():
    parser = argparse.ArgumentParser(description='LLM kullanarak GitHub içeriklerini zenginleştirir ve veri seti oluşturur.')
    parser.add_argument('--input', type=str, default='content', help='İşlenecek içerik klasörü')
    parser.add_argument('--output', type=str, default='dataset', help='Çıktı veri seti klasörü')
    parser.add_argument('--model', type=str, default='malhajar/Mistral-7B-Instruct-v0.2-turkish', help='Kullanılacak LLM modeli')
    
    args = parser.parse_args()
    
    # Çıktı klasörü yoksa oluştur
    os.makedirs(args.output, exist_ok=True)
    
    # Tam JSON çıktı dosyası yolu
    json_output = os.path.join(args.output, 'risale_dataset_llm.json')
    
    # Tüm dosyaları LLM ile işle ve JSON oluştur
    all_data = process_directory_with_llm(args.input, json_output, args.model)
    
    # HuggingFace formatına dönüştür (jsonl)
    hf_output = os.path.join(args.output, 'risale_dataset_llm.jsonl')
    convert_to_huggingface_dataset(json_output, hf_output)
    
    print("İşlem tamamlandı!")
    print(f"Veri seti dosyaları: {json_output} ve {hf_output}")
    print(f"Toplam {len(all_data)} veri öğesi oluşturuldu.")

if __name__ == "__main__":
    main()