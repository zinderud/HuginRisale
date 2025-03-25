#!/usr/bin/env python3

import os
import sys
from transformers import AutoModel, AutoTokenizer

# Gerekli kütüphanelerin kurulu olup olmadığını kontrol et ve kur
try:
    import torch
except ImportError:
    print("PyTorch bulunamadı, kuruluyor...")
    os.system("pip install torch")
    import torch

try:
    from transformers import AutoModel, AutoTokenizer
except ImportError:
    print("Transformers bulunamadı, kuruluyor...")
    os.system("pip install transformers")

# Model adı ve yerel kaydetme dizini
model_name = "zinderud/hugin-risale-finetuned"
save_directory = "./hugin-risale-finetuned"

# Modeli ve tokenizer'ı indirip yerel olarak kaydet
def download_and_save_model():
    print(f"{model_name} modeli indiriliyor ve {save_directory} dizinine kaydediliyor...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    
    # Yerel dizine kaydet
    tokenizer.save_pretrained(save_directory)
    model.save_pretrained(save_directory)
    print(f"Model ve tokenizer {save_directory} dizinine başarıyla kaydedildi!")

# Modeli yerel olarak yükle ve test et
def load_and_test_model():
    print("Model yerel olarak yükleniyor...")
    tokenizer = AutoTokenizer.from_pretrained(save_directory, local_files_only=True)
    model = AutoModel.from_pretrained(save_directory, local_files_only=True)

    # Örnek bir metin ile test
    text = "Bu bir test cümlesidir."
    inputs = tokenizer(text, return_tensors="pt")  # PyTorch tensor'ları döndürür
    outputs = model(**inputs)
    print("Model çıktısı:", outputs.last_hidden_state.shape)  # Örnek bir çıktı boyutu

# Ana fonksiyon
def main():
    # Model dosyaları yoksa indir
    if not os.path.exists(save_directory):
        download_and_save_model()
    else:
        print(f"{save_directory} zaten mevcut, indirme atlanıyor.")

    # Çevrimdışı modda çalışmasını sağla
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    print("Çevrimdışı mod etkinleştirildi.")

    # Modeli yükle ve test et
    load_and_test_model()

if __name__ == "__main__":
    main()