// YouTube Playlist Link Extractor
// Bu script bir YouTube oynatma listesindeki tüm videoların linklerini çıkarır

// Tarayıcı konsolunda çalıştırmak için:
// 1. YouTube oynatma listesi sayfasını açın
// 2. F12 tuşuna basarak geliştirici konsolunu açın
// 3. Bu kodu konsola yapıştırın ve Enter tuşuna basın

(function extractYouTubePlaylistLinks() {
    // Sayfa YouTube oynatma listesi sayfası mı kontrol et
    if (!window.location.href.includes('youtube.com/playlist')) {
        console.error('Bu bir YouTube oynatma listesi sayfası değil. Lütfen bir oynatma listesi sayfasında çalıştırın.');
        return;
    }

    // Tüm video öğelerini seç
    const videoElements = document.querySelectorAll('a#video-title');
    
    if (videoElements.length === 0) {
        console.error('Video öğeleri bulunamadı. Sayfa tamamen yüklendikten sonra tekrar deneyin.');
        return;
    }

    console.log(`Oynatma listesinde ${videoElements.length} video bulundu.`);
    console.log('Video linkleri:');
    console.log('-----------------');

    // Her video için link oluştur ve yazdır
    const links = Array.from(videoElements).map(element => {
        const videoId = new URLSearchParams(element.href.split('?')[1]).get('v');
        const videoTitle = element.title || 'Başlık yok';
        const fullLink = `https://www.youtube.com/watch?v=${videoId}`;
        console.log(`${videoTitle}: ${fullLink}`);
        return { title: videoTitle, link: fullLink };
    });

    // Tüm linkleri birleştir (kopyalamayı kolaylaştırmak için)
    const allLinks = links.map(item => item.link).join('\n');
    console.log('-----------------');
    console.log('Tüm linkler (kopyalamak için):');
    console.log(allLinks);
    
    // Opsiyonel: linkleri bir dosyaya kaydetme yöntemi
    const downloadLinks = () => {
        const blob = new Blob([allLinks], { type: 'text/plain' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'youtube_playlist_links.txt';
        a.click();
        URL.revokeObjectURL(url);
    };
    
    console.log('Linkleri dosya olarak indirmek için konsola "downloadLinks()" yazın.');
    window.downloadLinks = downloadLinks;
    
    return links; // Sonuçları döndür
})();