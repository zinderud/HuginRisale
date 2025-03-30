// YouTube Arama Sonuçları Link Çıkarıcı
// Bu script YouTube'da arama sonuçlarındaki tüm videoların linklerini çıkarır

// Tarayıcı konsolunda çalıştırmak için:
// 1. YouTube'da bir arama yapın
// 2. F12 tuşuna basarak geliştirici konsolunu açın
// 3. Bu kodu konsola yapıştırın ve Enter tuşuna basın

(function extractYouTubeSearchLinks() {
    // Sayfa YouTube arama sonuçları sayfası mı kontrol et
    if (!window.location.href.includes('youtube.com/results')) {
        console.error('Bu bir YouTube arama sonuçları sayfası değil. Lütfen bir arama yapın ve sonuç sayfasında çalıştırın.');
        return;
    }

    // Sayfayı tam yüklemek için biraz aşağı kaydır
    function scrollDown() {
        return new Promise(resolve => {
            const maxScrolls = 5; // İstediğiniz kadar değiştirebilirsiniz
            let scrollCount = 0;
            
            const scrollInterval = setInterval(() => {
                window.scrollBy(0, 1000);
                scrollCount++;
                
                if (scrollCount >= maxScrolls) {
                    clearInterval(scrollInterval);
                    setTimeout(resolve, 1000); // Son yüklemelerin tamamlanması için bekle
                }
            }, 1500);
        });
    }

    // Ana işlev
    async function extractLinks() {
        console.log('Daha fazla video yüklemek için sayfa aşağı kaydırılıyor...');
        await scrollDown();
        
        // Arama sonuçlarındaki video öğelerini seç
        // YouTube'un farklı seçicileri olabileceğinden birkaç yaygın seçici deniyoruz
        let videoElements = document.querySelectorAll('a#video-title');
        
        if (videoElements.length === 0) {
            videoElements = document.querySelectorAll('a.ytd-video-renderer');
        }
        
        if (videoElements.length === 0) {
            videoElements = document.querySelectorAll('a.yt-simple-endpoint.ytd-video-renderer');
        }
        
        if (videoElements.length === 0) {
            console.error('Video öğeleri bulunamadı. YouTube arayüzü değişmiş olabilir veya sayfa tamamen yüklenmemiş olabilir.');
            return;
        }

        console.log(`Arama sonucunda ${videoElements.length} video bulundu.`);
        console.log('Video linkleri:');
        console.log('-----------------');

        // Her video için link oluştur ve yazdır
        const links = Array.from(videoElements).map(element => {
            // Href özelliğinden video ID'sini çıkar
            const href = element.href || '';
            let videoId = '';
            let fullLink = '';
            
            if (href.includes('watch?v=')) {
                videoId = new URLSearchParams(href.split('?')[1]).get('v');
                fullLink = `https://www.youtube.com/watch?v=${videoId}`;
            } else if (href.includes('/watch/')) {
                // Alternatif URL formatı
                const parts = href.split('/watch/');
                videoId = parts[1];
                fullLink = `https://www.youtube.com/watch?v=${videoId}`;
            } else {
                // Doğrudan href'i kullan
                fullLink = href;
            }
            
            const videoTitle = element.title || element.getAttribute('title') || 'Başlık yok';
            
            // Sadece geçerli video linklerini dahil et
            if (fullLink && fullLink.includes('youtube.com/watch')) {
                console.log(`${videoTitle}: ${fullLink}`);
                return { title: videoTitle, link: fullLink };
            }
            return null;
        }).filter(item => item !== null);

        // Tekrarlanan linkleri kaldır
        const uniqueLinks = [];
        const uniqueUrls = new Set();
        
        links.forEach(item => {
            if (!uniqueUrls.has(item.link)) {
                uniqueUrls.add(item.link);
                uniqueLinks.push(item);
            }
        });

        // Tüm linkleri birleştir (kopyalamayı kolaylaştırmak için)
        const allLinks = uniqueLinks.map(item => item.link).join('\n');
        console.log('-----------------');
        console.log('Tüm linkler (kopyalamak için):');
        console.log(allLinks);
        
        // Opsiyonel: linkleri bir dosyaya kaydetme yöntemi
        const downloadLinks = () => {
            const blob = new Blob([allLinks], { type: 'text/plain' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'youtube_search_links.txt';
            a.click();
            URL.revokeObjectURL(url);
        };
        
        console.log('Linkleri dosya olarak indirmek için konsola "downloadLinks()" yazın.');
        window.downloadLinks = downloadLinks;
        
        return uniqueLinks; // Sonuçları döndür
    }
    
    // İşlevi çalıştır
    return extractLinks();
})();