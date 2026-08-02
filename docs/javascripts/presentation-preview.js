document.addEventListener('DOMContentLoaded', () => {
  const previews = document.querySelectorAll('.presentation-info__iframe[data-video-id][data-start-times]');

  previews.forEach((preview) => {
    const videoId = preview.dataset.videoId;
    const startTimes = preview.dataset.startTimes
      .split(',')
      .map((time) => Number.parseInt(time.trim(), 10))
      .filter(Number.isFinite);

    if (!videoId || startTimes.length === 0) {
      return;
    }

    const start = startTimes[Math.floor(Math.random() * startTimes.length)];
    const params = new URLSearchParams({
      autoplay: '1',
      mute: '1',
      controls: '0',
      loop: '1',
      playlist: videoId,
      modestbranding: '1',
      playsinline: '1',
      rel: '0',
      start: String(start),
    });

    preview.src = `https://www.youtube-nocookie.com/embed/${videoId}?${params.toString()}`;
  });
});
