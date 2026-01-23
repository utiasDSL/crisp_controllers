// Carousel functionality for Zensical/MkDocs Material
// Uses document$ observable for instant navigation support

document$.subscribe(function() {
  // Initialize carousel on page load
  initializeCarousel();
});

function initializeCarousel() {
  const carouselContainer = document.querySelector('.carousel-container');

  // Only run if carousel exists on the page
  if (!carouselContainer) {
    return;
  }

  const carousel = carouselContainer.querySelector('.carousel');
  const slides = Array.from(carouselContainer.querySelectorAll('.carousel-slide'));
  const prevBtn = carouselContainer.querySelector('.carousel-btn-prev');
  const nextBtn = carouselContainer.querySelector('.carousel-btn-next');
  const indicatorsContainer = carouselContainer.querySelector('.carousel-indicators');

  if (slides.length === 0) {
    console.warn('No carousel slides found');
    return;
  }

  let currentSlide = 0;

  // Create indicator dots
  slides.forEach((_, index) => {
    const indicator = document.createElement('button');
    indicator.classList.add('carousel-indicator');
    indicator.setAttribute('aria-label', `Go to slide ${index + 1}`);
    if (index === 0) {
      indicator.classList.add('active');
    }
    indicator.addEventListener('click', () => goToSlide(index));
    indicatorsContainer.appendChild(indicator);
  });

  const indicators = Array.from(indicatorsContainer.querySelectorAll('.carousel-indicator'));

  // Show specific slide
  function goToSlide(index) {
    // Remove all position classes from all slides
    slides.forEach(slide => {
      slide.classList.remove('active', 'prev', 'next');
    });

    // Remove active indicator
    indicators[currentSlide].classList.remove('active');

    // Update current slide index
    currentSlide = index;

    // Calculate prev and next indices with wrapping
    const prevIndex = (currentSlide - 1 + slides.length) % slides.length;
    const nextIndex = (currentSlide + 1) % slides.length;

    // Add position classes
    slides[currentSlide].classList.add('active');
    slides[prevIndex].classList.add('prev');
    slides[nextIndex].classList.add('next');

    // Update active indicator
    indicators[currentSlide].classList.add('active');
  }

  // Next slide
  function nextSlide() {
    const nextIndex = (currentSlide + 1) % slides.length;
    goToSlide(nextIndex);
  }

  // Previous slide
  function prevSlide() {
    const prevIndex = (currentSlide - 1 + slides.length) % slides.length;
    goToSlide(prevIndex);
  }

  // Event listeners
  if (prevBtn) {
    prevBtn.addEventListener('click', prevSlide);
  }

  if (nextBtn) {
    nextBtn.addEventListener('click', nextSlide);
  }

  // Keyboard navigation
  document.addEventListener('keydown', function(e) {
    // Only handle keyboard events if carousel is on the current page
    if (!document.querySelector('.carousel-container')) {
      return;
    }

    if (e.key === 'ArrowLeft') {
      prevSlide();
    } else if (e.key === 'ArrowRight') {
      nextSlide();
    }
  });

  // Optional: Auto-advance carousel (uncomment to enable)
  // let autoAdvanceInterval;
  // function startAutoAdvance() {
  //   autoAdvanceInterval = setInterval(nextSlide, 5000); // 5 seconds
  // }
  // function stopAutoAdvance() {
  //   clearInterval(autoAdvanceInterval);
  // }
  //
  // startAutoAdvance();
  //
  // // Pause auto-advance on hover
  // carousel.addEventListener('mouseenter', stopAutoAdvance);
  // carousel.addEventListener('mouseleave', startAutoAdvance);

  // Touch/swipe support for mobile
  let touchStartX = 0;
  let touchEndX = 0;

  carousel.addEventListener('touchstart', function(e) {
    touchStartX = e.changedTouches[0].screenX;
  }, false);

  carousel.addEventListener('touchend', function(e) {
    touchEndX = e.changedTouches[0].screenX;
    handleSwipe();
  }, false);

  function handleSwipe() {
    const swipeThreshold = 50; // minimum distance for swipe
    const diff = touchStartX - touchEndX;

    if (Math.abs(diff) > swipeThreshold) {
      if (diff > 0) {
        // Swiped left - show next slide
        nextSlide();
      } else {
        // Swiped right - show previous slide
        prevSlide();
      }
    }
  }

  console.log('Carousel initialized with', slides.length, 'slides');

  // Initialize carousel layout with first slide
  goToSlide(0);
}
