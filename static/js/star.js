const rating_input = document.querySelector('.rating input');
const rating_star = document.querySelector('.rating_star');

// 별점 드래그 할 때
rating_input.addEventListener('input', () => {
  rating_star.style.width = `${rating_input.value * 10}%`;
});