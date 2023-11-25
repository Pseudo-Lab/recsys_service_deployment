document.addEventListener('DOMContentLoaded', function () {
  // 페이지가 로드될 때 실행될 코드

  const rating_inputs = document.querySelectorAll('article input');
  const rating_stars = document.querySelectorAll('.rating_star');

  rating_inputs.forEach((input, index) => {
    // 각 input 태그에 대해 이벤트 리스너 추가
    input.addEventListener('input', () => {
      // 해당 input의 값을 기반으로 별점 변경
      const percentage = input.value * 10;
      rating_stars[index].style.width = `${percentage}%`;
    });
  });
});
