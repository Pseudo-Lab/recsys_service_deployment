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
            var movie_title = input.closest(".movie").querySelector("h2").textContent;
            var page_url = window.location.href;
            var tabName = "별점 준 영화들";
            var movie_id = input.closest(".movie").getAttribute("dbid");

            $.ajax({
                type: "POST",
                url: "/log_star/",
                contentType: "application/json;charset=utf-8",
                data: JSON.stringify({
                    'tab_name': '별점 준 영화들',
                    'movie_title': movie_title,
                    'page_url': page_url,
                    'percentage': percentage,
                    'movie_id': movie_id,
                }),
                dataType: "json",

                success: function (response) {
                    $(".clicked_movies").html(tabName);
                    var watchedMovieList = response.watched_movie;

                    $(".watched_list").empty();

                    for (var i = 0; i < watchedMovieList.length; i++) {
                        var wm = watchedMovieList[i];
                        var starRating = wm.star;

                        var movieHtml = `
                            <div class="watched_movie">
                                <div class="poster">
                                    <img src="${wm.posterUrl}" alt="">
                                </div>
                                <div class="title">
                                    <p>${wm.titleKo}</p>
                                    <p class="star">★ ${starRating}</p>
                                </div>
                            </div>
                        `;
                        $(".watched_list").append(movieHtml);
                    }
                    // scrollToBottom();
                }
            });
        });
    });
});

function scrollToBottom() {
    var watchedList = $(".watched_list");
    watchedList.scrollTop(watchedList.prop("scrollHeight"));
}
