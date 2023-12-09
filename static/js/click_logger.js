$(document).ready(function () {
    // 스크롤 항상 아래로 유지하는 함수
    function scrollToBottom() {
        var watchedList = $(".watched_list");
        watchedList.scrollTop(watchedList.prop("scrollHeight"));
    }

    // 수정: 'tab_name' 초기 값 설정
    var tabName = "클릭한 영화들";

    $("img").click(function () {
        var movie_title = $(this).next("h2").text();
        var page_url = window.location.href;

        // 수정: 'tab_name' 데이터 전달
        $.ajax({
            type: "POST",
            url: "/log_click/",
            contentType: "application/json;charset=utf-8",
            data: JSON.stringify({
                'movie_title': movie_title,
                'page_url': page_url
            }),
            dataType: "json",
            success: function (response) {
                // 클릭된 영화들 표시
                $(".clicked_movies").html(tabName);

                // 수정: 스크롤 항상 아래로 유지
                var watchedMovieList = response.watched_movie;
                var watchedList = $(".watched_list");
                watchedList.empty();
                for (var i = 0; i < watchedMovieList.length; i++) {
                    watchedList.append("<p>" + watchedMovieList[i] + "</p>");
                }
                scrollToBottom();
            }
        });
    });
});
