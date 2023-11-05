// static/js/click_logger.js
$(document).ready(function(){
    $("img").click(function(){
        var movie_title = $(this).next("h2").text();
        $.ajax({
            type: "POST",
            url: "/log_click/",
            contentType: "application/json;charset=utf-8",
            data: JSON.stringify({
                'movie_title': movie_title,
            }),
            dataType: "json",
            success: function(data) {
                console.log("Click logged successfully");
            }
        });
    });
});