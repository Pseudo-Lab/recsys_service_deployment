$(document).ready(function () {
    $(".heart path").click(function () {
        let isOn = $(this).hasClass("on");
        if (isOn) {
            $(this).removeClass("on");
        } else {
            $(this).addClass("on");
        }
    });
});
