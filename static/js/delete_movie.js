// static/js/delete_movie.js
$(document).ready(function() {
    // Attach a click event to the delete button
    $('.delete_button').click(function() {
        var movieIndex = $(this).data('movie-index');

        // Send an AJAX request to delete the movie
        $.ajax({
            type: 'POST',
            url: '/delete_movie/', // Update with your URL endpoint for delete functionality
            data: JSON.stringify({ movie_index: movieIndex }) 
            ,
            success: function(data) {
                console.log("성공")}
        });
    });
}); 



// document.ready(function() {
//     $('.delete_button').click(function() {
//         var movieTitle = $(this).parent().text().trim(); // 클릭한 영화 제목 가져오기
//         $.ajax({
//             type: 'POST',
//             url: '/delete_movie/', // 삭제 요청을 처리할 Django URL
//             data: {
//                 'movie_title': movieTitle // 삭제할 영화 제목을 전송
//             },
//             success: function(response) {
//                 if (response.success) {
//                     // 성공적으로 삭제되었을 때의 작업
//                     $(this).parent().remove(); // HTML에서 삭제된 항목 제거
//                     alert('영화가 성공적으로 삭제되었습니다.');
//                 } else {
//                     alert('영화 삭제 중 오류가 발생했습니다.');
//                 }
//             },
//             error: function(xhr, errmsg, err) {
//                 alert('영화 삭제 중 오류가 발생했습니다.');
//             }
//         });
//     });
// });
