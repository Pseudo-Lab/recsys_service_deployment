document.addEventListener('DOMContentLoaded', function() {
    var deleteButtons = document.querySelectorAll('.delete-button');

    deleteButtons.forEach(function(button) {
        button.addEventListener('click', function(event) {
            var timestamp = button.closest('.watched_movie').getAttribute('data-timestamp');
            var movieId = button.closest('.watched_movie').getAttribute('data-movie-id');

            // fetch() 함수를 사용하여 데이터를 서버로 전송
            fetch('/movie_recommendation/delete_movie_interaction/', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ timestamp: timestamp, movieId: movieId }),
            })
            .then(response => {
                if (!response.ok) {
                    throw new Error('Network response was not ok');
                }
                return response.json();
            })
            .then(data => {
                // 성공적으로 처리된 경우, 여기에 추가적인 작업을 할 수 있습니다.
                console.log('삭제 작업이 완료되었습니다.');
                // 삭제된 영화를 화면에서 제거하는 부분
                button.closest('.watched_movie').remove();
            })
            .catch(error => {
                console.error('Error:', error);
            });

            // 클릭 이벤트의 전파를 막습니다.
            event.stopPropagation();
        });
    });
});


document.addEventListener('DOMContentLoaded', function() {
    document.getElementById('delete-all-history').addEventListener('click', function() {
        if (confirm('정말로 모든 기록을 삭제하시겠습니까?')) {
            // AJAX 요청
            fetch('/movie_recommendation/delete_all_interactions/', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({}),
            })
            .then(response => {
                if (!response.ok) {
                    throw new Error('Network response was not ok');
                }
                return response.json();
            })
            .then(data => {
                // 성공적으로 처리된 경우, 여기에 추가적인 작업을 할 수 있습니다.
                console.log('모든 기록 삭제 완료:', data);
                // 페이지 리로드 또는 필요한 다른 작업을 수행할 수 있습니다.
                location.reload(); // 예시로 페이지를 리로드하는 방법
            })
            .catch(error => {
                console.error('Error:', error);
            });
        }
    });
});