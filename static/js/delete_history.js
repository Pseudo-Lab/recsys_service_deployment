$(document).ready(function() {
    var deleteButton = document.getElementById('delete-all-history');
    if (deleteButton) {
        deleteButton.addEventListener('click', function() {
            console.log('Delete history button clicked');
            $.ajax({
                type: "POST",
                url: "/movie_recommendation/delete_all_history/",
                contentType: "application/json;charset=utf-8",
                success: function(response) {
                    $(".watched_list").empty();
                    console.log('History deleted successfully');
                },
                error: function(xhr, status, error) {
                    console.error('Error occurred:', error);
                }
            });
        });
    } else {
        console.error('Delete button not found');
    }
});
