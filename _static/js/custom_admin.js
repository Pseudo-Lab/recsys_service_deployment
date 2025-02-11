document.addEventListener("DOMContentLoaded", function () {
    const initializePreview = () => {
        const textarea = document.querySelector("textarea[name='content']"); // 'content' 필드 대상
        const preview = document.getElementById("markdown-preview");

        if (textarea && preview) {
            let timeout = null;

            textarea.addEventListener("input", function () {
                clearTimeout(timeout); // 이전 타이머를 취소
                timeout = setTimeout(() => { // 300ms 디바운스
                    const content = textarea.value;

                    fetch("/archive/post_preview/", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({ content: content }),
                    })
                        .then((response) => response.json())
                        .then((data) => {
                            if (data.html) {
                                preview.innerHTML = data.html; // 미리보기 업데이트
                            } else {
                                preview.innerHTML = `<p>Error rendering preview.</p>`;
                            }
                        })
                        .catch((error) => {
                            console.error("Error fetching preview:", error);
                            preview.innerHTML = `<p>Error fetching preview.</p>`;
                        });
                }, 300);
            });
        }
    };

    // Admin의 동적 로딩 완료 후 초기화
    if (typeof django !== "undefined" && django.jQuery) {
        django.jQuery(document).on("formset:added", initializePreview); // Formset 추가 시
    }

    // 초기 실행
    initializePreview();
});
