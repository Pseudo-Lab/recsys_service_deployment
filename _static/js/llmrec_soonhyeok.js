
const chatHeader = document.querySelector('.chat-header');
const chatMessages = document.querySelector('.chat-messages');
const chatInputForm = document.querySelector('.chat-input-form');
const chatInput = document.querySelector('.chat-input');


const createChatMessageElement = (message) => {
    const serverIcon = '<img src="../../static/img/llm_icon/soonhyeok_llm_logo.png" alt="Server Icon" class="message-icon">';  // 아이콘 이미지
    if (message.text) {
        return `
        <div class="message ${message.sender === 'You' ? 'blue-bg' : 'gray-bg'}">
            ${message.sender !== 'You' ? `<div class="message-icon">${serverIcon}</div>` : ''}
            <div class="message-sender">${message.sender}</div>
            <div class="message-text">${message.text.replace(/\n/g, '<br>')}</div>
        </div>`;
    } else { 
        return `
        <div class="message ${message.sender === 'You' ? 'blue-bg' : 'gray-bg'}">
            ${message.sender !== 'You' ? `<div class="message-icon">${serverIcon}</div>` : ''}
            <div class="message-sender">${message.sender}</div>
        </div>`;
    }
};

// Automatically focus the chat input
chatInput.focus();

const renderMovieResults = (movies) => {
    let resultsHtml = '';
    movies.forEach(movie => {
        let ottHtml = '';
        
        // OTT 로고와 URL 렌더링
        for (const [ottName, ottUrl] of Object.entries(movie.link)) {
            if (ottName && ottUrl) { // ottName과 ottUrl이 모두 유효한 경우에만 처리
                ottHtml += `
                    <a href="${ottUrl}" target="_blank" class="ott-link">
                        <img src="../../static/img/llm_icon/ott_logo/${ottName}.png" alt="${ottName} Logo" class="ott-logo"/>
                    </a>
                `;
            }
        }
        resultsHtml += `
            <div class="movie-card">
                <img src="${movie.poster}" alt="Movie Poster" class="movie-poster"/>
                <div class="movie-title">${movie.title}</div>
                <div class="movie-details">
                    <p>평점: ${movie.rating}</p>
                    <p>감독: ${movie.director}</p>
                    <p>배우: ${movie.actors}</p>
                    <p>장르: ${movie.genre}</p>
                    <p>줄거리 요약: ${movie.summary}</p>
                    ${ottHtml ? `<div class="ott-links">${ottHtml}</div>` : ''} <!-- OTT 로고들 -->
                </div>
            </div>
        `;
    });
    chatMessages.innerHTML += resultsHtml;
};

const sendMessageToServer = (message) => {
    $.ajax({
        type: 'POST',
        url: window.location.pathname,
        contentType: 'application/json',
        data: JSON.stringify({ message : message }),
        success: function(response) {
            let senderName = '';
            if (response.status === 'success') {
                if (response.movies) {
                    const serverMessage = {
                        sender: response.intent,
                        timestamp: new Date().toLocaleString('en-US', { hour: 'numeric', minute: 'numeric', hour12: true }),
                        url: response.url  // 서버 응답에서 URL 추출
                    };
                    chatMessages.innerHTML += createChatMessageElement(serverMessage);
                    renderMovieResults(response.movies);
                } else {
                    const serverMessage = {
                        sender: senderName,
                        text: response.message,
                        timestamp: new Date().toLocaleString('en-US', { hour: 'numeric', minute: 'numeric', hour12: true }),
                        url: response.url  // 서버 응답에서 URL 추출
                    };
                    chatMessages.innerHTML += createChatMessageElement(serverMessage);
                };
                chatMessages.scrollTop = chatMessages.scrollHeight;  // 스크롤 조정
            } else {
                console.error('Error:', response.message);
            };

        },
        error: function(error) {
            console.error('Error sending message:', error.responseText);
        }
    });
};

const sendMessage = (e) => {
    e.preventDefault();
    const timestamp = new Date().toLocaleString('en-US', {hour: 'numeric', minute: 'numeric', hour12: true});
    const message = {
        sender: 'You',
        text: chatInput.value,
        timestamp,
    };
    sendMessageToServer(message);
    chatMessages.innerHTML += createChatMessageElement(message);
    chatInputForm.reset();
    chatMessages.scrollTop = chatMessages.scrollHeight;
};

chatInputForm.addEventListener('submit', sendMessage);
