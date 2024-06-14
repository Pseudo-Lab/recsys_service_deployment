const chatHeader = document.querySelector('.chat-header');
const chatMessages = document.querySelector('.chat-messages');
const chatInputForm = document.querySelector('.chat-input-form');
const chatInput = document.querySelector('.chat-input');
const clearChatBtn = document.querySelector('.clear-chat-button');

const createChatMessageElement = (message) => {
    if (message.url === '/llmrec/hyeonwoo/') {
        const serverIcon = '<img src="../../static/img/llm_icon/hyeonwoo.jpeg" alt="Server Icon" class="message-icon">';  // 아이콘 이미지
        return `
        <div class="message ${message.sender === 'You' ? 'blue-bg' : 'gray-bg'}">
            ${message.sender !== 'You' ? `<div class="message-icon">${serverIcon}</div>` : ''}
            <div class="message-sender">${message.sender}</div>
            <div class="message-text">${message.text.replace(/\n/g, '<br>')}</div>
        </div>`;
    } else if (message.url === '/llmrec/gyungah/') {
        const serverIcon = '<img src="../../static/img/llm_icon/gyungah.jpeg" alt="Server Icon" class="message-icon">';  // 아이콘 이미지
        return `
        <div class="message ${message.sender === 'You' ? 'blue-bg' : 'gray-bg'}">
            ${message.sender !== 'You' ? `<div class="message-icon">${serverIcon}</div>` : ''}
            <div class="message-sender">${message.sender}</div>
            <div class="message-text">${message.text.replace(/\n/g, '<br>')}</div>
        </div>`;
    } else {
        return `
        <div class="message ${message.sender === 'You' ? 'blue-bg' : 'gray-bg'}">
            <div class="message-sender">${message.sender}</div>
            <div class="message-text">${message.text.replace(/\n/g, '<br>')}</div>
        </div>`;
    }
};

// Automatically focus the chat input
chatInput.focus();

// Function to send a chat message to the server
const sendMessageToServer = (message) => {
    $.ajax({
        type: 'POST',
        url: window.location.pathname,
        contentType: 'application/json',
        data: JSON.stringify({ message: message }),
        success: function(response) {
            // 성공 응답 처리, URL 포함
            let senderName;
            if (response.url === '/llmrec/hyeonwoo/') {
                senderName = '쿠도 신이치';
            } else if (response.url === '/llmrec/gyungah/') {
                senderName = '장원영';
            } else {
                senderName = 'PseudoRec GPT';
            }

            const serverMessage = {
                sender: senderName,
                text: response.message,
                timestamp: new Date().toLocaleString('en-US', { hour: 'numeric', minute: 'numeric', hour12: true }),
                url: response.url  // 서버 응답에서 URL 추출
            };
            chatMessages.innerHTML += createChatMessageElement(serverMessage);  // 메시지 요소 생성
            chatMessages.scrollTop = chatMessages.scrollHeight;  // 스크롤 조정
        },
        error: function(error) {
            console.error('Error sending message:', error.responseText);  // 오류 응답 처리
        }
    });
};


// Event handler for sending a message
const sendMessage = (e) => {
    e.preventDefault();

    const timestamp = new Date().toLocaleString('en-US', {hour: 'numeric', minute: 'numeric', hour12: true});
    const message = {
        sender: 'You',
        text: chatInput.value,
        timestamp,
    };

    // Send the message to the server
    sendMessageToServer(message);

    // Add message to the chat window and reset the input
    chatMessages.innerHTML += createChatMessageElement(message);
    chatInputForm.reset();
    chatMessages.scrollTop = chatMessages.scrollHeight;
};

// Add the event listener to the form
chatInputForm.addEventListener('submit', sendMessage);