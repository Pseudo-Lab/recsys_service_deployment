document.addEventListener("DOMContentLoaded", function() {
    fetch("/llmrec/get_initial_recommendation/")
        .then(response => response.json())
        .then(data => {
            const chatMessages = document.getElementById('chatMessages');
            if (chatMessages) {
                const message = {
                    sender: 'PseudoRec GPT',
                    text: data.message,
                    timestamp: new Date().toLocaleString('en-US', {hour: 'numeric', minute: 'numeric', hour12: true}),
                };
                chatMessages.innerHTML += createInitialRecommendationMessageElement(message);
                chatMessages.scrollTop = chatMessages.scrollHeight;
            } else {
                console.error("chatMessages element not found");
            }
        });
});

const createInitialRecommendationMessageElement = (message) => {
    return `
        <div class="message ${message.sender === 'You' ? 'blue-bg' : 'gray-bg'}">
            <div class="message-sender">${message.sender} (Initial Recommendation)</div>
            <div class="message-text">${message.text.replace(/\n/g, '<br>')}</div>
            <div class="message-timestamp">${message.timestamp}</div>
        </div>
    `;
};

const chatMessages = document.querySelector('.chat-messages');
const chatInputForm = document.querySelector('.chat-input-form');
const chatInput = document.querySelector('.chat-input');

const createChatMessageElement = (message) => {
    if (message.url === '/llmrec/hyeonwoo/') {
        const serverIcon = '<img src="../../static/img/llm_icon/hyeonwoo.png" alt="Server Icon" class="message-icon">';
        return `
        <div class="message ${message.sender === 'You' ? 'blue-bg' : 'gray-bg'}">
            ${message.sender !== 'You' ? `<div class="message-icon">${serverIcon}</div>` : ''}
            <div class="message-sender">${message.sender}</div>
            <div class="message-text">${message.text.replace(/\n/g, '<br>')}</div>
        </div>`;
    } else if (message.url === '/llmrec/gyungah/') {
        const serverIcon = '<img src="../../static/img/llm_icon/gyungah.png" alt="Server Icon" class="message-icon">';
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
if (chatInput) {
    chatInput.focus();
}

// Function to send a chat message to the server
const sendMessageToServer = (message) => {
    $.ajax({
        type: 'POST',
        url: window.location.pathname,
        contentType: 'application/json',
        data: JSON.stringify({ message: message }),
        success: function(response) {
            let senderName;
            if (response.url === '/llmrec/hyeonwoo/') {
                senderName = '코난';
            } else if (response.url === '/llmrec/gyungah/') {
                senderName = '장원영';
            } else {
                senderName = 'PseudoRec GPT';
            }

            const serverMessage = {
                sender: senderName,
                text: response.message,
                timestamp: new Date().toLocaleString('en-US', { hour: 'numeric', minute: 'numeric', hour12: true }),
                url: response.url
            };
            chatMessages.innerHTML += createChatMessageElement(serverMessage);
            chatMessages.scrollTop = chatMessages.scrollHeight;
        },
        error: function(error) {
            console.error('Error sending message:', error.responseText);
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
    sendMessageToServer(message);
    chatMessages.innerHTML += createChatMessageElement(message);
    chatInputForm.reset();
    chatMessages.scrollTop = chatMessages.scrollHeight;
};

// Add the event listener to the form
if (chatInputForm) {
    chatInputForm.addEventListener('submit', sendMessage);
}
