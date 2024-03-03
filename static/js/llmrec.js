const chatHeader = document.querySelector('.chat-header')
const chatMessages = document.querySelector('.chat-messages')
const chatInputForm = document.querySelector('.chat-input-form')
const chatInput = document.querySelector('.chat-input')
const clearChatBtn = document.querySelector('.clear-chat-button')

const createChatMessageElement = (message) => `
<div class="message ${message.sender == 'You' ? 'blue-bg' : 'gray-bg'}">
    <div class="message-sender">${message.sender}</div>
    <div class="message-text">${message.text}</div>
    <div class="message-timestamp">${message.timestamp}</div>
</div>
`
let messageSender = 'You'

chatInput.focus()

// const updateMessageSender = (name) => {
//     messageSender = name
//     chatHeader.innerText = `${messageSender} chatting...`
//     chatInput.placeholder = `Type here, ${messageSender}`
// }

// const sendMessage = (e) => {
//     e.preventDefault()
//
//     const timestamp = new Date().toLocaleString('en-US', {hour: 'numeric', minute: 'numeric', hour12: true})
//     const message = {
//         sender: messageSender,
//         text: chatInput.value,
//         timestamp,
//     }
//
//     chatMessages.innerHTML += createChatMessageElement(message)
//
//     chatInputForm.reset()
//     chatMessages.scrollTop = chatMessages.scrollHeight
// }




// chat.js 파일에 이 코드를 추가합니다.
const requestURL = window.location.pathname;
const sendMessageToServer = (message) => {
    $.ajax({
        type: 'POST',
        url: requestURL,
        contentType: 'application/json',
        data: JSON.stringify({ message: message }),
        success: function(response) {
            // 서버에서의 성공 응답 처리
            console.log(response.status, response.message);

            // 채팅창에 서버 응답을 추가
            const serverMessage = {
                sender: 'PseudoRec GPT',
                text: response.message,
                timestamp: new Date().toLocaleString('en-US', { hour: 'numeric', minute: 'numeric', hour12: true })
            };
            chatMessages.innerHTML += createChatMessageElement(serverMessage);
            chatMessages.scrollTop = chatMessages.scrollHeight;
        },
        error: function(error) {
            // 서버에서의 에러 응답 처리
            console.error('Error sending message:', error.responseText);
        }
    });
};



const sendMessage = (e) => {
    e.preventDefault();

    const timestamp = new Date().toLocaleString('en-US', {hour: 'numeric', minute: 'numeric', hour12: true});
    const message = {
        sender: messageSender,
        text: chatInput.value,
        timestamp,
    };

    // 채팅 메시지를 서버로 전송합니다.
    sendMessageToServer(message);

    // 화면에 메시지 추가 및 리셋 코드는 그대로 유지합니다.
    chatMessages.innerHTML += createChatMessageElement(message);
    chatInputForm.reset();
    chatMessages.scrollTop = chatMessages.scrollHeight;
};


chatInputForm.addEventListener('submit', sendMessage);











