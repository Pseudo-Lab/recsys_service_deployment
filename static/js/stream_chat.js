const chatMessages = document.querySelector('.chat-messages');
const chatInputForm = document.querySelector('.chat-input-form');

chatInputForm.addEventListener('submit', function (event) {
    event.preventDefault(); // 폼 제출 기본 동작을 막음
    main();
});
function userMessage(input) {
    const timestamp = new Date().toLocaleString('en-US', {hour: 'numeric', minute: 'numeric', hour12: true});
    const message = {
        sender: 'You',
        text: input,
        timestamp,
    };
    chatMessages.innerHTML += createChatMessageElement("human", message);
}

function sendMessage(input){
    const queryParams = new URLSearchParams({ text: input }).toString();
    const url = '/llmrec/stream_chat?' + queryParams;

    chatMessages.innerHTML += createChatMessageElement("ai", {});
    const eventSource = new EventSource(url);
    eventSource.onmessage = function(event) {
        const data = JSON.parse(event.data);
        chatMessages.innerHTML += data;
    };

    eventSource.onerror = function(error) {
        eventSource.close();
        resetButtons();
    };
}

function setButtons() {
    document.getElementById('send-button').style.display = 'none';
    document.getElementById('stop-button').style.display = 'inline';
}

function resetButtons() {
    document.getElementById('send-button').style.display = 'inline';
    document.getElementById('stop-button').style.display = 'none';
}
const createChatMessageElement = (type, message) => {
    if (type === "human"){
        return `
            <div class="message-human">
                <div class="message-sender">${message.sender}</div>
                <div class="message-text">${message.text.replace(/\n/g, '<br>')}</div>
            </div>`;
        }
    else if (type === "ai") {
        return `
            <div class="message-ai">
                <div class="message-sender">AI</div>
            </div>`;
    }
}


function main() {
    const chatInput = document.querySelector('.chat-input');
    const input = chatInput.value;
    if (!input) return;

    userMessage(input)
    setButtons();

    sendMessage(input)

    chatInput.value = '';
}
