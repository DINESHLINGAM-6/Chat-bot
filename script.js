document.getElementById("send-btn").addEventListener("click", () => {
    const userInput = document.getElementById("user-input").value;
    if (userInput) {
        const chatBox = document.getElementById("chat-box");
        const userMessage = `<div class='user-message'>${userInput}</div>`;
        chatBox.innerHTML += userMessage;
        fetch('/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ message: userInput })
        })
        .then(response => response.json())
        .then(data => {
            const botMessage = `<div class='bot-message'>${data.response}</div>`;
            chatBox.innerHTML += botMessage;
            chatBox.scrollTop = chatBox.scrollHeight;
        });
        document.getElementById("user-input").value = '';
    }
});