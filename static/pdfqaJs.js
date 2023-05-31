var endpoint = "ws://localhost:9000/pdfchat";
var ws = new WebSocket(endpoint);
// Receive message from server word by word. Display the words as they are received.
ws.onmessage = function (event) {
  //   var messages = document.getElementById("messages");
  console.log(event);
  var data = JSON.parse(event.data);
  if (data.sender === "bot") {
    if (data.type === "start") {
      //   var header = document.getElementById("header");
      //   header.innerHTML = "Computing answer...";
      // var div = document.createElement("div");
      // div.className = "server-message";
      // var p = document.createElement("p");
      // p.innerHTML = "<strong>" + "Output: " + "</strong>";
      console.log("Start", data.message);
      var message_section = document.querySelector(".message_section");
      message_section.innerHTML += `<div class="flex items-center justify-start">
            <div
              class="flex flex-col items-start justify-center text-white rounded-xl p-4 bg-[#3A3F47] rounded-tl-none"
            >
              <p class="message_bot"></p>
              
              <div class="flex gap-3 mt-3 message_box">
                </div>
            </div>
          </div>`;
    } else if (data.type === "stream") {
      //   var header = document.getElementById("header");
      //   header.innerHTML = "Output is typing...";
      var message_section = document.querySelector(".message_section");
      // var p = messages.lastChild.lastChild;
      //   var last = message_section.lastChild;
      var messages = document.querySelectorAll(".message_bot");
      var p = messages[messages.length - 1];
      console.log(p);
      if (data.message === "\n") {
        p.innerHTML += "<br>";
      } else {
        p.innerHTML += data.message;
      }
    } else if (data.type === "info") {
      //   var header = document.getElementById("header");
      //   header.innerHTML = data.message;
    } else if (data.type === "end") {
      //   var header = document.getElementById("header");
      //   header.innerHTML = "Ask a question";
      var message_box = document.querySelectorAll(".message_box");
      var last = message_box[message_box.length - 1];
      data.source_documents.forEach((element) => {
        //   `<button
        //   class="text-sm mt-2 ml-2 text-[#949494] hover:underline"
        //   onclick="showUploadPdfFiles(${element.page})"
        // >${element.page}</button>`
        let newElement = document.createElement("button");
        newElement.className =
          "text-sm mt-2 mr-2 text-[#949494] hover:underline";
        newElement.innerHTML = `#${element.page}`;
        newElement.addEventListener("click", function () {
          showUploadPdfFiles(element.page,element.source);
        });
        last.appendChild(newElement);
      });
      var button = document.getElementById("send");
      button.innerHTML = `<svg viewBox="0 0 24 24" class="w-5 h-5 fill-[#3A3F47]">
      <path
        d="M3.478 2.405a.75.75 0 00-.926.94l2.432 7.905H13.5a.75.75 0 010 1.5H4.984l-2.432 7.905a.75.75 0 00.926.94 60.519 60.519 0 0018.445-8.986.75.75 0 000-1.218A60.517 60.517 0 003.478 2.405z"
      />
    </svg>`;
      button.disabled = false;
    } else if (data.type === "error") {
      //   var header = document.getElementById("header");
      //   header.innerHTML = "Ask a question";
      var button = document.getElementById("send");
      button.innerHTML = `<svg viewBox="0 0 24 24" class="w-5 h-5 fill-[#3A3F47]">
      <path
        d="M3.478 2.405a.75.75 0 00-.926.94l2.432 7.905H13.5a.75.75 0 010 1.5H4.984l-2.432 7.905a.75.75 0 00.926.94 60.519 60.519 0 0018.445-8.986.75.75 0 000-1.218A60.517 60.517 0 003.478 2.405z"
      />
    </svg>`;
      button.disabled = false;
      var p = messages.lastChild.lastChild;
      p.innerHTML += data.message;
    }
  } else {
    // var div = document.createElement("div");
    // div.className = "client-message";
    // var p = document.createElement("p");
    // p.innerHTML = "<strong>" + "You: " + "</strong>";
    // p.innerHTML += data.message;
    // div.appendChild(p);
    // messages.appendChild(div);
    console.log("You: " + data.message);
    var message_section = document.querySelector(".message_section");
    message_section.innerHTML += `<div class="flex items-center justify-end">
    <div
      class="flex flex-col items-start justify-center text-white rounded-xl p-4 bg-[#8AA1FF] rounded-br-none"
    >
      <p>${data.message}</p> 
      
    </div>
  </div>`;
  }
  // Scroll to the bottom of the chat
  //   messages.scrollTop = messages.scrollHeight;
};

function sendMessage(event) {
  event.preventDefault();
  var message = document.getElementById("messageText").value;
  if (message === "") {
    return;
  }
  ws.send(message);
  document.getElementById("messageText").value = "";

  // Turn the button into a loading button
  var button = document.getElementById("send");
  button.innerHTML = "Loading...";
  button.disabled = true;
}
