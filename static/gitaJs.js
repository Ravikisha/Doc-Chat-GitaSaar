var endpoint = "ws://localhost:9000/chat";
var ws = new WebSocket(endpoint);
// Receive message from server word by word. Display the words as they are received.
var isResponse = false;

ws.onmessage = function (event) {
  var output = document.querySelector(".output");

  var status = document.querySelector(".status");
  console.log(event);
  var data = JSON.parse(event.data);
  if (data.sender === "bot") {
    if (data.type === "start") {
      isResponse = true;
      status.innerHTML = "Thinking...";
    } else if (data.type === "stream") {
      status.innerHTML = "Typing...";
      var outputText = document.querySelector(".output-text");
      if (data.message === "\n") {
        outputText.innerHTML += "<br>";
      } else {
        outputText.innerHTML += data.message;
      }
    } else if (data.type === "info") {
      status.innerHTML = data.message;
    } else if (data.type === "end") {
      status.innerHTML = "Says";
      // console.log(data.source_documents);
      var quotes = document.querySelector(".quotes");
      var quoteData = `<h2 class="font-poppins font-bold text-gray-600 text-xl">
      Slokes <span class="blue_gradient"></span>
    </h2>`;
      data.source_documents.forEach((element) => {
        quoteData += `<blockquote>
        <p>
          ${element.Devanagari}
        </p>
        <p class="text-gray-500 text-sm">
          ${element.Translation}
        </p>
        
        <a class="gitapart text-sky-500 underline" href="https://vedabase.io/en/library/bg/${element.Chapter}/${element.Verse}/" target="_blank"><footer>- Gita ${element.Chapter}.${element.Verse}</footer> </a>
      </blockquote>`;
      });
      quotes.innerHTML = quoteData;
    } else if (data.type === "error") {
      status.innerHTML = "Says";
      var outputText = document.querySelector(".output-text");
      outputText.innerHTML += data.message;
    }
  }
  output.scrollTop = output.scrollHeight;
};

function sendMessage(event) {
  event.preventDefault();
  var message_input = document.querySelector("#message_input");
  var suggestions = document.querySelector(".suggestions");

  var message = message_input.value;
  if (message === "") {
    return;
  }
  ws.send(message);
  message_input.value = "";
  var output = document.querySelector(".output");
  var suggestions = document.querySelector(".suggestions");
  suggestions.style.display = "none";
  output.innerHTML = `
    <div class="flex flex-col">
    <div class="flex flex-col gap-3">
    <h2 class="font-poppins font-bold text-gray-600 text-xl">
      You <span class="blue_gradient">Asked</span>
    </h2>
    <div class="summary_box">
      <p class="font-inter font-medium text-xl text-gray-700">
        <span class="font-bold">Q. </span>
        ${message}
      </p>
    </div>
  </div>
  <div class="flex flex-col gap-3 mt-4">
    <h2 class="font-poppins font-bold text-gray-600 text-xl">
      Krishna <span class="blue_gradient status"></span>
    </h2>
    <div class="summary_box">
      <p
        class="font-inter font-medium text-xl text-gray-700 output-text"
      >
        
      </p>
    </div>
  </div>
  <div class="quotes">
    
    
  </div>
          </div>
    `;
}
