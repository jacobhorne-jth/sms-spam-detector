document.getElementById("checkButton").addEventListener("click", async () => {
  const text = document.getElementById("messageInput").value;

  if (!text.trim()) {
    alert("Please enter a message.");
    return;
  }

  const response = await fetch("http://127.0.0.1:8000/predict", {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify({ text })
  });

  const data = await response.json();

  const resultDiv = document.getElementById("result");
  if (data.prediction === "ham") {
    resultDiv.innerHTML = `✅ This is <span class='ham'>HAM</span> (Good Message).<br>Spam probability: ${(data.spam_probability * 100).toFixed(1)}%`;
  } else if (data.prediction === "spam") {
    resultDiv.innerHTML = `🚨 This is <span class='spam'>SPAM</span> (Potentially Dangerous).<br>Spam probability: ${(data.spam_probability * 100).toFixed(1)}%`;
  } else {
    resultDiv.innerHTML = "❓ Unable to determine.";
  }
});
