let recorder, chunks = [], lastAnswer = "";
const $ = (id) => document.getElementById(id);
function show(data) {
  $("error").textContent = "";
  $("result").classList.remove("hidden");
  $("transcript").textContent = data.transcript || "Text query";
  $("answer").textContent = data.answer || "No answer returned.";
  lastAnswer = data.answer || "";
  $("sources").innerHTML = (data.sources || []).map(s => `<div class="source"><strong>${s.chunk_id}</strong> · ${(s.text || "").replaceAll("<", "&lt;")}</div>`).join("") || "<div class='source'>No evidence returned.</div>";
  const t = data.timings || {};
  $("timings").textContent = `Total ${t.total_ms ?? "—"} ms · retrieval ${t.retrieval_ms ?? "—"} ms · generation ${t.generation_ms ?? "—"} ms · target ${t.target_met ? "met" : "not met"}`;
}
async function askText(query) {
  const res = await fetch("/query", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ query }) });
  const data = await res.json(); if (!res.ok) throw new Error(data.detail || "Request failed"); show(data);
}
$("text-form").addEventListener("submit", async (e) => { e.preventDefault(); try { await askText($("query").value); } catch (err) { $("error").textContent = err.message; } });
$("record").addEventListener("click", async () => {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    chunks = []; recorder = new MediaRecorder(stream); recorder.ondataavailable = e => chunks.push(e.data);
    recorder.onstop = async () => {
      stream.getTracks().forEach(t => t.stop()); const blob = new Blob(chunks, { type: "audio/webm" });
      const form = new FormData(); form.append("file", blob, "recording.webm");
      try { const res = await fetch("/voice-query", { method: "POST", body: form }); show(await res.json()); } catch (err) { $("error").textContent = err.message; }
    };
    recorder.start(); $("record").disabled = true; $("stop").disabled = false; $("recording-state").textContent = "Recording…";
  } catch (err) { $("error").textContent = "Microphone permission is needed for voice input."; }
});
$("stop").addEventListener("click", () => { if (recorder && recorder.state !== "inactive") recorder.stop(); $("record").disabled = false; $("stop").disabled = true; $("recording-state").textContent = "Transcribing…"; });
$("speak").addEventListener("click", () => { if (lastAnswer && "speechSynthesis" in window) speechSynthesis.speak(new SpeechSynthesisUtterance(lastAnswer)); });
