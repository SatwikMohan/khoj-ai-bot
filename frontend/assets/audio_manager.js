function createTexminAudioManager(settings) {
  const audio = document.querySelector("audio");
  const key = "texmin_audio_owner:" + settings.sessionId;
  const instance = crypto.randomUUID ? crypto.randomUUID() : String(Date.now()) + Math.random();
  const owner = JSON.stringify({requestId: settings.requestId, instance});
  const ready = new Map(), clips = [];
  let stopped = false, busy = false, completed = false;
  let nextSequence = 0, finalSequence = null, currentSequence = null;

  function emit(name, detail = {}) {
    document.dispatchEvent(new CustomEvent("texmin:audio-" + name, {detail}));
  }
  function current() {
    return !stopped && !!settings.requestId && sessionStorage.getItem(key) === owner;
  }
  function stop(reason) {
    if (stopped) return;
    stopped = true;
    ready.clear(); clips.length = 0;
    audio.pause(); audio.removeAttribute("src"); audio.load();
    emit("stopped", {reason});
    console.info(JSON.stringify({event:"playback_stopped",request_id:settings.requestId,
      sequence:currentSequence,reason}));
  }
  function completeIfReady() {
    if (!completed && current() && finalSequence !== null &&
        nextSequence >= finalSequence && !clips.length && !busy && !audio.hasAttribute("src")) {
      completed = true;
      emit("completed");
      console.info(JSON.stringify({event:"playback_completed",request_id:settings.requestId}));
    }
  }
  function playNext() {
    if (!current() || busy) return;
    if (!clips.length) {completeIfReady(); return;}
    const item = clips.shift();
    busy = true; currentSequence = item.sequence;
    audio.src = "data:" + (item.mime || "audio/wav") + ";base64," + item.audio;
    audio.load();
    console.info(JSON.stringify({event:"playback_queued",request_id:settings.requestId,
      sequence:item.sequence,language:item.language,voice_id:item.voiceId}));
    audio.play().catch(error => {
      if (!current()) return;
      console.warn("Audio playback failed", error);
      emit("play-failed", {sequence:item.sequence});
    });
  }
  if (settings.requestId) sessionStorage.setItem(key, owner);
  function check() {if (!stopped && settings.requestId && !current()) stop("replaced");}
  window.addEventListener("storage", event => {if (event.key === key) check();});
  window.setInterval(check, 100);
  window.addEventListener("message", event => {
    const item = event.data || {};
    if (!current() || item.type !== "texmin:avatar-queue" ||
        item.queueId !== settings.queueId || item.requestId !== settings.requestId) return;
    if ((item.audio || item.skip) && Number.isInteger(item.sequence) &&
        item.sequence >= nextSequence && !ready.has(item.sequence)) {
      ready.set(item.sequence, item);
      while (ready.has(nextSequence)) {
        const ordered = ready.get(nextSequence);
        if (ordered.audio) {
          clips.push(ordered);
          emit("queued", {sequence:nextSequence});
        }
        ready.delete(nextSequence++);
      }
    }
    if (item.final) finalSequence = item.finalSequence;
    playNext();
  });
  audio.addEventListener("play", () => {
    check();
    if (current()) {
      busy = true;
      console.info(JSON.stringify({event:"playback_started",request_id:settings.requestId,
        sequence:currentSequence}));
    }
  });
  audio.addEventListener("ended", () => {
    if (!current()) return;
    console.info(JSON.stringify({event:"playback_ended",request_id:settings.requestId,
      sequence:currentSequence}));
    busy = false; audio.removeAttribute("src");
    playNext();
  });
  return {current,stop};
}
