function createTexminAudioManager(settings) {
  const audio = document.querySelector("audio");
  const key = "texmin_audio_owner:" + settings.sessionId;
  const stateKey = "texmin_voice_audio_state:" + settings.sessionId;
  function mark(state, reason = "") {
    if (!settings.requestId) return;
    // A replaced player must not overwrite the newer turn's completion state.
    if (state === "stopped") {
      try {
        const latest = JSON.parse(localStorage.getItem(stateKey) || "null");
        if (latest && latest.requestId !== settings.requestId) return;
      } catch (_) {}
    }
    localStorage.setItem(stateKey, JSON.stringify({
      requestId:settings.requestId, state, reason, at:Date.now()
    }));
  }
  const instance = crypto.randomUUID ? crypto.randomUUID() : String(Date.now()) + Math.random();
  const owner = JSON.stringify({requestId: settings.requestId, instance});
  const ready = new Map(), clips = [];
  let stopped = false, busy = false, completed = false;
  let nextSequence = 0, finalSequence = null, currentSequence = null;
  let lastEndedAt = null, played = 0, dropped = 0;
  let lastQueueProgressAt = performance.now();
  const playbackGaps = [];
  function percentile(values, fraction) {
    if (!values.length) return null;
    const ordered = [...values].sort((a, b) => a - b);
    return Math.round(ordered[Math.ceil(fraction * ordered.length) - 1] * 10) / 10;
  }

  function emit(name, detail = {}) {
    document.dispatchEvent(new CustomEvent("texmin:audio-" + name, {detail}));
  }
  function current() {
    return !stopped && !!settings.requestId && sessionStorage.getItem(key) === owner;
  }
  function stop(reason) {
    if (stopped) return;
    stopped = true;
    mark("stopped", reason);
    dropped += ready.size + clips.length;
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
      mark("completed");
      emit("completed");
      console.info(JSON.stringify({event:"playback_completed",request_id:settings.requestId,
        played, dropped, gap_median_ms:percentile(playbackGaps,0.5),
        gap_p95_ms:percentile(playbackGaps,0.95)}));
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
      stop("playback_failed");
    });
  }
  if (settings.requestId) {
    sessionStorage.setItem(key, owner);
    mark("pending");
  }
  function check() {
    if (stopped || !settings.requestId) return;
    if (!current()) {stop("replaced"); return;}
    if (finalSequence !== null && nextSequence < finalSequence && !busy &&
        performance.now() - lastQueueProgressAt > settings.queueTimeoutMs) {
      stop("queue_stalled");
    }
  }
  window.addEventListener("storage", event => {if (event.key === key) check();});
  window.setInterval(check, 100);
  window.addEventListener("message", event => {
    const item = event.data || {};
    if (!current() || item.type !== "texmin:avatar-queue" ||
        item.queueId !== settings.queueId || item.requestId !== settings.requestId) return;
    if ((item.audio || item.skip) && Number.isInteger(item.sequence) &&
        item.sequence >= nextSequence && !ready.has(item.sequence)) {
      if (ready.size + clips.length >= settings.maxQueuedSegments) {
        stop("queue_limit");
        return;
      }
      ready.set(item.sequence, item);
      lastQueueProgressAt = performance.now();
      while (ready.has(nextSequence)) {
        const ordered = ready.get(nextSequence);
        if (ordered.audio) {
          clips.push(ordered);
          emit("queued", {sequence:nextSequence});
        }
        ready.delete(nextSequence++);
      }
    }
    if (item.final) {
      finalSequence = item.finalSequence;
      lastQueueProgressAt = performance.now();
    }
    playNext();
  });
  audio.addEventListener("play", () => {
    check();
    if (current()) {
      mark("playing");
      busy = true;
      played += 1;
      const gap = lastEndedAt === null ? null : performance.now() - lastEndedAt;
      if (gap !== null) playbackGaps.push(gap);
      console.info(JSON.stringify({event:"playback_started",request_id:settings.requestId,
        sequence:currentSequence, gap_after_previous_ms:gap === null ? null : Math.round(gap * 10) / 10,
        queue_depth:clips.length}));
    }
  });
  audio.addEventListener("ended", () => {
    if (!current()) return;
    console.info(JSON.stringify({event:"playback_ended",request_id:settings.requestId,
      sequence:currentSequence}));
    lastEndedAt = performance.now();
    lastQueueProgressAt = lastEndedAt;
    busy = false; audio.removeAttribute("src");
    playNext();
  });
  return {current,stop};
}
