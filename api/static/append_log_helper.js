/* Must load before app.js — provides appendStepLog (bounded step <pre> buffer). */
(function (global) {
  'use strict';
  var MAX_STEP_LOG_CHARS = 400000;
  function appendStepLog(outEl, chunk) {
    if (!outEl) return;
    if (chunk === undefined || chunk === null) return;
    var c = String(chunk);
    if (!c) return;
    var cur = outEl.textContent;
    var combined = cur + c;
    if (combined.length <= MAX_STEP_LOG_CHARS) {
      outEl.textContent = combined;
    } else {
      var over = combined.length - MAX_STEP_LOG_CHARS;
      var head = '\n\n\u2026 (' + over + ' earlier characters omitted) \u2026\n\n';
      var budget = Math.max(0, MAX_STEP_LOG_CHARS - head.length);
      outEl.textContent = head + combined.slice(-budget);
    }
    outEl.scrollTop = outEl.scrollHeight;
  }
  global.appendStepLog = appendStepLog;
})(typeof window !== 'undefined' ? window : this);
