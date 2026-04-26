/* Load before app.js. Bounded + rAF-batched pipeline logs (UMA / heavy GPU hosts). */
(function (global) {
  'use strict';

  var MAX_STEP_LOG_CHARS = 262144;
  if (typeof global.STEP_LOG_MAX_CHARS === 'number' && global.STEP_LOG_MAX_CHARS > 10000) {
    MAX_STEP_LOG_CHARS = global.STEP_LOG_MAX_CHARS;
  }

  /** @typedef {{ buf: string, raf: number|null }} Pending */
  /** @type {WeakMap<Element, Pending>} */
  var pending = new WeakMap();

  function injectPerfCssOnce() {
    if (document.getElementById('icd-step-log-perf')) return;
    var st = document.createElement('style');
    st.id = 'icd-step-log-perf';
    st.textContent =
      '.pipeline-step pre.step-output.visible{' +
      'content-visibility:auto;' +
      'contain:layout style;' +
      '}';
    document.head.appendChild(st);
  }

  function applyTruncated(outEl, combined) {
    if (combined.length <= MAX_STEP_LOG_CHARS) {
      outEl.textContent = combined;
    } else {
      var over = combined.length - MAX_STEP_LOG_CHARS;
      var head = '\n\n\u2026 (' + over + ' earlier characters omitted) \u2026\n\n';
      var budget = Math.max(0, MAX_STEP_LOG_CHARS - head.length);
      outEl.textContent = head + combined.slice(-budget);
    }
  }

  function getPending(outEl) {
    var p = pending.get(outEl);
    if (!p) {
      p = { buf: '', raf: null };
      pending.set(outEl, p);
    }
    return p;
  }

  function flushNow(outEl) {
    var p = pending.get(outEl);
    if (!p || !p.buf) {
      if (p) p.raf = null;
      return;
    }
    var cur = outEl.textContent;
    var combined = cur + p.buf;
    p.buf = '';
    p.raf = null;
    applyTruncated(outEl, combined);
    outEl.scrollTop = outEl.scrollHeight;
  }

  function scheduleFlush(outEl) {
    injectPerfCssOnce();
    var p = getPending(outEl);
    if (p.raf != null) return;
    p.raf = global.requestAnimationFrame(function () {
      flushNow(outEl);
    });
  }

  function appendStepLog(outEl, chunk) {
    if (!outEl) return;
    if (chunk === undefined || chunk === null) return;
    var c = String(chunk);
    if (!c) return;
    var p = getPending(outEl);
    p.buf += c;
    scheduleFlush(outEl);
  }

  /** Synchronous flush (e.g. before reading DOM elsewhere). */
  function flushStepLog(outEl) {
    if (!outEl) return;
    var p = pending.get(outEl);
    if (p && p.raf != null) {
      global.cancelAnimationFrame(p.raf);
      p.raf = null;
    }
    flushNow(outEl);
  }

  global.appendStepLog = appendStepLog;
  global.flushStepLog = flushStepLog;
})(typeof window !== 'undefined' ? window : this);
