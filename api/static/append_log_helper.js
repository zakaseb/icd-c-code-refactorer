/* Load before app.js. Bounded + rAF-batched pipeline logs (UMA / heavy GPU hosts).
 *
 * Long-running runs (hours, sometimes left overnight) used to crash the
 * browser tab.  Root cause: when the tab is backgrounded, browsers pause
 * or heavily throttle `requestAnimationFrame`, so the pending-buffer
 * (`p.buf`) used to coalesce SSE chunks would grow without bound.  When
 * the tab returned to foreground, the deferred flush had to allocate a
 * multi-GB string (`cur + p.buf`) and the tab OOM-crashed.
 *
 * This helper now enforces three independent caps so growth is bounded
 * regardless of tab visibility:
 *
 *   1. `MAX_STEP_LOG_CHARS`     — visible <pre>.textContent cap.
 *   2. `MAX_PENDING_BUF_CHARS`  — hard cap on the pending buffer; new
 *                                 chunks tail-displace older ones.
 *   3. `MAX_CHUNK_CHARS`        — single-chunk cap; oversized inputs
 *                                 are truncated before they enter buf.
 *
 * In addition we add a `setTimeout` fallback flush (which still fires
 * in hidden tabs, just throttled) and a `visibilitychange` flush so
 * pending data is drained whenever the tab gains/loses visibility.
 */
(function (global) {
  'use strict';

  var MAX_STEP_LOG_CHARS = 262144;            /* 256KB visible cap */
  var MAX_PENDING_BUF_CHARS = 524288;         /* 512KB pending-buf cap */
  var MAX_CHUNK_CHARS = MAX_STEP_LOG_CHARS;   /* single-chunk cap */
  var BG_FLUSH_INTERVAL_MS = 1000;            /* setTimeout fallback */
  /* Amortize tail-truncation cost: slice down to the cap only after
   * the buffer has grown past 2× the cap.  This keeps the amortized
   * per-append work O(chunk) instead of O(cap), which matters when
   * the agentic loop emits 1M+ tokens overnight. */
  var PENDING_BUF_HIGH_WATER_FACTOR = 2;

  if (typeof global.STEP_LOG_MAX_CHARS === 'number' && global.STEP_LOG_MAX_CHARS > 10000) {
    MAX_STEP_LOG_CHARS = global.STEP_LOG_MAX_CHARS;
    MAX_CHUNK_CHARS = MAX_STEP_LOG_CHARS;
  }
  if (typeof global.STEP_LOG_PENDING_MAX_CHARS === 'number' && global.STEP_LOG_PENDING_MAX_CHARS >= 16384) {
    MAX_PENDING_BUF_CHARS = global.STEP_LOG_PENDING_MAX_CHARS;
  }
  if (typeof global.STEP_LOG_BG_FLUSH_MS === 'number' && global.STEP_LOG_BG_FLUSH_MS >= 100) {
    BG_FLUSH_INTERVAL_MS = global.STEP_LOG_BG_FLUSH_MS;
  }

  /** @typedef {{ buf: string, raf: number|null, timer: number|null, el: Element }} Pending */
  /** @type {WeakMap<Element, Pending>} */
  var pending = new WeakMap();
  /** Strong refs to elements with un-flushed buffers, so visibilitychange
   *  handlers can drain them without traversing the DOM. */
  var pendingSet = new Set();

  function injectPerfCssOnce() {
    if (typeof document === 'undefined' || !document.getElementById) return;
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
      return;
    }
    var over = combined.length - MAX_STEP_LOG_CHARS;
    var head = '\n\n\u2026 (' + over + ' earlier characters omitted) \u2026\n\n';
    var budget = Math.max(0, MAX_STEP_LOG_CHARS - head.length);
    outEl.textContent = head + combined.slice(combined.length - budget);
  }

  function getPending(outEl) {
    var p = pending.get(outEl);
    if (!p) {
      p = { buf: '', raf: null, timer: null, el: outEl };
      pending.set(outEl, p);
    }
    return p;
  }

  function flushNow(outEl) {
    var p = pending.get(outEl);
    if (!p) return;
    if (p.raf != null) {
      try { global.cancelAnimationFrame(p.raf); } catch (e) { /* ignore */ }
      p.raf = null;
    }
    if (p.timer != null) {
      try { global.clearTimeout(p.timer); } catch (e) { /* ignore */ }
      p.timer = null;
    }
    if (!p.buf) {
      pendingSet['delete'](p);
      return;
    }
    /* Bound *both* operands of the concat below so we never allocate
     * something pathologically large when many chunks arrived while the
     * tab was hidden.  Each side is already bounded by its own cap, but
     * we re-check defensively in case the caps were edited at runtime. */
    var cur = outEl.textContent || '';
    if (cur.length > MAX_STEP_LOG_CHARS) {
      cur = cur.slice(cur.length - MAX_STEP_LOG_CHARS);
    }
    if (p.buf.length > MAX_PENDING_BUF_CHARS) {
      p.buf = p.buf.slice(p.buf.length - MAX_PENDING_BUF_CHARS);
    }
    var combined = cur + p.buf;
    p.buf = '';
    applyTruncated(outEl, combined);
    if (typeof outEl.scrollTop === 'number') {
      outEl.scrollTop = outEl.scrollHeight;
    }
    pendingSet['delete'](p);
  }

  function scheduleFlush(outEl) {
    injectPerfCssOnce();
    var p = getPending(outEl);
    pendingSet.add(p);
    if (p.raf == null && typeof global.requestAnimationFrame === 'function') {
      p.raf = global.requestAnimationFrame(function () {
        p.raf = null;
        flushNow(outEl);
      });
    }
    /* setTimeout fires even in backgrounded tabs (throttled to ~1Hz, but
     * runs); rAF can be fully paused.  Either callback is safe to call
     * twice — flushNow tolerates an empty buffer. */
    if (p.timer == null && typeof global.setTimeout === 'function') {
      p.timer = global.setTimeout(function () {
        p.timer = null;
        flushNow(outEl);
      }, BG_FLUSH_INTERVAL_MS);
    }
  }

  function appendStepLog(outEl, chunk) {
    if (!outEl) return;
    if (chunk === undefined || chunk === null) return;
    var c = (typeof chunk === 'string') ? chunk : String(chunk);
    if (!c) return;
    /* Bound any single oversized chunk (LLM may emit a huge token batch
     * after a long stall).  Keep the *tail* — most relevant content. */
    if (c.length > MAX_CHUNK_CHARS) {
      c = '\n\u2026 (' + (c.length - MAX_CHUNK_CHARS) +
          ' chars truncated from chunk) \u2026\n' +
          c.slice(c.length - MAX_CHUNK_CHARS);
    }
    var p = getPending(outEl);
    p.buf += c;
    /* Tail-displace the pending buffer so it stays bounded regardless
     * of how long the flush is deferred.  Amortized: only slice once
     * we've grown past 2× the cap so the cost is O(chunk), not O(cap),
     * per append. */
    var highWater = MAX_PENDING_BUF_CHARS * PENDING_BUF_HIGH_WATER_FACTOR;
    if (p.buf.length > highWater) {
      p.buf = p.buf.slice(p.buf.length - MAX_PENDING_BUF_CHARS);
    }
    scheduleFlush(outEl);
  }

  /** Synchronous flush (e.g. before reading DOM elsewhere). */
  function flushStepLog(outEl) {
    if (!outEl) return;
    flushNow(outEl);
  }

  /** Drain every element currently holding a pending buffer.  Cheap; only
   *  iterates elements that received output since their last flush. */
  function flushAllStepLogs() {
    /* Snapshot to a local array because flushNow mutates pendingSet. */
    var snapshot = [];
    pendingSet.forEach(function (p) { snapshot.push(p.el); });
    for (var i = 0; i < snapshot.length; i++) {
      flushNow(snapshot[i]);
    }
  }

  if (typeof document !== 'undefined' && typeof document.addEventListener === 'function') {
    document.addEventListener('visibilitychange', flushAllStepLogs, false);
    /* `pagehide`/`freeze` are fired when the tab is being put to sleep
     *  (BFCache / freeze).  Drain immediately so we don't carry a giant
     *  buffer across the freeze boundary. */
    document.addEventListener('pagehide', flushAllStepLogs, false);
  }

  global.appendStepLog = appendStepLog;
  global.flushStepLog = flushStepLog;
  global.flushAllStepLogs = flushAllStepLogs;

  /* Expose effective caps for diagnostics + tests.  These read the
   * post-override values so tests can assert defensive defaults. */
  global.__icdAppendLogConfig = function () {
    return {
      MAX_STEP_LOG_CHARS: MAX_STEP_LOG_CHARS,
      MAX_PENDING_BUF_CHARS: MAX_PENDING_BUF_CHARS,
      MAX_CHUNK_CHARS: MAX_CHUNK_CHARS,
      BG_FLUSH_INTERVAL_MS: BG_FLUSH_INTERVAL_MS,
    };
  };
})(typeof window !== 'undefined' ? window : (typeof global !== 'undefined' ? global : this));
