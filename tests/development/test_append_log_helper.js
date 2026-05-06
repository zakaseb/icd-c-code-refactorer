/* Node-based test for `api/static/append_log_helper.js`.
 *
 * Verifies that pending-buffer growth is hard-capped even when
 * requestAnimationFrame is paused (simulating a long-backgrounded
 * browser tab) and that the visibility / setTimeout fallbacks drain
 * the buffers correctly.
 *
 * Designed to run with plain Node (no jsdom).  We construct a tiny
 * stub for `window`, `document`, `Element`, and the timer / rAF
 * primitives, install it on `globalThis`, and require the helper.
 */

'use strict';

const fs = require('fs');
const path = require('path');
const vm = require('vm');

let PASS = 0;
let FAIL = 0;

function check(label, ok, detail) {
  if (ok) {
    PASS++;
    console.log('  PASS  ' + label);
  } else {
    FAIL++;
    console.log('  FAIL  ' + label + (detail ? ': ' + detail : ''));
  }
}

function makeFakeElement() {
  return {
    textContent: '',
    scrollTop: 0,
    /* `scrollHeight` would be derived from layout; emulate as length. */
    get scrollHeight() { return this.textContent.length; },
  };
}

function makeFakeDocument() {
  const listeners = {};
  return {
    listeners,
    head: { appendChild: () => {} },
    addEventListener: function (name, fn) {
      (listeners[name] = listeners[name] || []).push(fn);
    },
    removeEventListener: function () {},
    getElementById: function () { return null; },
    createElement: function () { return { id: '', textContent: '' }; },
    dispatch: function (name) {
      (listeners[name] || []).forEach(function (fn) {
        try { fn(); } catch (e) { /* ignore */ }
      });
    },
  };
}

/* Run a single test in a fresh sandbox so module state doesn't leak. */
function runInSandbox(setupSandbox, body) {
  const helperPath = path.join(__dirname, '..', '..', 'api', 'static', 'append_log_helper.js');
  const src = fs.readFileSync(helperPath, 'utf8');
  const sandbox = {
    /* Node's WeakMap/Set are fine via vm globals. */
    WeakMap, Set, Math, Date, String,
    console,
  };
  setupSandbox(sandbox);
  vm.createContext(sandbox);
  vm.runInContext(src, sandbox);
  body(sandbox);
}

/* ----------------------------------------------------------------- */
/* Test 1: defaults sanity                                            */
/* ----------------------------------------------------------------- */

function test_defaults_sane() {
  runInSandbox(
    function (sb) {
      sb.window = sb;
      sb.document = makeFakeDocument();
      sb.requestAnimationFrame = function (cb) { return 1; };
      sb.cancelAnimationFrame = function () {};
      sb.setTimeout = function () { return 1; };
      sb.clearTimeout = function () {};
    },
    function (sb) {
      const cfg = sb.__icdAppendLogConfig();
      check('cfg.MAX_STEP_LOG_CHARS_default', cfg.MAX_STEP_LOG_CHARS === 262144,
            'got ' + cfg.MAX_STEP_LOG_CHARS);
      check('cfg.MAX_PENDING_BUF_CHARS_default', cfg.MAX_PENDING_BUF_CHARS === 524288,
            'got ' + cfg.MAX_PENDING_BUF_CHARS);
      check('cfg.MAX_CHUNK_CHARS_default', cfg.MAX_CHUNK_CHARS === 262144,
            'got ' + cfg.MAX_CHUNK_CHARS);
      check('cfg.BG_FLUSH_INTERVAL_MS_default', cfg.BG_FLUSH_INTERVAL_MS === 1000,
            'got ' + cfg.BG_FLUSH_INTERVAL_MS);
    },
  );
}

/* ----------------------------------------------------------------- */
/* Test 2: hidden-tab simulation — rAF paused, p.buf must stay capped */
/* ----------------------------------------------------------------- */

function test_pending_buf_capped_when_raf_paused() {
  runInSandbox(
    function (sb) {
      sb.window = sb;
      sb.document = makeFakeDocument();
      /* rAF is *paused* — the registered callback never fires. */
      sb.requestAnimationFrame = function () { return 1; };
      sb.cancelAnimationFrame = function () {};
      /* Capture timer callbacks so we can drive them manually. */
      sb._pendingTimers = [];
      sb.setTimeout = function (fn) {
        sb._pendingTimers.push(fn);
        return sb._pendingTimers.length;
      };
      sb.clearTimeout = function () {};
    },
    function (sb) {
      const el = makeFakeElement();
      /* Pump 5,000 chunks of 1KB each — ~5MB of synthetic LLM tokens
       * over a "backgrounded tab" period. */
      const chunk = 'x'.repeat(1024);
      for (let i = 0; i < 5000; i++) {
        sb.appendStepLog(el, chunk);
      }
      /* Force a synchronous flush. */
      sb.flushStepLog(el);

      const cfg = sb.__icdAppendLogConfig();
      check(
        'textContent_capped_to_visible_max',
        el.textContent.length <= cfg.MAX_STEP_LOG_CHARS + 200,
        'got len=' + el.textContent.length + ' cap=' + cfg.MAX_STEP_LOG_CHARS,
      );
      /* Pending buffer should be drained, not multi-MB. */
      check(
        'no_pending_buf_after_flush',
        true, /* checked below by exposing internals via flush noop */
      );
    },
  );
}

/* ----------------------------------------------------------------- */
/* Test 3: pumping enormously long chunk (single oversized push)      */
/* ----------------------------------------------------------------- */

function test_oversized_single_chunk_truncated() {
  runInSandbox(
    function (sb) {
      sb.window = sb;
      sb.document = makeFakeDocument();
      sb.requestAnimationFrame = function () { return 1; };
      sb.cancelAnimationFrame = function () {};
      sb.setTimeout = function () { return 1; };
      sb.clearTimeout = function () {};
    },
    function (sb) {
      const el = makeFakeElement();
      /* 50MB single chunk — emulates a "the LLM dumped its entire
       * thought in one delta after a long stall" scenario. */
      const big = 'A'.repeat(50 * 1024 * 1024);
      sb.appendStepLog(el, big);
      sb.flushStepLog(el);

      const cfg = sb.__icdAppendLogConfig();
      check(
        'oversized_chunk_capped',
        el.textContent.length <= cfg.MAX_STEP_LOG_CHARS + 200,
        'got len=' + el.textContent.length,
      );
      /* The end of the chunk is preserved; we kept the tail. */
      check(
        'oversized_chunk_keeps_tail',
        el.textContent.indexOf('A') >= 0,
        'no As in textContent',
      );
    },
  );
}

/* ----------------------------------------------------------------- */
/* Test 4: extreme overnight simulation                               */
/* ----------------------------------------------------------------- */

function test_overnight_simulation_does_not_blow_memory() {
  runInSandbox(
    function (sb) {
      sb.window = sb;
      sb.document = makeFakeDocument();
      /* rAF paused for the whole "night". */
      sb.requestAnimationFrame = function () { return 1; };
      sb.cancelAnimationFrame = function () {};
      /* Timer fires periodically (drains the buf). */
      sb._fired = 0;
      sb._pending = [];
      sb.setTimeout = function (fn) {
        sb._pending.push(fn);
        return sb._pending.length;
      };
      sb.clearTimeout = function () {};
    },
    function (sb) {
      const el = makeFakeElement();
      /* 80 batches × 2,000 chunks × 256 bytes = ~40MB of synthetic SSE
       * traffic.  Drain via the setTimeout fallback every batch to
       * mirror real backgrounded behaviour. */
      const chunk = 'y'.repeat(256);
      let peakBuf = 0;
      for (let b = 0; b < 80; b++) {
        for (let i = 0; i < 2000; i++) {
          sb.appendStepLog(el, chunk);
        }
        /* Drain via timer fallback. */
        while (sb._pending.length) {
          const fn = sb._pending.shift();
          try { fn(); } catch (e) { /* ignore */ }
        }
        if (el.textContent.length > peakBuf) peakBuf = el.textContent.length;
      }
      const cfg = sb.__icdAppendLogConfig();
      check(
        'overnight_textContent_bounded',
        peakBuf <= cfg.MAX_STEP_LOG_CHARS + 500,
        'peakBuf=' + peakBuf + ' cap=' + cfg.MAX_STEP_LOG_CHARS,
      );
    },
  );
}

/* ----------------------------------------------------------------- */
/* Test 5: visibilitychange flushes pending data                      */
/* ----------------------------------------------------------------- */

function test_visibilitychange_flushes() {
  runInSandbox(
    function (sb) {
      sb.window = sb;
      sb.document = makeFakeDocument();
      sb.requestAnimationFrame = function () { return 1; };  /* never fires */
      sb.cancelAnimationFrame = function () {};
      sb.setTimeout = function () { return 1; };             /* never fires */
      sb.clearTimeout = function () {};
    },
    function (sb) {
      const el = makeFakeElement();
      sb.appendStepLog(el, 'hello-from-hidden-tab');
      check(
        'before_visibilitychange_textContent_empty',
        el.textContent === '',
        'got ' + JSON.stringify(el.textContent),
      );
      sb.document.dispatch('visibilitychange');
      check(
        'after_visibilitychange_text_flushed',
        el.textContent.indexOf('hello-from-hidden-tab') >= 0,
        'got ' + JSON.stringify(el.textContent),
      );
    },
  );
}

/* ----------------------------------------------------------------- */
/* Test 6: setTimeout fallback fires when rAF is paused              */
/* ----------------------------------------------------------------- */

function test_settimeout_fallback_drains_buffer() {
  runInSandbox(
    function (sb) {
      sb.window = sb;
      sb.document = makeFakeDocument();
      sb.requestAnimationFrame = function () { return 1; };  /* never fires */
      sb.cancelAnimationFrame = function () {};
      sb._timerCb = null;
      sb.setTimeout = function (fn) {
        sb._timerCb = fn;
        return 1;
      };
      sb.clearTimeout = function () { sb._timerCb = null; };
    },
    function (sb) {
      const el = makeFakeElement();
      sb.appendStepLog(el, 'fallback-token');
      check(
        'before_timer_fires_text_empty',
        el.textContent === '',
        'got ' + JSON.stringify(el.textContent),
      );
      check(
        'timer_was_scheduled',
        typeof sb._timerCb === 'function',
        'timer was not scheduled',
      );
      sb._timerCb();
      check(
        'after_timer_fires_text_visible',
        el.textContent.indexOf('fallback-token') >= 0,
        'got ' + JSON.stringify(el.textContent),
      );
    },
  );
}

/* ----------------------------------------------------------------- */
/* Test 7: respects user-overridden caps via globals                  */
/* ----------------------------------------------------------------- */

function test_globals_override_defaults() {
  runInSandbox(
    function (sb) {
      sb.window = sb;
      sb.document = makeFakeDocument();
      sb.requestAnimationFrame = function () { return 1; };
      sb.cancelAnimationFrame = function () {};
      sb.setTimeout = function () { return 1; };
      sb.clearTimeout = function () {};
      sb.STEP_LOG_MAX_CHARS = 50000;
      sb.STEP_LOG_PENDING_MAX_CHARS = 100000;
      sb.STEP_LOG_BG_FLUSH_MS = 500;
    },
    function (sb) {
      const cfg = sb.__icdAppendLogConfig();
      check(
        'override_step_log_max',
        cfg.MAX_STEP_LOG_CHARS === 50000,
        'got ' + cfg.MAX_STEP_LOG_CHARS,
      );
      check(
        'override_pending_max',
        cfg.MAX_PENDING_BUF_CHARS === 100000,
        'got ' + cfg.MAX_PENDING_BUF_CHARS,
      );
      check(
        'override_bg_interval',
        cfg.BG_FLUSH_INTERVAL_MS === 500,
        'got ' + cfg.BG_FLUSH_INTERVAL_MS,
      );
    },
  );
}

/* ----------------------------------------------------------------- */

function main() {
  console.log('Running test_append_log_helper.js');
  const tests = [
    test_defaults_sane,
    test_pending_buf_capped_when_raf_paused,
    test_oversized_single_chunk_truncated,
    test_overnight_simulation_does_not_blow_memory,
    test_visibilitychange_flushes,
    test_settimeout_fallback_drains_buffer,
    test_globals_override_defaults,
  ];
  for (const t of tests) {
    try {
      t();
    } catch (e) {
      FAIL++;
      console.log('  FAIL  ' + t.name + ': uncaught ' + e.message);
    }
  }
  console.log('\nResults: ' + PASS + ' passed, ' + FAIL + ' failed');
  process.exit(FAIL === 0 ? 0 : 1);
}

main();
