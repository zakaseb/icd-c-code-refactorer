(function () {
  'use strict';

  let sessionId = null;
  let codeFiles = [];
  let sourceIcdFile = null;
  let targetIcdFile = null;
  let repoZipFile = null;

  /* ----- DOM refs ------------------------------------------------ */
  const dropCode       = document.getElementById('drop-code');
  const dropSourceIcd  = document.getElementById('drop-source-icd');
  const dropTargetIcd  = document.getElementById('drop-target-icd');
  const dropRepoZip    = document.getElementById('drop-repo-zip');
  const inputCode      = document.getElementById('input-code');
  const inputSourceIcd = document.getElementById('input-source-icd');
  const inputTargetIcd = document.getElementById('input-target-icd');
  const inputRepoZip   = document.getElementById('input-repo-zip');
  const codeFileList   = document.getElementById('code-file-list');
  const srcIcdList     = document.getElementById('source-icd-file-list');
  const tgtIcdList     = document.getElementById('target-icd-file-list');
  const repoZipList    = document.getElementById('repo-zip-file-list');
  const processBtn     = document.getElementById('process-btn');
  const resetBtn       = document.getElementById('reset-btn');
  const sandboxRetriesInput = document.getElementById('sandbox-retries-input');
  const sandboxRetriesIndefinite = document.getElementById('sandbox-retries-indefinite');
  const sandboxRetriesHint = document.getElementById('sandbox-retries-hint');
  let sandboxRetriesIndefiniteMode = false;
  const procSection    = document.getElementById('processing-section');
  const pipelineSteps  = document.getElementById('pipeline-steps');
  const resultsSection = document.getElementById('results-section');
  const resultTabs     = document.getElementById('result-tabs');
  const codePreview    = document.getElementById('code-preview');
  const downloadBtn    = document.getElementById('download-btn');
  const downloadRepoBtn = document.getElementById('download-repo-btn');
  const pauseBtn       = document.getElementById('pause-btn');
  const resumeBtn      = document.getElementById('resume-btn');
  const convSection    = document.getElementById('conversation-section');
  const convMessages   = document.getElementById('conversation-messages');
  const convInput      = document.getElementById('conversation-input');
  const sendMsgBtn     = document.getElementById('send-msg-btn');
  const regenerateBtn  = document.getElementById('regenerate-btn');
  let activeEventSource = null;
  let resumeCurrentProcess = null;

  /* ----- Helpers ------------------------------------------------- */

  async function initSession() {
    const r = await fetch('/api/session/create', { method: 'POST' });
    const d = await r.json();
    sessionId = d.session_id;
  }

  function fmtSize(b) {
    if (b < 1024)          return b + ' B';
    if (b < 1024 * 1024)   return (b / 1024).toFixed(1) + ' KB';
    return (b / (1024 * 1024)).toFixed(1) + ' MB';
  }

  function setupDropZone(zone, input, onFiles) {
    input.addEventListener('change', () => {
      if (input.files.length) {
        onFiles(Array.from(input.files));
        input.value = '';
      }
    });
    zone.addEventListener('dragenter', e => e.preventDefault());
    zone.addEventListener('dragover', e => { e.preventDefault(); zone.classList.add('drag-over'); });
    zone.addEventListener('dragleave', e => {
      if (!zone.contains(e.relatedTarget)) zone.classList.remove('drag-over');
    });
    zone.addEventListener('drop', e => {
      e.preventDefault();
      e.stopPropagation();
      zone.classList.remove('drag-over');
      if (e.dataTransfer.files.length) onFiles(Array.from(e.dataTransfer.files));
    });
  }

  /* ----- Render helpers ------------------------------------------ */

  function renderCodeFiles() {
    codeFileList.innerHTML = '';
    codeFiles.forEach((f, i) => {
      const d = document.createElement('div');
      d.className = 'file-item';
      d.innerHTML =
        '<span class="file-icon">&lt;/&gt;</span>' +
        '<span class="file-name">' + f.name + '</span>' +
        '<span class="file-size">' + fmtSize(f.size) + '</span>' +
        '<button class="file-remove" data-idx="' + i + '">&times;</button>';
      codeFileList.appendChild(d);
    });
    codeFileList.querySelectorAll('.file-remove').forEach(btn => {
      btn.addEventListener('click', e => {
        codeFiles.splice(parseInt(e.target.dataset.idx), 1);
        renderCodeFiles();
        updateBtn();
      });
    });
    const card = document.getElementById('card-code');
    card.classList.toggle('has-files', codeFiles.length > 0);
    dropCode.classList.toggle('uploaded', codeFiles.length > 0);
  }

  function renderIcdFile(file, listEl, dropEl, cardId) {
    listEl.innerHTML = '';
    if (file) {
      const d = document.createElement('div');
      d.className = 'file-item';
      d.innerHTML =
        '<span class="file-icon">PDF</span>' +
        '<span class="file-name">' + file.name + '</span>' +
        '<span class="file-size">' + fmtSize(file.size) + '</span>';
      listEl.appendChild(d);
      document.getElementById(cardId).classList.add('has-files');
      dropEl.classList.add('uploaded');
    } else {
      document.getElementById(cardId).classList.remove('has-files');
      dropEl.classList.remove('uploaded');
    }
  }

  function updateBtn() {
    processBtn.disabled = !(codeFiles.length > 0 && sourceIcdFile && targetIcdFile);
  }

  function setSandboxRetriesIndefinite(on) {
    sandboxRetriesIndefiniteMode = !!on;
    if (sandboxRetriesIndefinite) {
      sandboxRetriesIndefinite.classList.toggle('active', sandboxRetriesIndefiniteMode);
      sandboxRetriesIndefinite.setAttribute(
        'aria-pressed',
        sandboxRetriesIndefiniteMode ? 'true' : 'false'
      );
    }
    if (sandboxRetriesInput) {
      sandboxRetriesInput.disabled = sandboxRetriesIndefiniteMode;
    }
    if (sandboxRetriesHint) {
      sandboxRetriesHint.textContent = sandboxRetriesIndefiniteMode
        ? 'Indefinite: the sandbox build keeps iterating until the project builds successfully.'
        : 'Stop after this many sandbox build reiterations, or keep going until the build succeeds.';
    }
  }

  function getSandboxRetriesQueryValue() {
    if (sandboxRetriesIndefiniteMode) return 'indefinite';
    if (!sandboxRetriesInput) return '25';
    var n = parseInt(sandboxRetriesInput.value, 10);
    if (!Number.isFinite(n) || n < 1) {
      n = 25;
      sandboxRetriesInput.value = String(n);
    }
    return String(n);
  }

  function processStreamUrl(path) {
    var q = 'sandbox_retries=' + encodeURIComponent(getSandboxRetriesQueryValue());
    return path + (path.indexOf('?') >= 0 ? '&' : '?') + q;
  }

  if (sandboxRetriesIndefinite) {
    sandboxRetriesIndefinite.addEventListener('click', function () {
      setSandboxRetriesIndefinite(!sandboxRetriesIndefiniteMode);
    });
  }
  if (sandboxRetriesInput) {
    sandboxRetriesInput.addEventListener('input', function () {
      if (sandboxRetriesIndefiniteMode) setSandboxRetriesIndefinite(false);
    });
  }
  setSandboxRetriesIndefinite(false);

  function renderZipFile(file, listEl, dropEl, cardId) {
    listEl.innerHTML = '';
    if (file) {
      const d = document.createElement('div');
      d.className = 'file-item';
      d.innerHTML =
        '<span class="file-icon">ZIP</span>' +
        '<span class="file-name">' + file.name + '</span>' +
        '<span class="file-size">' + fmtSize(file.size) + '</span>';
      listEl.appendChild(d);
      document.getElementById(cardId).classList.add('has-files');
      dropEl.classList.add('uploaded');
    } else {
      document.getElementById(cardId).classList.remove('has-files');
      dropEl.classList.remove('uploaded');
    }
  }

  /* ----- Upload helpers ------------------------------------------ */

  async function uploadCodeFiles() {
    const fd = new FormData();
    codeFiles.forEach(f => fd.append('files', f));
    return (await fetch('/api/upload/code/' + sessionId, { method: 'POST', body: fd })).json();
  }

  async function uploadIcd(endpoint, file) {
    const fd = new FormData();
    fd.append('file', file);
    return (await fetch('/api/upload/' + endpoint + '/' + sessionId, { method: 'POST', body: fd })).json();
  }

  async function uploadRepoZip() {
    if (!repoZipFile) return null;
    const fd = new FormData();
    fd.append('file', repoZipFile);
    return (await fetch('/api/upload/repo-zip/' + sessionId, { method: 'POST', body: fd })).json();
  }

  /* ----- Pipeline step UI ---------------------------------------- */

  function createStep(id, label) {
    const el = document.createElement('div');
    el.className = 'pipeline-step';
    el.id = 'step-' + id;
    el.innerHTML =
      '<div class="step-header">' +
        '<span class="step-status pending">' +
          '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/></svg>' +
        '</span>' +
        '<span class="step-label">' + label + '</span>' +
        '<span class="step-toggle">&#9660;</span>' +
      '</div>' +
      '<pre class="step-output"></pre>';
    el.querySelector('.step-header').addEventListener('click', () => {
      el.querySelector('.step-output').classList.toggle('visible');
    });
    return el;
  }

  function setStepStatus(el, status) {
    const s = el.querySelector('.step-status');
    s.className = 'step-status ' + status;
    if (status === 'running') {
      s.innerHTML = '<svg class="spinner" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 12a9 9 0 11-6.219-8.56"/></svg>';
    } else if (status === 'complete') {
      s.innerHTML = '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>';
    } else if (status === 'error') {
      s.innerHTML = '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/></svg>';
    }
  }

  /* ----- Main process flow --------------------------------------- */

  async function runProcess() {
    processBtn.disabled = true;
    resetBtn.style.display = '';
    if (pauseBtn) pauseBtn.style.display = '';
    if (resumeBtn) resumeBtn.style.display = 'none';
    procSection.classList.remove('hidden');
    resultsSection.classList.add('hidden');
    pipelineSteps.innerHTML = '';

    /* 1. Upload all resources */
    const upStep = createStep('upload', 'Uploading files\u2026');
    pipelineSteps.appendChild(upStep);
    setStepStatus(upStep, 'running');

    try {
      await initSession();
      const uploads = [
        uploadCodeFiles(),
        uploadIcd('source-icd', sourceIcdFile),
        uploadIcd('target-icd', targetIcdFile),
      ];
      if (repoZipFile) uploads.push(uploadRepoZip());
      await Promise.all(uploads);
      setStepStatus(upStep, 'complete');
      upStep.querySelector('.step-label').textContent =
        'Files uploaded' + (repoZipFile ? ' (including repository)' : '');
    } catch (err) {
      setStepStatus(upStep, 'error');
      const out = upStep.querySelector('.step-output');
      out.textContent = err.message;
      out.classList.add('visible');
      processBtn.disabled = false;
      return;
    }

    /* 2. Stream processing via SSE */
    const analysisStep = createStep('analysis', 'Analyzing ICD differences\u2026');
    pipelineSteps.appendChild(analysisStep);
    setStepStatus(analysisStep, 'running');
    const analysisOutput = analysisStep.querySelector('.step-output');
    analysisOutput.classList.add('visible');
    analysisOutput.textContent = 'Connecting to LLM... Large documents may take 5-10 minutes for the first response.';

    const fileSteps = {};
    let generatedFiles = [];
    let verificationStep = null;
    let compileStep = null;
    let sandboxStep = null;
    let gitnexusStep = null;
    let hasSandboxBuild = false;

    let es = new EventSource(processStreamUrl('/api/process/' + sessionId));
    activeEventSource = es;

    es.onmessage = function (event) {
      const msg = JSON.parse(event.data);

      switch (msg.type) {
        case 'token': {
          let step;
          if (msg.stage === 'analysis') step = analysisStep;
          else if (msg.stage === 'gitnexus') step = gitnexusStep;
          else if (msg.stage === 'verification') step = verificationStep;
          else if (msg.stage === 'compile') step = compileStep;
          else if (msg.stage === 'sandbox_build') step = sandboxStep;
          else if (msg.stage === 'transform' && msg.file) step = fileSteps[msg.file];
          if (step) {
            const o = step.querySelector('.step-output');
            if (o.textContent.startsWith('Connecting to LLM')) o.textContent = '';
            o.textContent += msg.token;
            o.scrollTop = o.scrollHeight;
          }
          break;
        }
        case 'stage':
          if (msg.stage === 'transform' && msg.file && !fileSteps[msg.file]) {
            const s = createStep('file-' + msg.index,
              'Transforming ' + msg.file + ' (' + (msg.index + 1) + '/' + msg.total + ')');
            pipelineSteps.appendChild(s);
            setStepStatus(s, 'running');
            s.querySelector('.step-output').classList.add('visible');
            fileSteps[msg.file] = s;
          } else if (msg.stage === 'gitnexus' && !gitnexusStep) {
            gitnexusStep = createStep('gitnexus',
              'GitNexus: extracting codebase understanding\u2026');
            pipelineSteps.appendChild(gitnexusStep);
            setStepStatus(gitnexusStep, 'running');
            gitnexusStep.querySelector('.step-output').classList.add('visible');
          } else if (msg.stage === 'verification' && !verificationStep) {
            verificationStep = createStep('verification',
              'Verifying generated code against ICDs & repository\u2026');
            pipelineSteps.appendChild(verificationStep);
            setStepStatus(verificationStep, 'running');
            verificationStep.querySelector('.step-output').classList.add('visible');
          } else if (msg.stage === 'compile' && !compileStep) {
            compileStep = createStep('compile',
              'Compiling generated .c files to .o objects\u2026');
            pipelineSteps.appendChild(compileStep);
            setStepStatus(compileStep, 'running');
            compileStep.querySelector('.step-output').classList.add('visible');
          } else if (msg.stage === 'sandbox_build' && !sandboxStep) {
            sandboxStep = createStep('sandbox-build',
              'Building generated code in sandbox environment\u2026');
            pipelineSteps.appendChild(sandboxStep);
            setStepStatus(sandboxStep, 'running');
            sandboxStep.querySelector('.step-output').classList.add('visible');
          }
          break;

        case 'compile_file_result':
          if (compileStep) {
            const out = compileStep.querySelector('.step-output');
            const tag = msg.success ? 'OK' : 'FAIL';
            const line =
              '\n[' + tag + '] ' + msg.file +
              (msg.object ? ' \u2192 ' + msg.object : '') +
              ' (attempt ' + (msg.attempt || 1) + ')\n';
            if (typeof window.appendStepLog === 'function') {
              window.appendStepLog(out, line);
            } else {
              out.textContent += line;
              out.scrollTop = out.scrollHeight;
            }
          }
          break;

        case 'compile_summary':
          if (compileStep) {
            compileStep.querySelector('.step-label').textContent =
              'Per-file compile: ' + msg.ok + '/' + msg.total + ' compiled' +
              (msg.failed ? ', ' + msg.failed + ' still failing' : '');
          }
          break;

        case 'compile_artifacts_ready':
          // Surface the Download All button NOW so the user can grab the
          // .c / .h / .o triples + compile_report.txt while sandbox_build
          // is still running. Re-call showResults whenever new artefacts
          // become available so the tab list stays current. The server
          // already pre-filters .o out of msg.files, but we re-check
          // defensively before wiring them up as preview tabs.
          generatedFiles = (Array.isArray(msg.files) ? msg.files : []).filter(function (f) {
            return !f.toLowerCase().endsWith('.o');
          });
          if (generatedFiles.length) {
            showResults(generatedFiles, hasSandboxBuild, true);
          }
          break;

        case 'stage_complete':
          if (msg.stage === 'analysis') {
            setStepStatus(analysisStep, 'complete');
            analysisStep.querySelector('.step-label').textContent = 'ICD analysis complete';
          } else if (msg.stage === 'gitnexus' && gitnexusStep) {
            setStepStatus(gitnexusStep, 'complete');
            gitnexusStep.querySelector('.step-label').textContent =
              'GitNexus codebase report complete';
          } else if (msg.stage === 'verification' && verificationStep) {
            setStepStatus(verificationStep, 'complete');
            verificationStep.querySelector('.step-label').textContent = 'Verification complete';
          } else if (msg.stage === 'compile' && compileStep) {
            setStepStatus(compileStep, 'complete');
            const label = compileStep.querySelector('.step-label').textContent;
            if (label === 'Compiling generated .c files to .o objects\u2026') {
              compileStep.querySelector('.step-label').textContent =
                'Per-file compile complete \u2014 artefacts available for download';
            }
          } else if (msg.stage === 'sandbox_build' && sandboxStep) {
            setStepStatus(sandboxStep, 'complete');
            sandboxStep.querySelector('.step-label').textContent = 'Sandbox build complete';
          }
          break;

        case 'file_complete':
          if (fileSteps[msg.file]) {
            setStepStatus(fileSteps[msg.file], 'complete');
            fileSteps[msg.file].querySelector('.step-label').textContent =
              msg.file + ' transformed (' + fmtSize(msg.size) + ')';
          }
          break;

        case 'sandbox_build_result':
          if (sandboxStep) {
            hasSandboxBuild = true;
            const attempts = msg.iterations ? ` in ${msg.iterations} attempt${msg.iterations > 1 ? 's' : ''}` : '';
            sandboxStep.querySelector('.step-label').textContent =
              `Sandbox build succeeded${attempts} \u2014 repository packaged`;
          }
          break;

        case 'info': {
          if (msg.stage === 'analysis') {
            const out = analysisStep.querySelector('.step-output');
            if (out.textContent.startsWith('Connecting to LLM')) out.textContent = '';
            out.textContent += '\n' + msg.message + '\n';
            out.scrollTop = out.scrollHeight;
          } else if (msg.stage === 'gitnexus' && gitnexusStep) {
            const out = gitnexusStep.querySelector('.step-output');
            out.textContent += '\n' + msg.message + '\n';
            out.scrollTop = out.scrollHeight;
          } else if (msg.stage === 'verification' && verificationStep) {
            const out = verificationStep.querySelector('.step-output');
            out.textContent += '\n' + msg.message + '\n';
            out.scrollTop = out.scrollHeight;
          } else if (msg.stage === 'compile' && compileStep) {
            const out = compileStep.querySelector('.step-output');
            if (typeof window.appendStepLog === 'function') {
              window.appendStepLog(out, '\n' + msg.message + '\n');
            } else {
              out.textContent += '\n' + msg.message + '\n';
              out.scrollTop = out.scrollHeight;
            }
          } else if (msg.stage === 'sandbox_build' && sandboxStep) {
            const out = sandboxStep.querySelector('.step-output');
            out.textContent += '\n' + msg.message + '\n';
            out.scrollTop = out.scrollHeight;
          } else if (msg.stage === 'transform' && msg.file && fileSteps[msg.file]) {
            const out = fileSteps[msg.file].querySelector('.step-output');
            out.textContent += '\n' + msg.message + '\n';
            out.scrollTop = out.scrollHeight;
          }
          break;
        }

        case 'error': {
          es.close();
          activeEventSource = null;
          if (msg.file && fileSteps[msg.file]) {
            setStepStatus(fileSteps[msg.file], 'error');
            fileSteps[msg.file].querySelector('.step-label').textContent = msg.file + ' failed';
            const out = fileSteps[msg.file].querySelector('.step-output');
            out.textContent += '\nError: ' + msg.message + '\n';
            out.classList.add('visible');
          } else {
            setStepStatus(analysisStep, 'error');
            analysisStep.querySelector('.step-label').textContent = 'Analysis failed';
            const out = analysisStep.querySelector('.step-output');
            if (out.textContent.startsWith('Connecting to LLM')) out.textContent = '';
            out.textContent += 'Error: ' + msg.message;
            out.classList.add('visible');
          }
          processBtn.disabled = false;
          break;
        }
        case 'paused': {
          es.close();
          activeEventSource = null;
          if (pauseBtn) pauseBtn.style.display = 'none';
          if (resumeBtn) resumeBtn.style.display = '';
          processBtn.disabled = true;
          const target = sandboxStep || verificationStep || analysisStep;
          if (target) {
            const out = target.querySelector('.step-output');
            out.textContent += '\n' + msg.message + '\n';
            out.classList.add('visible');
            out.scrollTop = out.scrollHeight;
          }
          generatedFiles = msg.files || generatedFiles || [];
          showResults(generatedFiles, hasSandboxBuild, true);
          break;
        }
        case 'complete':
          es.close();
          activeEventSource = null;
          if (pauseBtn) pauseBtn.style.display = 'none';
          if (resumeBtn) resumeBtn.style.display = 'none';
          generatedFiles = msg.files || [];
          if (msg.sandbox_build !== undefined) hasSandboxBuild = true;
          showResults(generatedFiles, hasSandboxBuild);
          break;
      }
    };

    es.onerror = function () {
      es.close();
      activeEventSource = null;
      if (generatedFiles.length) {
        showResults(generatedFiles, hasSandboxBuild);
        return;
      }
      setStepStatus(analysisStep, 'error');
      analysisStep.querySelector('.step-label').textContent = 'Analysis stream interrupted';
      const out = analysisStep.querySelector('.step-output');
      if (out.textContent.startsWith('Connecting to LLM')) out.textContent = '';
      out.textContent += '\nError: connection to processing stream was interrupted. Please retry.';
      out.classList.add('visible');
      processBtn.disabled = false;
    };

    resumeCurrentProcess = async function () {
      if (!sessionId || !es) return;
      if (resumeBtn) resumeBtn.disabled = true;
      try {
        await fetch('/api/resume/' + sessionId, { method: 'POST' });
        const onMessage = es.onmessage;
        const onError = es.onerror;
        es = new EventSource(processStreamUrl('/api/process/' + sessionId));
        activeEventSource = es;
        es.onmessage = onMessage;
        es.onerror = onError;
        if (pauseBtn) {
          pauseBtn.style.display = '';
          pauseBtn.disabled = false;
        }
        if (resumeBtn) resumeBtn.style.display = 'none';
        processBtn.disabled = true;
      } finally {
        if (resumeBtn) resumeBtn.disabled = false;
      }
    };
  }

  async function pauseProcessing() {
    if (!sessionId) return;
    if (pauseBtn) pauseBtn.disabled = true;
    try {
      await fetch('/api/pause/' + sessionId, { method: 'POST' });
      if (activeEventSource) {
        // Keep the stream open until the backend acknowledges with a `paused`
        // event so the final checkpoint/report event reaches the UI.
      }
    } catch (err) {
      console.error('Failed to pause processing:', err);
      if (pauseBtn) pauseBtn.disabled = false;
    }
  }

  async function resumeProcessing() {
    if (typeof resumeCurrentProcess === 'function') {
      await resumeCurrentProcess();
    }
  }

  /* ----- Results ------------------------------------------------- */

  function showResults(files, sandboxBuild, allowEmpty) {
    // Defensive: drop any .o objects that might slip through — they are
    // binary and not previewable, but they DO ship in the /api/download zip.
    const previewable = (files || []).filter(function (f) {
      return !f.toLowerCase().endsWith('.o');
    });
    if (!previewable.length && !allowEmpty) return;
    resultsSection.classList.remove('hidden');
    if (previewable.length) convSection.classList.remove('hidden');
    // Preserve the currently active tab if it still exists in the new list.
    const activeTabEl = resultTabs.querySelector('.file-tab.active');
    const previouslyActive = activeTabEl ? activeTabEl.dataset.file : null;
    resultTabs.innerHTML = '';
    let activeIdx = 0;
    if (previouslyActive) {
      const idx = previewable.indexOf(previouslyActive);
      if (idx >= 0) activeIdx = idx;
    }
    previewable.forEach((f, i) => {
      const tab = document.createElement('button');
      tab.className = 'file-tab' + (i === activeIdx ? ' active' : '');
      tab.textContent = f;
      tab.dataset.file = f;
      tab.addEventListener('click', () => loadPreview(f));
      resultTabs.appendChild(tab);
    });
    if (previewable.length) {
      // Only re-load the preview if we don't already have it open.
      if (previewable[activeIdx] !== previouslyActive) {
        loadPreview(previewable[activeIdx]);
      }
    } else {
      codePreview.textContent = 'Processing is paused before generated code is available. Use Download All (ZIP) to download reports captured so far.';
    }
    if (downloadRepoBtn) {
      if (sandboxBuild) {
        downloadRepoBtn.classList.remove('hidden');
      } else {
        downloadRepoBtn.classList.add('hidden');
      }
    }
  }

  async function loadPreview(filename) {
    resultTabs.querySelectorAll('.file-tab').forEach(t =>
      t.classList.toggle('active', t.dataset.file === filename));
    try {
      const r = await fetch('/api/preview/' + sessionId + '/' + filename);
      const d = await r.json();
      codePreview.textContent = d.content;
    } catch {
      codePreview.textContent = 'Error loading file preview.';
    }
  }

  /* ----- Reset --------------------------------------------------- */

  function resetAll() {
    codeFiles = [];
    sourceIcdFile = null;
    targetIcdFile = null;
    repoZipFile = null;
    renderCodeFiles();
    renderIcdFile(null, srcIcdList, dropSourceIcd, 'card-source-icd');
    renderIcdFile(null, tgtIcdList, dropTargetIcd, 'card-target-icd');
    renderZipFile(null, repoZipList, dropRepoZip, 'card-repo-zip');
    updateBtn();
    procSection.classList.add('hidden');
    resultsSection.classList.add('hidden');
    convSection.classList.add('hidden');
    convMessages.innerHTML = '';
    convInput.value = '';
    regenerateBtn.disabled = true;
    pipelineSteps.innerHTML = '';
    resetBtn.style.display = 'none';
    if (pauseBtn) {
      pauseBtn.style.display = 'none';
      pauseBtn.disabled = false;
    }
    if (resumeBtn) {
      resumeBtn.style.display = 'none';
      resumeBtn.disabled = false;
    }
    if (activeEventSource) {
      activeEventSource.close();
      activeEventSource = null;
    }
    resumeCurrentProcess = null;
    if (downloadRepoBtn) downloadRepoBtn.classList.add('hidden');
    inputCode.value = '';
    inputSourceIcd.value = '';
    inputTargetIcd.value = '';
    inputRepoZip.value = '';
  }

  /* ----- Conversation -------------------------------------------- */

  function escapeHtml(text) {
    var d = document.createElement('div');
    d.textContent = text;
    return d.innerHTML;
  }

  function renderConversation(messages) {
    convMessages.innerHTML = '';
    messages.forEach(function (msg) {
      var div = document.createElement('div');
      div.className = 'conversation-msg msg-' + msg.role;
      var roleLabel = msg.role === 'user' ? 'You' : 'System';
      var timeStr = '';
      if (msg.timestamp) {
        try { timeStr = new Date(msg.timestamp).toLocaleTimeString(); } catch (e) { /* ignore */ }
      }
      div.innerHTML =
        '<div class="msg-role">' + escapeHtml(roleLabel) + '</div>' +
        '<pre class="msg-content">' + escapeHtml(msg.content) + '</pre>' +
        (timeStr ? '<div class="msg-time">' + escapeHtml(timeStr) + '</div>' : '');
      convMessages.appendChild(div);
    });
    convMessages.scrollTop = convMessages.scrollHeight;
  }

  async function sendMessage() {
    var text = convInput.value.trim();
    if (!text || !sessionId) return;
    sendMsgBtn.disabled = true;
    try {
      var r = await fetch('/api/conversation/' + sessionId, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: text }),
      });
      var d = await r.json();
      convInput.value = '';
      renderConversation(d.messages);
      regenerateBtn.disabled = false;
    } catch (err) {
      console.error('Failed to send message:', err);
    } finally {
      sendMsgBtn.disabled = false;
    }
  }

  /* ----- Re-generation ------------------------------------------- */

  async function runRegenerate() {
    regenerateBtn.disabled = true;
    sendMsgBtn.disabled = true;
    procSection.classList.remove('hidden');
    pipelineSteps.innerHTML = '';
    resultsSection.classList.add('hidden');

    var regenStep = createStep('regeneration', 'Re-generating code with user feedback\u2026');
    pipelineSteps.appendChild(regenStep);
    setStepStatus(regenStep, 'running');
    regenStep.querySelector('.step-output').classList.add('visible');
    regenStep.querySelector('.step-output').textContent = 'Starting re-generation\u2026';

    var fileSteps = {};
    var generatedFiles = [];
    var verificationStep = null;
    var regenInfoDone = false;
    var sandboxStep = null;
    var hasSandboxBuild = false;

    var es = new EventSource(processStreamUrl('/api/regenerate/' + sessionId));

    es.onmessage = function (event) {
      var msg = JSON.parse(event.data);

      switch (msg.type) {
        case 'token': {
          var step;
          if (msg.stage === 'regeneration') step = regenStep;
          else if (msg.stage === 'verification') step = verificationStep;
          else if (msg.stage === 'sandbox_build') step = sandboxStep;
          else if (msg.stage === 'transform' && msg.file) step = fileSteps[msg.file];
          if (step) {
            var o = step.querySelector('.step-output');
            o.textContent += msg.token;
            o.scrollTop = o.scrollHeight;
          }
          break;
        }
        case 'stage':
          if (msg.stage === 'transform' && msg.file && !fileSteps[msg.file]) {
            if (!regenInfoDone) {
              setStepStatus(regenStep, 'complete');
              regenStep.querySelector('.step-label').textContent = 'Re-generation initialized';
              regenInfoDone = true;
            }
            var s = createStep('file-' + msg.index,
              'Re-generating ' + msg.file + ' (' + (msg.index + 1) + '/' + msg.total + ')');
            pipelineSteps.appendChild(s);
            setStepStatus(s, 'running');
            s.querySelector('.step-output').classList.add('visible');
            fileSteps[msg.file] = s;
          } else if (msg.stage === 'verification' && !verificationStep) {
            verificationStep = createStep('verification', 'Verifying re-generated code\u2026');
            pipelineSteps.appendChild(verificationStep);
            setStepStatus(verificationStep, 'running');
            verificationStep.querySelector('.step-output').classList.add('visible');
          } else if (msg.stage === 'sandbox_build' && !sandboxStep) {
            sandboxStep = createStep('sandbox-build',
              'Building re-generated code in sandbox environment\u2026');
            pipelineSteps.appendChild(sandboxStep);
            setStepStatus(sandboxStep, 'running');
            sandboxStep.querySelector('.step-output').classList.add('visible');
          }
          break;

        case 'stage_complete':
          if (msg.stage === 'verification' && verificationStep) {
            setStepStatus(verificationStep, 'complete');
            verificationStep.querySelector('.step-label').textContent = 'Verification complete';
          } else if (msg.stage === 'sandbox_build' && sandboxStep) {
            setStepStatus(sandboxStep, 'complete');
            sandboxStep.querySelector('.step-label').textContent = 'Sandbox build complete';
          }
          break;

        case 'file_complete':
          if (fileSteps[msg.file]) {
            setStepStatus(fileSteps[msg.file], 'complete');
            fileSteps[msg.file].querySelector('.step-label').textContent =
              msg.file + ' re-generated (' + fmtSize(msg.size) + ')';
          }
          break;

        case 'sandbox_build_result':
          if (sandboxStep) {
            hasSandboxBuild = true;
            var reAttempts = msg.iterations ? ` in ${msg.iterations} attempt${msg.iterations > 1 ? 's' : ''}` : '';
            sandboxStep.querySelector('.step-label').textContent =
              `Sandbox build succeeded${reAttempts} \u2014 repository packaged`;
          }
          break;

        case 'info': {
          var target;
          if (msg.stage === 'regeneration') target = regenStep;
          else if (msg.stage === 'verification' && verificationStep) target = verificationStep;
          else if (msg.stage === 'sandbox_build' && sandboxStep) target = sandboxStep;
          else if (msg.stage === 'transform' && msg.file && fileSteps[msg.file]) target = fileSteps[msg.file];
          if (target) {
            var out = target.querySelector('.step-output');
            out.textContent += '\n' + msg.message + '\n';
            out.scrollTop = out.scrollHeight;
          }
          break;
        }

        case 'error':
          es.close();
          if (msg.file && fileSteps[msg.file]) {
            setStepStatus(fileSteps[msg.file], 'error');
            fileSteps[msg.file].querySelector('.step-label').textContent = msg.file + ' failed';
            var outE = fileSteps[msg.file].querySelector('.step-output');
            outE.textContent += '\nError: ' + msg.message + '\n';
            outE.classList.add('visible');
          } else {
            setStepStatus(regenStep, 'error');
            regenStep.querySelector('.step-label').textContent = 'Re-generation failed';
            var outR = regenStep.querySelector('.step-output');
            outR.textContent += '\nError: ' + msg.message + '\n';
            outR.classList.add('visible');
          }
          regenerateBtn.disabled = false;
          sendMsgBtn.disabled = false;
          break;

        case 'complete':
          es.close();
          generatedFiles = msg.files || [];
          if (msg.sandbox_build !== undefined) hasSandboxBuild = true;
          showResults(generatedFiles, hasSandboxBuild);
          regenerateBtn.disabled = true;
          sendMsgBtn.disabled = false;
          refreshConversation();
          break;
      }
    };

    es.onerror = function () {
      es.close();
      if (generatedFiles.length) {
        showResults(generatedFiles);
        regenerateBtn.disabled = true;
        sendMsgBtn.disabled = false;
        return;
      }
      setStepStatus(regenStep, 'error');
      regenStep.querySelector('.step-label').textContent = 'Re-generation stream interrupted';
      var outErr = regenStep.querySelector('.step-output');
      outErr.textContent += '\nError: connection interrupted. Please retry.';
      outErr.classList.add('visible');
      regenerateBtn.disabled = false;
      sendMsgBtn.disabled = false;
    };
  }

  async function refreshConversation() {
    if (!sessionId) return;
    try {
      var r = await fetch('/api/conversation/' + sessionId);
      var d = await r.json();
      renderConversation(d.messages);
    } catch (e) { /* ignore */ }
  }

  /* ----- Wire up events ------------------------------------------ */

  setupDropZone(dropCode, inputCode, files => {
    files.filter(f => f.name.endsWith('.c') || f.name.endsWith('.h'))
         .forEach(f => { if (!codeFiles.find(c => c.name === f.name)) codeFiles.push(f); });
    renderCodeFiles();
    updateBtn();
  });

  setupDropZone(dropSourceIcd, inputSourceIcd, files => {
    const pdf = files.find(f => f.name.toLowerCase().endsWith('.pdf'));
    if (pdf) { sourceIcdFile = pdf; renderIcdFile(pdf, srcIcdList, dropSourceIcd, 'card-source-icd'); updateBtn(); }
  });

  setupDropZone(dropTargetIcd, inputTargetIcd, files => {
    const pdf = files.find(f => f.name.toLowerCase().endsWith('.pdf'));
    if (pdf) { targetIcdFile = pdf; renderIcdFile(pdf, tgtIcdList, dropTargetIcd, 'card-target-icd'); updateBtn(); }
  });

  setupDropZone(dropRepoZip, inputRepoZip, files => {
    const zip = files.find(f => f.name.toLowerCase().endsWith('.zip'));
    if (zip) { repoZipFile = zip; renderZipFile(zip, repoZipList, dropRepoZip, 'card-repo-zip'); updateBtn(); }
  });

  document.addEventListener('dragover', e => e.preventDefault());
  document.addEventListener('drop', e => e.preventDefault());

  processBtn.addEventListener('click', runProcess);
  resetBtn.addEventListener('click', resetAll);
  if (pauseBtn) pauseBtn.addEventListener('click', pauseProcessing);
  if (resumeBtn) resumeBtn.addEventListener('click', resumeProcessing);
  downloadBtn.addEventListener('click', () => {
    if (sessionId) window.location.href = '/api/download/' + sessionId;
  });
  if (downloadRepoBtn) {
    downloadRepoBtn.addEventListener('click', () => {
      if (sessionId) window.location.href = '/api/download-repo/' + sessionId;
    });
  }
  sendMsgBtn.addEventListener('click', sendMessage);
  regenerateBtn.addEventListener('click', runRegenerate);
  convInput.addEventListener('keydown', function (e) {
    if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) sendMessage();
  });

  initSession();
})();
