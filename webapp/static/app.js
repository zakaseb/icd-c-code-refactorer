(function () {
  'use strict';

  let sessionId = null;
  let codeFiles = [];
  let sourceIcdFile = null;
  let targetIcdFile = null;

  /* ----- DOM refs ------------------------------------------------ */
  const dropCode       = document.getElementById('drop-code');
  const dropSourceIcd  = document.getElementById('drop-source-icd');
  const dropTargetIcd  = document.getElementById('drop-target-icd');
  const inputCode      = document.getElementById('input-code');
  const inputSourceIcd = document.getElementById('input-source-icd');
  const inputTargetIcd = document.getElementById('input-target-icd');
  const codeFileList   = document.getElementById('code-file-list');
  const srcIcdList     = document.getElementById('source-icd-file-list');
  const tgtIcdList     = document.getElementById('target-icd-file-list');
  const processBtn     = document.getElementById('process-btn');
  const resetBtn       = document.getElementById('reset-btn');
  const procSection    = document.getElementById('processing-section');
  const pipelineSteps  = document.getElementById('pipeline-steps');
  const resultsSection = document.getElementById('results-section');
  const resultTabs     = document.getElementById('result-tabs');
  const codePreview    = document.getElementById('code-preview');
  const downloadBtn    = document.getElementById('download-btn');

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
    procSection.classList.remove('hidden');
    resultsSection.classList.add('hidden');
    pipelineSteps.innerHTML = '';

    /* 1. Upload all resources */
    const upStep = createStep('upload', 'Uploading files\u2026');
    pipelineSteps.appendChild(upStep);
    setStepStatus(upStep, 'running');

    try {
      await initSession();
      await Promise.all([
        uploadCodeFiles(),
        uploadIcd('source-icd', sourceIcdFile),
        uploadIcd('target-icd', targetIcdFile),
      ]);
      setStepStatus(upStep, 'complete');
      upStep.querySelector('.step-label').textContent = 'Files uploaded';
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

    const es = new EventSource('/api/process/' + sessionId);

    es.onmessage = function (event) {
      const msg = JSON.parse(event.data);

      switch (msg.type) {
        case 'token': {
          let step;
          if (msg.stage === 'analysis') step = analysisStep;
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
          }
          break;

        case 'stage_complete':
          if (msg.stage === 'analysis') {
            setStepStatus(analysisStep, 'complete');
            analysisStep.querySelector('.step-label').textContent = 'ICD analysis complete';
          }
          break;

        case 'file_complete':
          if (fileSteps[msg.file]) {
            setStepStatus(fileSteps[msg.file], 'complete');
            fileSteps[msg.file].querySelector('.step-label').textContent =
              msg.file + ' transformed (' + fmtSize(msg.size) + ')';
          }
          break;

        case 'info': {
          if (msg.stage === 'analysis') {
            const out = analysisStep.querySelector('.step-output');
            if (out.textContent.startsWith('Connecting to LLM')) out.textContent = '';
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
        case 'complete':
          es.close();
          generatedFiles = msg.files || [];
          showResults(generatedFiles);
          break;
      }
    };

    es.onerror = function () {
      es.close();
      if (generatedFiles.length) {
        showResults(generatedFiles);
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
  }

  /* ----- Results ------------------------------------------------- */

  function showResults(files) {
    if (!files.length) return;
    resultsSection.classList.remove('hidden');
    resultTabs.innerHTML = '';
    files.forEach((f, i) => {
      const tab = document.createElement('button');
      tab.className = 'file-tab' + (i === 0 ? ' active' : '');
      tab.textContent = f;
      tab.dataset.file = f;
      tab.addEventListener('click', () => loadPreview(f));
      resultTabs.appendChild(tab);
    });
    loadPreview(files[0]);
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
    renderCodeFiles();
    renderIcdFile(null, srcIcdList, dropSourceIcd, 'card-source-icd');
    renderIcdFile(null, tgtIcdList, dropTargetIcd, 'card-target-icd');
    updateBtn();
    procSection.classList.add('hidden');
    resultsSection.classList.add('hidden');
    pipelineSteps.innerHTML = '';
    resetBtn.style.display = 'none';
    inputCode.value = '';
    inputSourceIcd.value = '';
    inputTargetIcd.value = '';
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

  document.addEventListener('dragover', e => e.preventDefault());
  document.addEventListener('drop', e => e.preventDefault());

  processBtn.addEventListener('click', runProcess);
  resetBtn.addEventListener('click', resetAll);
  downloadBtn.addEventListener('click', () => {
    if (sessionId) window.location.href = '/api/download/' + sessionId;
  });

  initSession();
})();
