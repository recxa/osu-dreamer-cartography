// ═══ Setup View ═══

const Setup = {
  datasetDir: '',
  checkpointPath: '',
  precomputed: null,

  async init() {
    document.getElementById('btn-pick-folder').addEventListener('click', () => Setup.pickFolder());
    document.getElementById('btn-pick-checkpoint').addEventListener('click', () => Setup.pickCheckpoint());
    document.getElementById('btn-run').addEventListener('click', () => Setup.startRun());

    document.getElementById('param-threshold').addEventListener('input', (e) => {
      document.getElementById('threshold-display').textContent = parseFloat(e.target.value).toFixed(2);
    });

    // Check for pre-computed data on load
    await Setup.checkPrecomputed();

  },

  async checkPrecomputed() {
    try {
      const res = await fetch('/api/precomputed');
      Setup.precomputed = await res.json();
      Setup.renderSampleCard();
    } catch (e) {
      // API not available, hide the card
      document.getElementById('sample-data-card').classList.add('hidden');
    }
  },

  renderSampleCard() {
    const pc = Setup.precomputed;
    const el = document.getElementById('sample-status');

    if (pc.can_skip_encoding && pc.can_skip_vae) {
      // Data is ready — show "proceed with sample data" button
      el.innerHTML = `
        <div class="sample-ready">
          <p>
            <strong>Sample data available.</strong>
            Pre-computed encodings found (${pc.n_9dim} beatmaps, 9-dim + 32-dim latent).
            Steps 1-4 and VAE encoding will be skipped.
          </p>
          <div>
            <button class="btn btn-primary" id="btn-run-sample">proceed with sample data</button>
          </div>
        </div>
      `;
      document.getElementById('btn-run-sample').addEventListener('click', () => Setup.runPrecomputed());
    } else if (pc.can_skip_encoding) {
      el.innerHTML = `
        <div class="sample-ready">
          <p>
            <strong>Partial data available.</strong>
            ${pc.n_9dim} 9-dim encodings found. Missing latent encodings — VAE step will run.
          </p>
          <div>
            <button class="btn btn-primary" id="btn-run-sample">proceed with available data</button>
          </div>
        </div>
      `;
      document.getElementById('btn-run-sample').addEventListener('click', () => Setup.runPrecomputed());
    } else {
      // No data — show download button
      el.innerHTML = `
        <div class="sample-info">
          <p>
            <strong>No beatmap data found.</strong>
            Download our pre-computed test beatmap encodings (~900 MB) to run the analysis immediately,
            or use your own .osz files below.
          </p>
          <button class="btn btn-outline" id="btn-download-sample">download sample data</button>
        </div>
      `;
      document.getElementById('btn-download-sample').addEventListener('click', () => Setup.downloadSample());
    }
  },

  async downloadSample() {
    const el = document.getElementById('sample-status');
    el.innerHTML = `
      <div class="download-progress">
        <span class="spinner"></span>
        <span id="download-msg">Starting download...</span>
      </div>
    `;

    // Connect SSE for download progress
    const sse = new EventSource('/api/sse');

    sse.addEventListener('download_progress', (e) => {
      const data = JSON.parse(e.data);
      const msg = document.getElementById('download-msg');
      if (msg) msg.textContent = data.message;
    });

    sse.addEventListener('download_done', (e) => {
      const data = JSON.parse(e.data);
      sse.close();
      if (data.success) {
        Setup.checkPrecomputed();
      } else {
        el.innerHTML = `
          <div class="sample-info">
            <p style="color:var(--danger)"><strong>Download failed.</strong> Check your connection and try again.</p>
            <button class="btn btn-outline" id="btn-download-sample">retry</button>
          </div>
        `;
        document.getElementById('btn-download-sample').addEventListener('click', () => Setup.downloadSample());
      }
    });

    sse.onerror = () => {
      sse.close();
    };

    // Trigger download
    fetch('/api/download-sample', { method: 'POST' });
  },

  runPrecomputed() {
    // Go to data tab first for review, then user confirms to run
    Dataset.pendingRun = { mode: 'precomputed', checkpoint: Setup.checkpointPath };
    Dataset.proceedToData();
  },

  async pickFolder() {
    const status = document.getElementById('setup-status');
    status.textContent = 'opening folder picker...';

    try {
      const res = await fetch('/api/pick-folder', { method: 'POST' });
      const data = await res.json();
      if (data.path) {
        Setup.datasetDir = data.path;
        document.getElementById('dataset-dir').value = data.path;
        status.textContent = '';
        Setup.validateForm();
        Setup.previewDataset(data.path);
      } else {
        status.textContent = 'no folder selected';
      }
    } catch (e) {
      status.textContent = 'error opening picker';
    }
  },

  async pickCheckpoint() {
    const status = document.getElementById('setup-status');
    status.textContent = 'opening file picker...';

    try {
      const res = await fetch('/api/pick-file', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ filetypes: [['Checkpoint', '*.ckpt'], ['All files', '*.*']] })
      });
      const data = await res.json();
      if (data.path) {
        Setup.checkpointPath = data.path;
        document.getElementById('checkpoint-path').value = data.path;
        status.textContent = '';
      } else {
        status.textContent = 'no file selected';
      }
    } catch (e) {
      status.textContent = 'error opening picker';
    }
  },

  previewDataset(dirPath) {
    const preview = document.getElementById('dataset-preview');
    preview.classList.remove('hidden');
    preview.innerHTML = `
      <span class="preview-stat">path: <strong>${dirPath}</strong></span>
    `;
  },

  validateForm() {
    const btn = document.getElementById('btn-run');
    btn.disabled = !Setup.datasetDir;
  },

  startRun() {
    if (!Setup.datasetDir) return;

    // Go to data tab first for review, then user confirms to run
    Dataset.pendingRun = {
      mode: 'full',
      dataset_dir: Setup.datasetDir,
      checkpoint: Setup.checkpointPath,
      threshold: parseFloat(document.getElementById('param-threshold').value)
    };
    Dataset.proceedToData();
  }
};

Setup.init();
