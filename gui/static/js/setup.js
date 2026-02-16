// ═══ Setup View ═══

const Setup = {
  datasetDir: '',
  checkpointPath: '',

  init() {
    document.getElementById('btn-pick-folder').addEventListener('click', () => Setup.pickFolder());
    document.getElementById('btn-pick-checkpoint').addEventListener('click', () => Setup.pickCheckpoint());
    document.getElementById('btn-run').addEventListener('click', () => Setup.startRun());

    document.getElementById('param-threshold').addEventListener('input', (e) => {
      document.getElementById('threshold-display').textContent = parseFloat(e.target.value).toFixed(2);
    });
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
    // Simple count: just show the path for now (actual counting happens server-side in step 1)
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

    Runner.start({
      dataset_dir: Setup.datasetDir,
      checkpoint: Setup.checkpointPath,
      threshold: parseFloat(document.getElementById('param-threshold').value)
    });
  }
};

Setup.init();
