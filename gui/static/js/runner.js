// ═══ Pipeline Runner (SSE) ═══

const Runner = {
  eventSource: null,
  steps: [],

  async start(params) {
    Runner._beginRun();

    try {
      const res = await fetch('/api/run', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(params)
      });
      const data = await res.json();
      if (data.error) {
        Runner.appendLog(`ERROR: ${data.error}\n`);
      }
    } catch (e) {
      Runner.appendLog(`Failed to start pipeline: ${e.message}\n`);
    }
  },

  async startPrecomputed(params) {
    Runner._beginRun();

    try {
      const res = await fetch('/api/run-precomputed', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(params)
      });
      const data = await res.json();
      if (data.error) {
        Runner.appendLog(`ERROR: ${data.error}\n`);
      }
    } catch (e) {
      Runner.appendLog(`Failed to start pipeline: ${e.message}\n`);
    }
  },

  _beginRun() {
    App.enableNav('running');
    App.showView('running');
    App.disableNav('setup');
    document.getElementById('log-output').textContent = '';
    Runner.connectSSE();
  },

  connectSSE() {
    Runner.eventSource = new EventSource('/api/sse');

    Runner.eventSource.addEventListener('pipeline_start', (e) => {
      const data = JSON.parse(e.data);
      Runner.steps = data.steps || [];
      Runner.renderSteps();
      Runner.appendLog(`Pipeline started\n`);
      Runner.appendLog(`  Dataset: ${data.dataset_dir}\n`);
      Runner.appendLog(`  Checkpoint: ${data.checkpoint}${data.checkpoint_found ? ' [found]' : ' [NOT FOUND]'}\n\n`);
    });

    Runner.eventSource.addEventListener('step_start', (e) => {
      const data = JSON.parse(e.data);
      Runner.setStepStatus(data.step, 'current');
      Runner.appendLog(`━━━ Step ${data.step}: ${data.name} ━━━\n`);
    });

    Runner.eventSource.addEventListener('step_log', (e) => {
      const data = JSON.parse(e.data);
      Runner.appendLog(data.log);
    });

    Runner.eventSource.addEventListener('step_done', (e) => {
      const data = JSON.parse(e.data);
      Runner.setStepStatus(data.step, 'done');
    });

    Runner.eventSource.addEventListener('step_error', (e) => {
      const data = JSON.parse(e.data);
      Runner.setStepStatus(data.step, 'error');
      Runner.appendLog(`  ERROR: ${data.error}\n`);
    });

    Runner.eventSource.addEventListener('pipeline_done', (e) => {
      const data = JSON.parse(e.data);
      Runner.eventSource.close();
      Runner.eventSource = null;

      if (data.success) {
        Runner.appendLog(`\n✓ Pipeline complete\n`);
        App.enableNav('results');
        App.enableNav('setup');
        // Auto-switch to results after a beat
        setTimeout(() => {
          App.showView('results');
          Results.load();
        }, 800);
      } else {
        Runner.appendLog(`\n✗ Pipeline failed: ${data.error || 'unknown error'}\n`);
        App.enableNav('setup');
        // Still try to show partial results
        App.enableNav('results');
      }
    });

    Runner.eventSource.onerror = () => {
      Runner.appendLog('\n[connection lost]\n');
      Runner.eventSource.close();
      Runner.eventSource = null;
      App.enableNav('setup');
    };
  },

  renderSteps() {
    const list = document.getElementById('steps-list');
    list.innerHTML = Runner.steps.map(s => `
      <div class="step-item${s.skip ? ' skip' : ''}" id="step-${s.id}">
        <span class="step-icon" id="step-icon-${s.id}">${s.skip ? '⊘' : '○'}</span>
        <span class="step-num">${s.id}</span>
        <span class="step-name">${s.name}${s.skip ? ' (pre-computed)' : ''}</span>
      </div>
    `).join('');
  },

  setStepStatus(stepId, status) {
    const item = document.getElementById(`step-${stepId}`);
    const icon = document.getElementById(`step-icon-${stepId}`);
    if (!item || !icon) return;

    // Clear previous states
    item.classList.remove('current', 'done', 'error');
    item.classList.add(status);

    if (status === 'current') icon.innerHTML = '<span class="spinner"></span>';
    else if (status === 'done') icon.textContent = '✓';
    else if (status === 'error') icon.textContent = '✗';
  },

  appendLog(text) {
    const log = document.getElementById('log-output');
    log.textContent += text;
    log.scrollTop = log.scrollHeight;
  },

  cancel() {
    fetch('/api/cancel', { method: 'POST' });
    Runner.appendLog('\n[cancelling...]\n');
  }
};

// Cancel button
document.getElementById('btn-cancel').addEventListener('click', () => Runner.cancel());
