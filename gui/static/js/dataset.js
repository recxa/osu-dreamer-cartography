// ═══ Dataset Browser ═══

const Dataset = {
  data: null,
  sortCol: 'title',
  sortAsc: true,
  filter: '',

  async load() {
    try {
      const res = await fetch('/api/dataset-info');
      const info = await res.json();
      if (!info.available) return;

      Dataset.data = info;
      document.getElementById('dataset-card').classList.remove('hidden');
      Dataset.renderStats();
      Dataset.initControls();
    } catch (e) {
      // Silently fail — dataset card stays hidden
    }
  },

  renderStats() {
    const d = Dataset.data;
    const el = document.getElementById('dataset-stats');

    el.innerHTML = `
      <div class="ds-stat"><span class="ds-stat-val">${d.unique_songs}</span><span class="ds-stat-lbl">songs</span></div>
      <div class="ds-stat"><span class="ds-stat-val">${d.unique_mappers}</span><span class="ds-stat-lbl">mappers</span></div>
      <div class="ds-stat"><span class="ds-stat-val">${d.unique_artists}</span><span class="ds-stat-lbl">artists</span></div>
      <div class="ds-stat"><span class="ds-stat-val">${d.representatives}</span><span class="ds-stat-lbl">beatmaps (reps)</span></div>
      <div class="ds-stat"><span class="ds-stat-val">${d.total_beatmaps}</span><span class="ds-stat-lbl">total diffs</span></div>
    `;
  },

  initControls() {
    document.getElementById('btn-browse-dataset').addEventListener('click', () => {
      const browser = document.getElementById('dataset-browser');
      const btn = document.getElementById('btn-browse-dataset');
      const isHidden = browser.classList.contains('hidden');
      browser.classList.toggle('hidden');
      btn.textContent = isHidden ? 'hide dataset' : 'browse full dataset';
      if (isHidden) Dataset.renderTable();
    });

    document.getElementById('dataset-search').addEventListener('input', (e) => {
      Dataset.filter = e.target.value.toLowerCase();
      Dataset.renderTable();
    });

    // Column sorting
    document.querySelectorAll('#dataset-table th[data-sort]').forEach(th => {
      th.addEventListener('click', () => {
        const col = th.dataset.sort;
        if (Dataset.sortCol === col) {
          Dataset.sortAsc = !Dataset.sortAsc;
        } else {
          Dataset.sortCol = col;
          Dataset.sortAsc = true;
        }
        // Update sort indicators
        document.querySelectorAll('#dataset-table th').forEach(h => h.classList.remove('sort-asc', 'sort-desc'));
        th.classList.add(Dataset.sortAsc ? 'sort-asc' : 'sort-desc');
        Dataset.renderTable();
      });
    });
  },

  renderTable() {
    let rows = Dataset.data.table;

    // Filter
    if (Dataset.filter) {
      const q = Dataset.filter;
      rows = rows.filter(r =>
        (r.title || '').toLowerCase().includes(q) ||
        (r.artist || '').toLowerCase().includes(q) ||
        (r.creator || '').toLowerCase().includes(q) ||
        (r.version || '').toLowerCase().includes(q)
      );
    }

    // Sort
    const col = Dataset.sortCol;
    rows = [...rows].sort((a, b) => {
      let va = a[col], vb = b[col];
      if (va == null) va = '';
      if (vb == null) vb = '';
      if (typeof va === 'number' && typeof vb === 'number') {
        return Dataset.sortAsc ? va - vb : vb - va;
      }
      va = String(va).toLowerCase();
      vb = String(vb).toLowerCase();
      return Dataset.sortAsc ? va.localeCompare(vb) : vb.localeCompare(va);
    });

    // Count
    document.getElementById('dataset-count').textContent =
      `${rows.length} of ${Dataset.data.table.length} beatmaps`;

    // Render (virtualized: cap at 200 rows for performance, rest via scroll hint)
    const maxRows = 200;
    const tbody = document.getElementById('dataset-tbody');
    const fragment = document.createDocumentFragment();

    rows.slice(0, maxRows).forEach(r => {
      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td class="cell-title" title="${Dataset.esc(r.title)}">${Dataset.esc(r.title)}</td>
        <td class="cell-artist" title="${Dataset.esc(r.artist)}">${Dataset.esc(r.artist)}</td>
        <td>${Dataset.esc(r.creator)}</td>
        <td>${Dataset.esc(r.version)}</td>
        <td class="num">${r.od != null ? r.od.toFixed(1) : '—'}</td>
        <td class="num">${r.ar != null ? r.ar.toFixed(1) : '—'}</td>
        <td class="num">${r.num_mappers || '—'}</td>
        <td class="num">${r.beatmapset_id ? `<a href="https://osu.ppy.sh/beatmapsets/${r.beatmapset_id}" target="_blank" rel="noopener" class="ds-link">osu!</a>` : '—'}</td>
      `;
      fragment.appendChild(tr);
    });

    tbody.innerHTML = '';
    tbody.appendChild(fragment);

    if (rows.length > maxRows) {
      const tr = document.createElement('tr');
      tr.innerHTML = `<td colspan="8" class="dataset-more">... and ${rows.length - maxRows} more (use search to narrow)</td>`;
      tbody.appendChild(tr);
    }
  },

  esc(s) {
    if (!s) return '';
    return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
  }
};
