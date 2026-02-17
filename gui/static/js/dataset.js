// ═══ Dataset Browser (Data Tab) ═══

const Dataset = {
  data: null,
  sortCol: 'title',
  sortAsc: true,
  filter: '',
  loaded: false,
  pendingRun: null, // Set by setup.js before navigating here

  // Called by setup.js after user picks a run mode
  proceedToData() {
    App.enableNav('data');
    App.showView('data');
    Dataset.load();
  },

  async load() {
    if (Dataset.loaded) {
      Dataset.updateHint();
      return;
    }

    const loading = document.getElementById('data-loading');
    const content = document.getElementById('data-content');
    loading.classList.remove('hidden');

    try {
      const res = await fetch('/api/dataset-info');
      const info = await res.json();
      if (!info.available) {
        loading.innerHTML = '<span style="color:var(--text-muted)">No dataset found. Download sample data or run the pipeline first.</span>';
        return;
      }

      Dataset.data = info;
      Dataset.loaded = true;

      loading.classList.add('hidden');
      content.classList.remove('hidden');

      Dataset.renderStats();
      Dataset.renderCharts();
      Dataset.renderTable();
      Dataset.initControls();
      Dataset.updateHint();
    } catch (e) {
      loading.innerHTML = `<span style="color:var(--danger)">Failed to load dataset: ${e.message}</span>`;
    }
  },

  updateHint() {
    const hint = document.getElementById('data-mode-hint');
    if (!hint) return;
    const run = Dataset.pendingRun;
    if (!run) {
      hint.textContent = '';
      return;
    }
    if (run.mode === 'precomputed') {
      hint.textContent = 'using pre-computed encodings — steps 1-4 will be skipped';
    } else {
      hint.textContent = `using ${run.dataset_dir}`;
    }
  },

  renderStats() {
    const d = Dataset.data;
    document.getElementById('dataset-stats').innerHTML = `
      <div class="ds-stat"><span class="ds-stat-val">${d.unique_songs}</span><span class="ds-stat-lbl">songs</span></div>
      <div class="ds-stat"><span class="ds-stat-val">${d.unique_mappers}</span><span class="ds-stat-lbl">mappers</span></div>
      <div class="ds-stat"><span class="ds-stat-val">${d.unique_artists}</span><span class="ds-stat-lbl">artists</span></div>
      <div class="ds-stat"><span class="ds-stat-val">${d.representatives}</span><span class="ds-stat-lbl">beatmaps (reps)</span></div>
      <div class="ds-stat"><span class="ds-stat-val">${d.total_beatmaps}</span><span class="ds-stat-lbl">total diffs</span></div>
    `;
  },

  // ─── D3 Charts ───

  renderCharts() {
    Dataset.chartMapperDist();
    Dataset.chartODDist();
    Dataset.chartArtistDist();
  },

  chartMapperDist() {
    const dist = Dataset.data.mapper_distribution;
    const entries = Object.entries(dist).map(([k, v]) => ({ label: k + ' mappers', count: v }));

    const container = d3.select('#chart-mapper-dist');
    container.selectAll('*').remove();

    const cw = container.node().getBoundingClientRect().width || 300;
    const margin = { top: 8, right: 12, bottom: 30, left: 44 };
    const w = cw - margin.left - margin.right;
    const h = 140 - margin.top - margin.bottom;

    const svg = container.append('svg')
      .attr('width', cw).attr('height', 140)
      .append('g').attr('transform', `translate(${margin.left},${margin.top})`);

    const x = d3.scaleBand().domain(entries.map(d => d.label)).range([0, w]).padding(0.3);
    const y = d3.scaleLinear().domain([0, d3.max(entries, d => d.count)]).nice().range([h, 0]);

    svg.append('g').call(d3.axisLeft(y).ticks(4).tickSize(-w))
      .call(g => g.select('.domain').remove())
      .call(g => g.selectAll('.tick line').attr('stroke', '#1a1c20'));

    svg.selectAll('rect').data(entries).join('rect')
      .attr('x', d => x(d.label)).attr('y', d => y(d.count))
      .attr('width', x.bandwidth()).attr('height', d => h - y(d.count))
      .attr('fill', '#8ab4d4').attr('rx', 2);

    svg.selectAll('.bar-label').data(entries).join('text')
      .attr('class', 'bar-label')
      .attr('x', d => x(d.label) + x.bandwidth() / 2)
      .attr('y', d => y(d.count) - 4)
      .attr('text-anchor', 'middle')
      .style('font-size', '10px').style('fill', 'var(--text-mid)')
      .text(d => d.count);

    svg.append('g').attr('transform', `translate(0,${h})`)
      .call(d3.axisBottom(x).tickSize(0))
      .call(g => g.select('.domain').remove())
      .selectAll('text').style('font-size', '9px');
  },

  chartODDist() {
    const table = Dataset.data.table;
    const odVals = table.map(r => r.od).filter(v => v != null);

    const bins = d3.bin().domain([0, 10.5]).thresholds(d3.range(0, 10.5, 0.5))(odVals);

    const container = d3.select('#chart-od-dist');
    container.selectAll('*').remove();

    const cw = container.node().getBoundingClientRect().width || 300;
    const margin = { top: 8, right: 12, bottom: 30, left: 44 };
    const w = cw - margin.left - margin.right;
    const h = 140 - margin.top - margin.bottom;

    const svg = container.append('svg')
      .attr('width', cw).attr('height', 140)
      .append('g').attr('transform', `translate(${margin.left},${margin.top})`);

    const x = d3.scaleLinear().domain([0, 10.5]).range([0, w]);
    const y = d3.scaleLinear().domain([0, d3.max(bins, d => d.length)]).nice().range([h, 0]);

    svg.append('g').call(d3.axisLeft(y).ticks(4).tickSize(-w))
      .call(g => g.select('.domain').remove())
      .call(g => g.selectAll('.tick line').attr('stroke', '#1a1c20'));

    svg.selectAll('rect').data(bins).join('rect')
      .attr('x', d => x(d.x0) + 1).attr('y', d => y(d.length))
      .attr('width', d => Math.max(0, x(d.x1) - x(d.x0) - 2))
      .attr('height', d => h - y(d.length))
      .attr('fill', d => d.x0 >= 8 ? '#8ab4d4' : '#1a2530').attr('rx', 1);

    svg.append('g').attr('transform', `translate(0,${h})`)
      .call(d3.axisBottom(x).ticks(6).tickFormat(d => d.toFixed(0)))
      .call(g => g.select('.domain').remove());

    svg.append('text')
      .attr('x', w / 2).attr('y', h + 24)
      .attr('text-anchor', 'middle')
      .style('font-size', '9px').style('fill', 'var(--text-muted)')
      .text('Overall Difficulty');
  },

  chartArtistDist() {
    const table = Dataset.data.table;
    const counts = {};
    const seen = new Set();
    table.forEach(r => {
      const key = r.artist + '|' + r.song_group_idx;
      if (!seen.has(key)) {
        seen.add(key);
        counts[r.artist] = (counts[r.artist] || 0) + 1;
      }
    });

    const top = Object.entries(counts)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 15)
      .map(([artist, count]) => ({ artist, count }));

    const container = d3.select('#chart-artist-dist');
    container.selectAll('*').remove();

    const cw = container.node().getBoundingClientRect().width || 300;
    const margin = { top: 8, right: 40, bottom: 8, left: 8 };
    const rowH = 20;
    const h = top.length * rowH + margin.top + margin.bottom;
    const w = cw - margin.left - margin.right;

    const svg = container.append('svg')
      .attr('width', cw).attr('height', h)
      .append('g').attr('transform', `translate(${margin.left},${margin.top})`);

    const x = d3.scaleLinear().domain([0, d3.max(top, d => d.count)]).range([0, w - 50]);

    top.forEach((d, i) => {
      const y = i * rowH;

      svg.append('rect')
        .attr('x', 0).attr('y', y + 2)
        .attr('width', x(d.count)).attr('height', rowH - 4)
        .attr('fill', '#1a2530').attr('rx', 2);

      svg.append('text')
        .attr('x', 4).attr('y', y + rowH / 2 + 3)
        .style('font-size', '10px').style('fill', 'var(--text-mid)')
        .text(d.artist.length > 28 ? d.artist.slice(0, 26) + '...' : d.artist);

      svg.append('text')
        .attr('x', x(d.count) + 6).attr('y', y + rowH / 2 + 3)
        .style('font-size', '10px').style('fill', 'var(--accent)')
        .text(d.count);
    });
  },

  // ─── Controls ───

  initControls() {
    document.getElementById('dataset-search').addEventListener('input', (e) => {
      Dataset.filter = e.target.value.toLowerCase();
      Dataset.renderTable();
    });

    document.querySelectorAll('#dataset-table th[data-sort]').forEach(th => {
      th.addEventListener('click', () => {
        const col = th.dataset.sort;
        if (Dataset.sortCol === col) {
          Dataset.sortAsc = !Dataset.sortAsc;
        } else {
          Dataset.sortCol = col;
          Dataset.sortAsc = true;
        }
        document.querySelectorAll('#dataset-table th').forEach(h => h.classList.remove('sort-asc', 'sort-desc'));
        th.classList.add(Dataset.sortAsc ? 'sort-asc' : 'sort-desc');
        Dataset.renderTable();
      });
    });

    // Confirm / Back buttons
    document.getElementById('btn-data-confirm').addEventListener('click', () => {
      const run = Dataset.pendingRun;
      if (!run) return;
      if (run.mode === 'precomputed') {
        Runner.startPrecomputed({ checkpoint: run.checkpoint });
      } else {
        Runner.start(run);
      }
    });

  },

  // ─── Table ───

  renderTable() {
    let rows = Dataset.data.table;

    if (Dataset.filter) {
      const q = Dataset.filter;
      rows = rows.filter(r =>
        (r.title || '').toLowerCase().includes(q) ||
        (r.artist || '').toLowerCase().includes(q) ||
        (r.creator || '').toLowerCase().includes(q) ||
        (r.version || '').toLowerCase().includes(q)
      );
    }

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

    document.getElementById('dataset-count').textContent =
      `${rows.length} of ${Dataset.data.table.length} beatmaps`;

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
