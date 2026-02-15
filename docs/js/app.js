// ═══ Latent Space Cartography ═══

const state = {
  trackA: null,
  trackB: null,
  synthesis: null,
  cv: null,
  heatmap: { ordering: 'ranked', threshold: 0, showValues: false }
};

// ─── Load & Init ───

Promise.all([
  d3.json('data/track_a.json'),
  d3.json('data/track_b.json'),
  d3.json('data/synthesis.json'),
  d3.json('data/cv_comparison.json')
]).then(([trackA, trackB, synthesis, cv]) => {
  state.trackA = trackA;
  state.trackB = trackB;
  state.synthesis = {
    matrix: synthesis.fix1_correlation_comparison.correlation_matrix_v2_downsampled,
    dimNames: synthesis.fix1_correlation_comparison.dim_names_9,
    ranking: trackB.dim_ranking.map(d => d.dim)
  };
  state.cv = cv;

  renderPartA();
  renderPartB();
  renderHeatmap();
  initHeatmapStats();
  renderPartD();
  initControls();
  initScrollReveal();
}).catch(err => console.error('Data load error:', err));

// ─── Color scale for heatmap (dark-center diverging) ───

function heatmapColor(value) {
  // Maps [-1, +1] → red → dark → blue
  const t = (value + 1) / 2; // normalize to [0, 1]
  if (t < 0.5) {
    return d3.interpolateRgb('#c94444', '#151518')(t * 2);
  } else {
    return d3.interpolateRgb('#151518', '#4488cc')((t - 0.5) * 2);
  }
}

// ─── Part A: Interpretable Dimensions ───

function renderPartA() {
  const data = state.trackA;
  const dims = Object.entries(data.summary).map(([name, s]) => ({
    name, mean: s.pearson_mean, std: s.pearson_std
  }));

  const container = d3.select('#chart-a');
  const cw = container.node().getBoundingClientRect().width || 460;
  const margin = { top: 12, right: 12, bottom: 44, left: 48 };
  const w = cw - margin.left - margin.right;
  const h = 170 - margin.top - margin.bottom;

  const svg = container.append('svg')
    .attr('width', cw).attr('height', 170)
    .append('g').attr('transform', `translate(${margin.left},${margin.top})`);

  const x = d3.scaleBand().domain(dims.map(d => d.name)).range([0, w]).padding(0.25);
  const y = d3.scaleLinear().domain([0, d3.max(dims, d => d.mean + d.std)]).nice().range([h, 0]);

  // Grid lines
  svg.append('g').call(d3.axisLeft(y).ticks(4).tickSize(-w))
    .call(g => g.select('.domain').remove())
    .call(g => g.selectAll('.tick line').attr('stroke', '#1a1c20'));

  // Bars
  svg.selectAll('rect').data(dims).join('rect')
    .attr('x', d => x(d.name)).attr('y', d => y(d.mean))
    .attr('width', x.bandwidth()).attr('height', d => h - y(d.mean))
    .attr('fill', '#8ab4d4').attr('rx', 2)
    .on('mouseover', function() { d3.select(this).attr('fill', '#a0ccee'); })
    .on('mouseout', function() { d3.select(this).attr('fill', '#8ab4d4'); });

  // X axis
  svg.append('g').attr('transform', `translate(0,${h})`)
    .call(d3.axisBottom(x).tickSize(0))
    .call(g => g.select('.domain').remove())
    .selectAll('text').attr('transform', 'rotate(-35)').style('text-anchor', 'end');

  // Y label
  svg.append('text').attr('transform', 'rotate(-90)')
    .attr('y', -36).attr('x', -h / 2).attr('text-anchor', 'middle')
    .style('font-size', '10px').style('fill', '#556').text('pearson r');

  // Stats
  addStats('#stats-a', [
    { v: data.songs_analyzed, l: 'songs' },
    { v: data.pairs_compared, l: 'pairs' },
    { v: fmt(data.cross_mapper_cosine_mean), l: 'cosine sim' }
  ]);
}

// ─── Part B: Latent Dimensions ───

function renderPartB() {
  const data = state.trackB;
  const dims = data.dim_ranking.map(d => ({ dim: d.dim, mean: d.pearson_mean }));

  const container = d3.select('#chart-b');
  const cw = container.node().getBoundingClientRect().width || 460;
  const margin = { top: 12, right: 12, bottom: 44, left: 48 };
  const w = cw - margin.left - margin.right;
  const h = 170 - margin.top - margin.bottom;

  const svg = container.append('svg')
    .attr('width', cw).attr('height', 170)
    .append('g').attr('transform', `translate(${margin.left},${margin.top})`);

  const x = d3.scaleBand().domain(d3.range(32)).range([0, w]).padding(0.08);
  const y = d3.scaleLinear().domain([0, d3.max(dims, d => d.mean)]).nice().range([h, 0]);

  // Grid
  svg.append('g').call(d3.axisLeft(y).ticks(4).tickSize(-w))
    .call(g => g.select('.domain').remove())
    .call(g => g.selectAll('.tick line').attr('stroke', '#1a1c20'));

  // Threshold line
  svg.append('line')
    .attr('x1', 0).attr('x2', w).attr('y1', y(0.3)).attr('y2', y(0.3))
    .attr('stroke', '#2a4050').attr('stroke-dasharray', '4,3');
  svg.append('text').attr('x', w - 4).attr('y', y(0.3) - 4)
    .attr('text-anchor', 'end').style('font-size', '9px').style('fill', '#2a4050').text('r = 0.3');

  // Bars
  svg.selectAll('rect').data(dims).join('rect')
    .attr('x', (d, i) => x(i)).attr('y', d => y(d.mean))
    .attr('width', x.bandwidth()).attr('height', d => h - y(d.mean))
    .attr('fill', d => d.mean > 0.3 ? '#8ab4d4' : '#1a2530').attr('rx', 1);

  // X axis (sparse ticks)
  svg.append('g').attr('transform', `translate(0,${h})`)
    .call(d3.axisBottom(x).tickValues(d3.range(0, 32, 4)).tickFormat(i => `z${dims[i].dim}`).tickSize(0))
    .call(g => g.select('.domain').remove());

  // Y label
  svg.append('text').attr('transform', 'rotate(-90)')
    .attr('y', -36).attr('x', -h / 2).attr('text-anchor', 'middle')
    .style('font-size', '10px').style('fill', '#556').text('pearson r');

  // Stats
  const nPerceptual = dims.filter(d => d.mean > 0.3).length;
  addStats('#stats-b', [
    { v: data.songs_analyzed, l: 'songs' },
    { v: data.pairs_compared, l: 'pairs' },
    { v: fmt(data.full_cosine_mean), l: 'cosine sim' },
    { v: `${nPerceptual}/32`, l: 'perceptual' }
  ]);
}

// ─── Part C: Heatmap ───

function renderHeatmap() {
  const { matrix, dimNames, ranking } = state.synthesis;
  const { ordering, threshold, showValues } = state.heatmap;

  const container = d3.select('#heatmap-container');
  container.selectAll('*').remove();

  const cellSize = 42;
  const margin = { top: 85, right: 16, bottom: 55, left: 65 };
  const w = dimNames.length * cellSize;
  const h = matrix.length * cellSize;

  const svg = container.append('svg')
    .attr('width', w + margin.left + margin.right)
    .attr('height', h + margin.top + margin.bottom)
    .append('g').attr('transform', `translate(${margin.left},${margin.top})`);

  const rowOrder = ordering === 'ranked' ? ranking : d3.range(32);
  const tooltip = d3.select('#tooltip');

  // Cells
  rowOrder.forEach((latentDim, i) => {
    matrix[latentDim].forEach((value, j) => {
      const filtered = Math.abs(value) < threshold;

      svg.append('rect')
        .attr('class', 'heatmap-cell')
        .attr('x', j * cellSize).attr('y', i * cellSize)
        .attr('width', cellSize).attr('height', cellSize)
        .attr('fill', heatmapColor(value)).attr('rx', 1)
        .classed('filtered', filtered)
        .on('mouseover', function(event) {
          if (filtered) return;
          tooltip.classed('visible', true)
            .style('left', (event.clientX + 12) + 'px')
            .style('top', (event.clientY - 20) + 'px')
            .html(`<strong>z_${latentDim}</strong> &harr; <strong>${dimNames[j]}</strong><br>r = ${value.toFixed(3)}`);
        })
        .on('mouseout', () => tooltip.classed('visible', false))
        .on('click', () => selectCell(latentDim, j, value));

      if (showValues && !filtered) {
        svg.append('text').attr('class', 'cell-value')
          .attr('x', j * cellSize + cellSize / 2)
          .attr('y', i * cellSize + cellSize / 2 + 3)
          .text(value.toFixed(2));
      }
    });
  });

  // X labels
  dimNames.forEach((name, i) => {
    svg.append('text').attr('class', 'heatmap-label')
      .attr('x', i * cellSize + cellSize / 2).attr('y', -8)
      .attr('text-anchor', 'end')
      .attr('transform', `rotate(-45, ${i * cellSize + cellSize / 2}, -8)`)
      .text(name);
  });

  // Y labels
  rowOrder.forEach((dim, i) => {
    svg.append('text').attr('class', 'heatmap-label')
      .attr('x', -6).attr('y', i * cellSize + cellSize / 2 + 4)
      .attr('text-anchor', 'end').text(`z_${dim}`);
  });

  // Legend
  const lw = 150, lh = 10;
  const lx = w - lw, ly = h + 18;

  const defs = svg.append('defs');
  const grad = defs.append('linearGradient').attr('id', 'lg').attr('x1', '0%').attr('x2', '100%');
  d3.range(-1, 1.01, 0.05).forEach(v => {
    grad.append('stop')
      .attr('offset', ((v + 1) / 2 * 100) + '%')
      .attr('stop-color', heatmapColor(v));
  });

  svg.append('rect').attr('x', lx).attr('y', ly)
    .attr('width', lw).attr('height', lh).attr('rx', 2)
    .style('fill', 'url(#lg)');

  svg.append('text').attr('class', 'legend-text')
    .attr('x', lx).attr('y', ly + lh + 12).text('−1');
  svg.append('text').attr('class', 'legend-text')
    .attr('x', lx + lw).attr('y', ly + lh + 12).attr('text-anchor', 'end').text('+1');
  svg.append('text').attr('class', 'legend-text')
    .attr('x', lx + lw / 2).attr('y', ly + lh + 12).attr('text-anchor', 'middle').text('r');
}

function selectCell(latentDim, interpIdx, value) {
  const name = state.synthesis.dimNames[interpIdx];

  d3.select('#cell-info').html(`
    <div class="info-detail">
      <span class="info-detail-label">latent</span>
      <span class="info-detail-value">z_${latentDim}</span>
      <span class="info-detail-label">interp</span>
      <span class="info-detail-value">${name}</span>
      <span class="info-detail-label">r</span>
      <span class="info-detail-value">${value.toFixed(4)}</span>
      <span class="info-detail-label">|r|</span>
      <span class="info-detail-value">${Math.abs(value).toFixed(4)}</span>
      <span class="info-detail-label">sign</span>
      <span class="info-detail-value">${value > 0 ? 'positive' : 'negative'}</span>
    </div>
  `);

  // Highlight
  const cs = 42;
  const rowOrder = state.heatmap.ordering === 'ranked' ? state.synthesis.ranking : d3.range(32);
  const ri = rowOrder.indexOf(latentDim);

  d3.selectAll('.heatmap-cell').classed('selected', false)
    .filter(function() {
      return Math.abs(+d3.select(this).attr('x') - interpIdx * cs) < 1 &&
             Math.abs(+d3.select(this).attr('y') - ri * cs) < 1;
    }).classed('selected', true);
}

function initHeatmapStats() {
  const flat = state.synthesis.matrix.flat();
  const abs = flat.map(Math.abs);

  const stats = [
    { v: d3.mean(abs).toFixed(3), l: 'mean |r|' },
    { v: d3.max(abs).toFixed(3), l: 'max |r|' },
    { v: d3.deviation(abs).toFixed(3), l: 'std |r|' },
    { v: flat.filter(v => v > 0).length, l: 'positive' },
    { v: flat.filter(v => v < 0).length, l: 'negative' },
    { v: flat.filter(v => Math.abs(v) < 0.1).length, l: '|r| < 0.1' }
  ];

  const el = d3.select('#matrix-stats');
  stats.forEach(s => {
    el.append('div').attr('class', 'stat')
      .html(`<span class="stat-val">${s.v}</span><span class="stat-lbl">${s.l}</span>`);
  });
}

// ─── Part D: Cross-Validation ───

function renderPartD() {
  const data = state.cv;
  const origVal = data.original.full_cosine_mean;
  const testVal = data.cv_test.full_cosine_mean;

  // Comparison bars (HTML, no D3 needed)
  document.getElementById('chart-d').innerHTML = `
    <div class="comparison-row">
      <span class="comparison-label">Original (${data.original.songs || 850})</span>
      <div class="comparison-track">
        <div class="comparison-fill" data-w="${origVal * 100}%"></div>
      </div>
      <span class="comparison-value">${origVal.toFixed(3)}</span>
    </div>
    <div class="comparison-row">
      <span class="comparison-label">CV Test (${data.cv_test.songs || 271})</span>
      <div class="comparison-track">
        <div class="comparison-fill" data-w="${testVal * 100}%"></div>
      </div>
      <span class="comparison-value">${testVal.toFixed(3)}</span>
    </div>
  `;

  // Animate bars on scroll
  const fills = document.querySelectorAll('#chart-d .comparison-fill');
  const obs = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        setTimeout(() => fills.forEach(f => f.style.width = f.dataset.w), 150);
        obs.disconnect();
      }
    });
  }, { threshold: 0.3 });
  obs.observe(document.getElementById('chart-d'));

  // Verdict
  const verdictText = data.delta.verdict.split('—')[1]?.trim() || 'generalization confirmed';
  document.getElementById('cv-verdict').innerHTML = `
    <div class="verdict-box">
      <div class="verdict-label">verdict</div>
      <div class="verdict-text">${verdictText}</div>
      <div class="verdict-delta">&Delta; = +${data.delta.cosine_absolute.toFixed(4)} (+${data.delta.cosine_percent.toFixed(1)}%)</div>
    </div>
  `;

  // Stats
  addStats('#stats-d', [
    { v: data.original.pairs, l: 'original pairs' },
    { v: data.cv_test.pairs, l: 'test pairs' },
    { v: `+${data.delta.cosine_percent.toFixed(1)}%`, l: '\u0394 cosine' }
  ]);
}

// ─── Controls ───

function initControls() {
  document.getElementById('ordering').addEventListener('change', e => {
    state.heatmap.ordering = e.target.value;
    renderHeatmap();
  });

  document.getElementById('threshold').addEventListener('input', e => {
    state.heatmap.threshold = parseFloat(e.target.value);
    document.getElementById('threshold-value').textContent = `|r| > ${state.heatmap.threshold.toFixed(2)}`;
    renderHeatmap();
  });

  document.getElementById('show-values').addEventListener('change', e => {
    state.heatmap.showValues = e.target.checked;
    renderHeatmap();
  });
}

// ─── Scroll Reveal ───

function initScrollReveal() {
  const panels = document.querySelectorAll('.panel');
  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        entry.target.classList.add('visible');
        observer.unobserve(entry.target);
      }
    });
  }, { threshold: 0.05 });

  panels.forEach((el, i) => {
    el.style.transitionDelay = `${i * 0.07}s`;
    observer.observe(el);
  });
}

// ─── Helpers ───

function fmt(v) { return v != null ? v.toFixed(3) : '—'; }

function addStats(selector, items) {
  const el = d3.select(selector);
  items.forEach(s => {
    el.append('div').attr('class', 'stat')
      .html(`<span class="stat-val">${s.v}</span><span class="stat-lbl">${s.l}</span>`);
  });
}
