// ═══ D3 Chart Renderers ═══

const Charts = {

  // ─── Color scale for heatmap (dark-center diverging) ───
  heatmapColor(value) {
    const t = (value + 1) / 2;
    if (t < 0.5) {
      return d3.interpolateRgb('#c94444', '#151518')(t * 2);
    } else {
      return d3.interpolateRgb('#151518', '#4488cc')((t - 0.5) * 2);
    }
  },

  // ─── Track A: Interpretable Dimensions Bar Chart ───
  renderTrackA(selector, data) {
    const dims = Object.entries(data.summary).map(([name, s]) => ({
      name, mean: s.pearson_mean, std: s.pearson_std
    })).sort((a, b) => b.mean - a.mean);

    const container = d3.select(selector);
    container.selectAll('*').remove();

    const cw = container.node().getBoundingClientRect().width || 460;
    const margin = { top: 12, right: 12, bottom: 44, left: 48 };
    const w = cw - margin.left - margin.right;
    const h = 170 - margin.top - margin.bottom;

    const svg = container.append('svg')
      .attr('width', cw).attr('height', 170)
      .append('g').attr('transform', `translate(${margin.left},${margin.top})`);

    const x = d3.scaleBand().domain(dims.map(d => d.name)).range([0, w]).padding(0.25);
    const y = d3.scaleLinear().domain([0, d3.max(dims, d => d.mean + d.std)]).nice().range([h, 0]);

    // Grid
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
  },

  // ─── Track B: Latent Dimensions Bar Chart ───
  renderTrackB(selector, data) {
    const dims = data.dim_ranking.map(d => ({ dim: d.dim, mean: d.pearson_mean }));

    const container = d3.select(selector);
    container.selectAll('*').remove();

    const cw = container.node().getBoundingClientRect().width || 460;
    const margin = { top: 12, right: 12, bottom: 44, left: 48 };
    const w = cw - margin.left - margin.right;
    const h = 170 - margin.top - margin.bottom;

    const svg = container.append('svg')
      .attr('width', cw).attr('height', 170)
      .append('g').attr('transform', `translate(${margin.left},${margin.top})`);

    const x = d3.scaleBand().domain(d3.range(dims.length)).range([0, w]).padding(0.08);
    const yMax = d3.max(dims, d => d.mean) || 0.5;
    const y = d3.scaleLinear().domain([0, yMax]).nice().range([h, 0]);

    // Grid
    svg.append('g').call(d3.axisLeft(y).ticks(4).tickSize(-w))
      .call(g => g.select('.domain').remove())
      .call(g => g.selectAll('.tick line').attr('stroke', '#1a1c20'));

    // Threshold line
    if (yMax > 0.3) {
      svg.append('line')
        .attr('x1', 0).attr('x2', w).attr('y1', y(0.3)).attr('y2', y(0.3))
        .attr('stroke', '#2a4050').attr('stroke-dasharray', '4,3');
      svg.append('text').attr('x', w - 4).attr('y', y(0.3) - 4)
        .attr('text-anchor', 'end').style('font-size', '9px').style('fill', '#2a4050').text('r = 0.3');
    }

    // Bars
    svg.selectAll('rect').data(dims).join('rect')
      .attr('x', (d, i) => x(i)).attr('y', d => y(d.mean))
      .attr('width', x.bandwidth()).attr('height', d => h - y(d.mean))
      .attr('fill', d => d.mean > 0.3 ? '#8ab4d4' : '#1a2530').attr('rx', 1);

    // X axis (sparse ticks)
    const tickStep = Math.max(1, Math.floor(dims.length / 8));
    svg.append('g').attr('transform', `translate(0,${h})`)
      .call(d3.axisBottom(x)
        .tickValues(d3.range(0, dims.length, tickStep))
        .tickFormat(i => `z${dims[i].dim}`)
        .tickSize(0))
      .call(g => g.select('.domain').remove());

    // Y label
    svg.append('text').attr('transform', 'rotate(-90)')
      .attr('y', -36).attr('x', -h / 2).attr('text-anchor', 'middle')
      .style('font-size', '10px').style('fill', '#556').text('pearson r');
  },

  // ─── Synthesis Heatmap ───
  renderHeatmap(containerSelector, data, options = {}) {
    const { matrix, dimNames, ranking } = data;
    const { ordering = 'ranked', threshold = 0, showValues = false } = options;

    const container = d3.select(containerSelector);
    container.selectAll('*').remove();

    const nLatent = matrix.length;
    const cellSize = 42;
    const margin = { top: 85, right: 16, bottom: 55, left: 65 };
    const w = dimNames.length * cellSize;
    const h = nLatent * cellSize;

    const svg = container.append('svg')
      .attr('width', w + margin.left + margin.right)
      .attr('height', h + margin.top + margin.bottom)
      .append('g').attr('transform', `translate(${margin.left},${margin.top})`);

    const rowOrder = ordering === 'ranked' ? ranking : d3.range(nLatent);
    const tooltip = d3.select('#tooltip');

    // Cells
    rowOrder.forEach((latentDim, i) => {
      (matrix[latentDim] || []).forEach((value, j) => {
        const filtered = Math.abs(value) < threshold;

        svg.append('rect')
          .attr('class', 'heatmap-cell')
          .attr('x', j * cellSize).attr('y', i * cellSize)
          .attr('width', cellSize).attr('height', cellSize)
          .attr('fill', Charts.heatmapColor(value)).attr('rx', 1)
          .classed('filtered', filtered)
          .on('mouseover', function(event) {
            if (filtered) return;
            tooltip.classed('visible', true)
              .style('left', (event.clientX + 12) + 'px')
              .style('top', (event.clientY - 20) + 'px')
              .html(`<strong>z_${latentDim}</strong> &harr; <strong>${dimNames[j]}</strong><br>r = ${value.toFixed(3)}`);
          })
          .on('mouseout', () => tooltip.classed('visible', false))
          .on('click', () => Charts.selectHeatmapCell(latentDim, j, value, dimNames, rowOrder));

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
    const grad = defs.append('linearGradient').attr('id', 'hm-lg').attr('x1', '0%').attr('x2', '100%');
    d3.range(-1, 1.01, 0.05).forEach(v => {
      grad.append('stop')
        .attr('offset', ((v + 1) / 2 * 100) + '%')
        .attr('stop-color', Charts.heatmapColor(v));
    });

    svg.append('rect').attr('x', lx).attr('y', ly)
      .attr('width', lw).attr('height', lh).attr('rx', 2)
      .style('fill', 'url(#hm-lg)');

    svg.append('text').attr('class', 'legend-text')
      .attr('x', lx).attr('y', ly + lh + 12).text('−1');
    svg.append('text').attr('class', 'legend-text')
      .attr('x', lx + lw).attr('y', ly + lh + 12).attr('text-anchor', 'end').text('+1');
    svg.append('text').attr('class', 'legend-text')
      .attr('x', lx + lw / 2).attr('y', ly + lh + 12).attr('text-anchor', 'middle').text('r');
  },

  selectHeatmapCell(latentDim, interpIdx, value, dimNames, rowOrder) {
    const name = dimNames[interpIdx];

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

    const cs = 42;
    const ri = rowOrder.indexOf(latentDim);
    d3.selectAll('.heatmap-cell').classed('selected', false)
      .filter(function() {
        return Math.abs(+d3.select(this).attr('x') - interpIdx * cs) < 1 &&
               Math.abs(+d3.select(this).attr('y') - ri * cs) < 1;
      }).classed('selected', true);
  },

  // ─── Helper: add stat blocks ───
  addStats(selector, items) {
    const el = d3.select(selector);
    el.selectAll('*').remove();
    items.forEach(s => {
      el.append('div').attr('class', 'stat')
        .html(`<span class="stat-val">${s.v}</span><span class="stat-lbl">${s.l}</span>`);
    });
  },

  fmt(v) { return v != null ? v.toFixed(3) : '—'; }
};
