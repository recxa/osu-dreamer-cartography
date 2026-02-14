// Load all data and populate landing page
Promise.all([
    d3.json('data/track_a.json'),
    d3.json('data/track_b.json'),
    d3.json('data/synthesis.json'),
    d3.json('data/cv_comparison.json')
]).then(([trackA, trackB, synthesis, cvComparison]) => {
    populateTrackA(trackA);
    populateTrackB(trackB);
    populateSynthesis(synthesis);
    populateCV(cvComparison);
}).catch(err => {
    console.error('Error loading data:', err);
});

function populateTrackA(data) {
    // Stats
    document.getElementById('track-a-songs').textContent = data.songs_analyzed || '—';
    document.getElementById('track-a-pairs').textContent = data.pairs_compared || '—';
    document.getElementById('track-a-cosine').textContent =
        data.cross_mapper_cosine_mean != null ? data.cross_mapper_cosine_mean.toFixed(3) : '—';

    // Mini bar chart — Track A uses "summary" not "dim_summary"
    const dimSummary = data.summary;
    if (!dimSummary) return;

    const dims = Object.keys(dimSummary).map(key => ({
        name: key,
        value: dimSummary[key].pearson_mean
    }));

    renderMiniBarChart('#track-a-preview', dims, 'Pearson r');
}

function populateTrackB(data) {
    // Stats
    document.getElementById('track-b-songs').textContent = data.songs_analyzed || '—';
    document.getElementById('track-b-pairs').textContent = data.pairs_compared || '—';
    document.getElementById('track-b-cosine').textContent =
        data.full_cosine_mean != null ? data.full_cosine_mean.toFixed(3) : '—';

    // Top 10 latent dims bar chart
    if (!data.dim_ranking) return;

    const topDims = data.dim_ranking.slice(0, 10).map(d => ({
        name: `z_${d.dim}`,
        value: d.pearson_mean
    }));

    renderMiniBarChart('#track-b-preview', topDims, 'Pearson r');
}

function populateSynthesis(data) {
    // Synthesis data is nested under fix1_correlation_comparison
    const fix1 = data.fix1_correlation_comparison;
    const corrMatrix = fix1?.correlation_matrix_v2_downsampled;

    document.getElementById('synthesis-beatmaps').textContent =
        data.n_beatmaps_used || fix1?.n_latent ? '722' : '—';

    if (corrMatrix) {
        const flatValues = corrMatrix.flat().map(Math.abs);
        const meanAbsR = d3.mean(flatValues);
        document.getElementById('synthesis-mean-r').textContent =
            meanAbsR != null ? meanAbsR.toFixed(3) : '—';
    }

    // Mini heatmap preview
    renderMiniHeatmap('#synthesis-preview', corrMatrix);
}

function populateCV(data) {
    // Stats
    const delta = data.delta?.cosine_absolute;
    const deltaPct = data.delta?.cosine_percent;

    document.getElementById('cv-delta').textContent =
        delta !== undefined ? `${delta >= 0 ? '+' : ''}${delta.toFixed(4)} (${deltaPct >= 0 ? '+' : ''}${deltaPct.toFixed(1)}%)` : '—';

    document.getElementById('cv-verdict').textContent =
        data.delta?.verdict?.split('—')[1]?.trim() || '—';

    // Comparison chart
    if (data.original && data.cv_test) {
        renderCVComparison('#cv-preview', data);
    }
}

function renderMiniBarChart(selector, data, yLabel) {
    const container = d3.select(selector);
    container.selectAll('*').remove();

    const containerWidth = container.node().offsetWidth || 600;
    const margin = { top: 20, right: 20, bottom: 50, left: 60 };
    const width = containerWidth - margin.left - margin.right;
    const height = 180 - margin.top - margin.bottom;

    const svg = container.append('svg')
        .attr('width', width + margin.left + margin.right)
        .attr('height', height + margin.top + margin.bottom)
        .append('g')
        .attr('transform', `translate(${margin.left},${margin.top})`);

    const x = d3.scaleBand()
        .domain(data.map(d => d.name))
        .range([0, width])
        .padding(0.2);

    const y = d3.scaleLinear()
        .domain([0, d3.max(data, d => d.value)])
        .nice()
        .range([height, 0]);

    // Bars
    svg.selectAll('.bar')
        .data(data)
        .join('rect')
        .attr('class', 'bar')
        .attr('x', d => x(d.name))
        .attr('y', d => y(d.value))
        .attr('width', x.bandwidth())
        .attr('height', d => height - y(d.value))
        .attr('rx', 2);

    // X axis
    svg.append('g')
        .attr('class', 'axis')
        .attr('transform', `translate(0,${height})`)
        .call(d3.axisBottom(x))
        .selectAll('text')
        .attr('transform', 'rotate(-45)')
        .style('text-anchor', 'end');

    // Y axis
    svg.append('g')
        .attr('class', 'axis')
        .call(d3.axisLeft(y).ticks(5));

    // Y label
    svg.append('text')
        .attr('transform', 'rotate(-90)')
        .attr('y', 0 - margin.left)
        .attr('x', 0 - (height / 2))
        .attr('dy', '1em')
        .style('text-anchor', 'middle')
        .style('font-size', '12px')
        .style('fill', '#64748b')
        .text(yLabel);
}

function renderMiniHeatmap(selector, matrix) {
    if (!matrix) return;

    const container = d3.select(selector);
    container.selectAll('*').remove();

    const containerWidth = container.node().offsetWidth || 300;
    const cellW = Math.min(Math.floor(containerWidth / matrix[0].length), 20);
    const cellH = Math.min(Math.floor(180 / matrix.length), 6);
    const width = matrix[0].length * cellW;
    const height = matrix.length * cellH;

    const wrapper = container.append('div')
        .style('display', 'flex')
        .style('flex-direction', 'column')
        .style('align-items', 'center')
        .style('padding', '1rem');

    const svg = wrapper.append('svg')
        .attr('width', width)
        .attr('height', height);

    const colorScale = d3.scaleDiverging()
        .domain([-1, 0, 1])
        .interpolator(d3.interpolateRdBu);

    matrix.forEach((row, i) => {
        row.forEach((value, j) => {
            svg.append('rect')
                .attr('x', j * cellW)
                .attr('y', i * cellH)
                .attr('width', cellW)
                .attr('height', cellH)
                .attr('fill', colorScale(value));
        });
    });

    wrapper.append('div')
        .style('text-align', 'center')
        .style('margin-top', '8px')
        .style('font-size', '12px')
        .style('color', '#64748b')
        .text('32×9 correlation matrix preview — click "Explore heatmap" for interactive version');
}

function renderCVComparison(selector, data) {
    const container = d3.select(selector);
    container.selectAll('*').remove();

    const containerWidth = container.node().offsetWidth || 600;
    const margin = { top: 20, right: 60, bottom: 40, left: 180 };
    const width = containerWidth - margin.left - margin.right;
    const height = 140 - margin.top - margin.bottom;

    const svg = container.append('svg')
        .attr('width', width + margin.left + margin.right)
        .attr('height', height + margin.top + margin.bottom)
        .append('g')
        .attr('transform', `translate(${margin.left},${margin.top})`);

    const chartData = [
        { label: 'Original (all 850)', value: data.original.full_cosine_mean },
        { label: 'CV Test (held-out 271)', value: data.cv_test.full_cosine_mean }
    ];

    const y = d3.scaleBand()
        .domain(chartData.map(d => d.label))
        .range([0, height])
        .padding(0.3);

    const x = d3.scaleLinear()
        .domain([0, 1])
        .range([0, width]);

    svg.selectAll('.bar')
        .data(chartData)
        .join('rect')
        .attr('class', 'bar')
        .attr('y', d => y(d.label))
        .attr('x', 0)
        .attr('width', d => x(d.value))
        .attr('height', y.bandwidth())
        .attr('rx', 3);

    svg.append('g')
        .attr('class', 'axis')
        .call(d3.axisLeft(y))
        .selectAll('text')
        .style('font-size', '12px');

    svg.append('g')
        .attr('class', 'axis')
        .attr('transform', `translate(0,${height})`)
        .call(d3.axisBottom(x).ticks(5));

    // Value labels
    svg.selectAll('.value-label')
        .data(chartData)
        .join('text')
        .attr('x', d => x(d.value) + 8)
        .attr('y', d => y(d.label) + y.bandwidth() / 2)
        .attr('dy', '0.35em')
        .style('font-size', '14px')
        .style('font-weight', '700')
        .style('fill', '#1e293b')
        .text(d => d.value.toFixed(3));
}
