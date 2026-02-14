let data = null;
let currentOrdering = 'ranked';
let threshold = 0;
let showValues = false;
let selectedCell = null;

// Load data
d3.json('data/synthesis.json').then(synthData => {
    d3.json('data/track_b.json').then(trackBData => {
        data = {
            matrix: synthData.fix1_correlation_comparison.correlation_matrix_v2_downsampled,
            dimNames: synthData.fix1_correlation_comparison.dim_names_9,
            ranking: trackBData.dim_ranking.map(d => d.dim)
        };

        initializeStats();
        render();

        // Event listeners
        document.getElementById('ordering').addEventListener('change', e => {
            currentOrdering = e.target.value;
            render();
        });

        document.getElementById('threshold').addEventListener('input', e => {
            threshold = parseFloat(e.target.value);
            document.getElementById('threshold-value').textContent = `|r| > ${threshold.toFixed(2)}`;
            render();
        });

        document.getElementById('show-values').addEventListener('change', e => {
            showValues = e.target.checked;
            render();
        });
    });
});

function initializeStats() {
    const flatValues = data.matrix.flat();
    const absValues = flatValues.map(Math.abs);

    const stats = {
        'Mean |r|': d3.mean(absValues).toFixed(3),
        'Max |r|': d3.max(absValues).toFixed(3),
        'Min |r|': d3.min(absValues).toFixed(3),
        'Std |r|': d3.deviation(absValues).toFixed(3),
        'Positive cells': flatValues.filter(v => v > 0).length,
        'Negative cells': flatValues.filter(v => v < 0).length,
        'Near-zero (|r|<0.1)': flatValues.filter(v => Math.abs(v) < 0.1).length
    };

    const statsContainer = d3.select('#matrix-stats');
    Object.entries(stats).forEach(([label, value]) => {
        statsContainer.append('div')
            .attr('class', 'stat-item')
            .html(`
                <span class="stat-item-label">${label}</span>
                <span class="stat-item-value">${value}</span>
            `);
    });
}

function render() {
    const container = d3.select('#heatmap-container');
    container.selectAll('*').remove();

    const margin = { top: 100, right: 40, bottom: 60, left: 80 };
    const cellSize = 45;
    const width = data.dimNames.length * cellSize;
    const height = data.matrix.length * cellSize;

    const svg = container.append('svg')
        .attr('width', width + margin.left + margin.right)
        .attr('height', height + margin.top + margin.bottom)
        .append('g')
        .attr('transform', `translate(${margin.left},${margin.top})`);

    // Determine row order
    const rowOrder = currentOrdering === 'ranked'
        ? data.ranking
        : d3.range(32);

    // Color scale
    const colorScale = d3.scaleSequential()
        .domain([-1, 1])
        .interpolator(d3.interpolateRdBu);

    // Tooltip
    const tooltip = d3.select('.tooltip');

    // Draw cells
    rowOrder.forEach((latentDim, i) => {
        const row = data.matrix[latentDim];

        row.forEach((value, j) => {
            const absValue = Math.abs(value);
            const isFiltered = absValue < threshold;

            const cell = svg.append('rect')
                .attr('class', 'heatmap-cell')
                .attr('x', j * cellSize)
                .attr('y', i * cellSize)
                .attr('width', cellSize)
                .attr('height', cellSize)
                .attr('fill', colorScale(value))
                .classed('filtered', isFiltered)
                .on('mouseover', function(event) {
                    if (!isFiltered) {
                        tooltip.classed('visible', true)
                            .style('left', (event.pageX + 10) + 'px')
                            .style('top', (event.pageY - 20) + 'px')
                            .html(`
                                <strong>z_${latentDim}</strong> ↔ <strong>${data.dimNames[j]}</strong><br>
                                r = ${value.toFixed(3)}
                            `);
                    }
                })
                .on('mouseout', function() {
                    tooltip.classed('visible', false);
                })
                .on('click', function() {
                    selectCell(latentDim, j, value);
                });

            // Add value text if enabled
            if (showValues && !isFiltered) {
                svg.append('text')
                    .attr('class', 'cell-value')
                    .attr('x', j * cellSize + cellSize / 2)
                    .attr('y', i * cellSize + cellSize / 2 + 3)
                    .text(value.toFixed(2));
            }
        });
    });

    // X-axis labels (interpretable dims)
    data.dimNames.forEach((name, i) => {
        svg.append('text')
            .attr('class', 'heatmap-label')
            .attr('x', i * cellSize + cellSize / 2)
            .attr('y', -10)
            .attr('text-anchor', 'end')
            .attr('transform', `rotate(-45, ${i * cellSize + cellSize / 2}, -10)`)
            .text(name);
    });

    // Y-axis labels (latent dims)
    rowOrder.forEach((latentDim, i) => {
        svg.append('text')
            .attr('class', 'heatmap-label')
            .attr('x', -10)
            .attr('y', i * cellSize + cellSize / 2 + 4)
            .attr('text-anchor', 'end')
            .text(`z_${latentDim}`);
    });

    // Add color legend
    const legendWidth = 200;
    const legendHeight = 15;

    const legendScale = d3.scaleLinear()
        .domain([-1, 1])
        .range([0, legendWidth]);

    const legendGradient = svg.append('defs')
        .append('linearGradient')
        .attr('id', 'legend-gradient')
        .attr('x1', '0%')
        .attr('x2', '100%');

    legendGradient.selectAll('stop')
        .data(d3.range(-1, 1.01, 0.1))
        .join('stop')
        .attr('offset', d => ((d + 1) / 2 * 100) + '%')
        .attr('stop-color', d => colorScale(d));

    svg.append('rect')
        .attr('x', width - legendWidth - 10)
        .attr('y', height + 20)
        .attr('width', legendWidth)
        .attr('height', legendHeight)
        .style('fill', 'url(#legend-gradient)');

    svg.append('text')
        .attr('class', 'legend-text')
        .attr('x', width - legendWidth - 10)
        .attr('y', height + 50)
        .text('-1.0');

    svg.append('text')
        .attr('class', 'legend-text')
        .attr('x', width - 10)
        .attr('y', height + 50)
        .attr('text-anchor', 'end')
        .text('+1.0');

    svg.append('text')
        .attr('class', 'legend-text')
        .attr('x', width - legendWidth/2 - 10)
        .attr('y', height + 50)
        .attr('text-anchor', 'middle')
        .text('Pearson r');
}

function selectCell(latentDim, interpretableIdx, value) {
    selectedCell = { latentDim, interpretableIdx, value };

    const interpretableName = data.dimNames[interpretableIdx];

    // Update info panel
    const infoPanel = d3.select('#cell-info');
    infoPanel.html(`
        <div class="detail">
            <span class="detail-label">Latent dimension:</span>
            <span class="detail-value">z_${latentDim}</span>

            <span class="detail-label">Interpretable dimension:</span>
            <span class="detail-value">${interpretableName}</span>

            <span class="detail-label">Pearson correlation:</span>
            <span class="detail-value">${value.toFixed(4)}</span>

            <span class="detail-label">Absolute value:</span>
            <span class="detail-value">${Math.abs(value).toFixed(4)}</span>

            <span class="detail-label">Direction:</span>
            <span class="detail-value">${value > 0 ? 'Positive correlation' : 'Negative correlation'}</span>
        </div>
    `);

    // Highlight selected cell
    d3.selectAll('.heatmap-cell').classed('selected', false);
    d3.selectAll('.heatmap-cell').filter(function() {
        const x = +d3.select(this).attr('x');
        const y = +d3.select(this).attr('y');
        const cellSize = 45;
        return Math.abs(x - interpretableIdx * cellSize) < 1 &&
               Math.abs(y - data.ranking.indexOf(latentDim) * cellSize) < 1 &&
               currentOrdering === 'ranked' ||
               Math.abs(x - interpretableIdx * cellSize) < 1 &&
               Math.abs(y - latentDim * cellSize) < 1 &&
               currentOrdering === 'sequential';
    }).classed('selected', true);
}
