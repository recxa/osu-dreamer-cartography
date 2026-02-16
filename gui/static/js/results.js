// ═══ Dynamic Results Viewer ═══
// All descriptions are generated from the data — nothing is hardcoded.

const Results = {
  data: null,
  heatmapState: { ordering: 'ranked', threshold: 0, showValues: false },

  async load() {
    const loading = document.getElementById('results-loading');
    const content = document.getElementById('results-content');
    loading.classList.remove('hidden');
    content.classList.add('hidden');

    try {
      const res = await fetch('/api/results');
      Results.data = await res.json();
      Results.render();
    } catch (e) {
      loading.innerHTML = `<span style="color:var(--danger)">failed to load results: ${e.message}</span>`;
      return;
    }

    loading.classList.add('hidden');
    content.classList.remove('hidden');
  },

  render() {
    const d = Results.data;

    if (d.has_track_a) Results.renderTrackA(d.track_a);
    else document.getElementById('panel-a').classList.add('hidden');

    if (d.has_track_b) Results.renderTrackB(d.track_b);
    else document.getElementById('panel-b').classList.add('hidden');

    if (d.has_synthesis) Results.renderSynthesis(d);
    else document.getElementById('panel-c').classList.add('hidden');

    // Show notice if Track B is missing
    if (!d.has_track_b) {
      document.getElementById('no-trackb-notice').classList.remove('hidden');
    }

    Results.initActions();
  },

  // ─── Track A ───
  renderTrackA(trackA) {
    // Generate dynamic description
    const summary = trackA.summary;
    const ranked = Object.entries(summary)
      .sort((a, b) => b[1].pearson_mean - a[1].pearson_mean);

    const topDim = ranked[0];
    const bottomDim = ranked[ranked.length - 1];
    const highAgree = ranked.filter(([_, s]) => s.pearson_mean > 0.3);
    const lowAgree = ranked.filter(([_, s]) => s.pearson_mean < 0.1);

    let desc = `Cross-mapper agreement on 9-dim encoding across ${trackA.songs_analyzed} songs.`;
    if (topDim) {
      desc += ` <strong>${topDim[0]}</strong> (r=${topDim[1].pearson_mean.toFixed(2)}) shows the highest agreement`;
    }
    if (bottomDim && bottomDim[0] !== topDim[0]) {
      desc += `; <strong>${bottomDim[0]}</strong> (r=${bottomDim[1].pearson_mean.toFixed(2)}) the lowest`;
    }
    desc += '.';

    if (highAgree.length > 0 && lowAgree.length > 0) {
      desc += ` ${highAgree.length} dimension${highAgree.length > 1 ? 's' : ''} above r=0.3 (perceptual), ${lowAgree.length} below r=0.1 (stylistic).`;
    }

    document.getElementById('desc-a').innerHTML = desc;

    Charts.renderTrackA('#chart-a', trackA);
    Charts.addStats('#stats-a', [
      { v: trackA.songs_analyzed, l: 'songs' },
      { v: trackA.pairs_compared, l: 'pairs' },
      { v: Charts.fmt(trackA.cross_mapper_cosine_mean), l: 'cosine sim' }
    ]);
  },

  // ─── Track B ───
  renderTrackB(trackB) {
    const dims = trackB.dim_ranking;
    const nPerceptual = dims.filter(d => d.pearson_mean > 0.3).length;
    const nStylistic = dims.filter(d => d.pearson_mean < 0.05).length;
    const nMixed = dims.length - nPerceptual - nStylistic;
    const topDim = dims[0];

    let desc = `${dims.length}-dim VAE latent space agreement across ${trackB.songs_analyzed} songs.`;
    if (topDim) {
      desc += ` <strong>${nPerceptual}/${dims.length} dims</strong> classified as perceptual (r>0.3).`;
      desc += ` Top dimension z<sub>${topDim.dim}</sub> achieves r=${topDim.pearson_mean.toFixed(3)}.`;
    }

    if (nPerceptual === 0) {
      desc += ' <strong>No dimensions</strong> exceed the perceptual threshold — the VAE may need retraining or the dataset may be too small.';
    }

    document.getElementById('desc-b').innerHTML = desc;

    Charts.renderTrackB('#chart-b', trackB);
    Charts.addStats('#stats-b', [
      { v: trackB.songs_analyzed, l: 'songs' },
      { v: trackB.pairs_compared, l: 'pairs' },
      { v: Charts.fmt(trackB.full_cosine_mean), l: 'cosine sim' },
      { v: `${nPerceptual}/${dims.length}`, l: 'perceptual' }
    ]);
  },

  // ─── Synthesis Heatmap ───
  renderSynthesis(allData) {
    const synth = allData.synthesis;
    const trackB = allData.track_b;

    // Build heatmap data structure
    const matrix = synth.correlation_matrix;
    const dimNames = synth.dim_names_9;
    const ranking = trackB.dim_ranking.map(d => d.dim);

    Results._heatmapData = { matrix, dimNames, ranking };

    // Generate description
    const flat = matrix.flat();
    const absVals = flat.map(Math.abs);
    const meanAbs = d3.mean(absVals);
    const maxAbs = d3.max(absVals);

    // Find strongest correlation
    let maxR = 0, maxZi = 0, maxDi = 0;
    matrix.forEach((row, zi) => {
      row.forEach((val, di) => {
        if (Math.abs(val) > Math.abs(maxR)) {
          maxR = val;
          maxZi = zi;
          maxDi = di;
        }
      });
    });

    // Categorize what perceptual dims encode
    const perceptualDims = trackB.dim_ranking.filter(d => d.pearson_mean > 0.3);
    const topEncodings = {};
    perceptualDims.forEach(pd => {
      const row = matrix[pd.dim];
      if (!row) return;
      const maxIdx = row.reduce((best, val, i) => Math.abs(val) > Math.abs(row[best]) ? i : best, 0);
      const dimName = dimNames[maxIdx];
      topEncodings[dimName] = (topEncodings[dimName] || 0) + 1;
    });
    const topEncoded = Object.entries(topEncodings).sort((a, b) => b[1] - a[1]);

    let desc = `Correlation between ${matrix.length} latent dims and ${dimNames.length} interpretable dims.`;
    if (maxAbs > 0.01) {
      desc += ` Strongest link: <strong>z_${maxZi}</strong> &harr; <strong>${dimNames[maxDi]}</strong> (r=${maxR.toFixed(3)}).`;
    }
    if (topEncoded.length > 0) {
      const topNames = topEncoded.slice(0, 2).map(([n]) => `<strong>${n}</strong>`).join(' and ');
      desc += ` Perceptual latent dims primarily encode ${topNames}.`;
    }
    if (meanAbs < 0.05) {
      desc += ' Overall correlations are weak — the VAE may encode higher-order features not captured by the 9-dim representation.';
    }

    document.getElementById('desc-c').innerHTML = desc;

    // Render heatmap
    Charts.renderHeatmap('#heatmap-container', Results._heatmapData, Results.heatmapState);
    Results.renderMatrixStats(matrix);
    Results.initHeatmapControls();
  },

  renderMatrixStats(matrix) {
    const flat = matrix.flat();
    const abs = flat.map(Math.abs);

    Charts.addStats('#matrix-stats', [
      { v: d3.mean(abs).toFixed(3), l: 'mean |r|' },
      { v: d3.max(abs).toFixed(3), l: 'max |r|' },
      { v: d3.deviation(abs).toFixed(3), l: 'std |r|' },
      { v: flat.filter(v => v > 0).length, l: 'positive' },
      { v: flat.filter(v => v < 0).length, l: 'negative' },
      { v: flat.filter(v => Math.abs(v) < 0.1).length, l: '|r| < 0.1' }
    ]);
  },

  initHeatmapControls() {
    document.getElementById('hm-ordering').addEventListener('change', e => {
      Results.heatmapState.ordering = e.target.value;
      Charts.renderHeatmap('#heatmap-container', Results._heatmapData, Results.heatmapState);
    });

    document.getElementById('hm-threshold').addEventListener('input', e => {
      Results.heatmapState.threshold = parseFloat(e.target.value);
      document.getElementById('hm-threshold-value').textContent = `|r| > ${Results.heatmapState.threshold.toFixed(2)}`;
      Charts.renderHeatmap('#heatmap-container', Results._heatmapData, Results.heatmapState);
    });

    document.getElementById('hm-show-values').addEventListener('change', e => {
      Results.heatmapState.showValues = e.target.checked;
      Charts.renderHeatmap('#heatmap-container', Results._heatmapData, Results.heatmapState);
    });
  },

  // ─── Actions ───
  initActions() {
    document.getElementById('btn-export').addEventListener('click', () => {
      const blob = new Blob([JSON.stringify(Results.data, null, 2)], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = 'cartography_results.json';
      a.click();
      URL.revokeObjectURL(url);
    });

    document.getElementById('btn-new-run').addEventListener('click', () => {
      App.showView('setup');
    });
  }
};
