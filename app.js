/* ============================================================
   Predicción de Matrícula — frontend
   Carga resultados.json (precomputado en Python) y renderiza
   gráficos + tablas interactivas con Plotly.
   ============================================================ */

const SERIE_LABEL = {
  NVO_INGRESO: 'Nuevo Ingreso',
  REINGRESO:   'Reingreso',
  MATRICULA:   'Matrícula total',
};

const COLORS = {
  hist:                 '#1f3a5f',
  ganador:              '#b54a2f',
  'Regresión Lineal':   '#2a6873',
  'Exponencial':        '#6b4485',
  'Holt':               '#3d6b3a',
  'Holt amortiguado':   '#b58a2a',
  'Media Móvil (3)':    '#8b7355',
  ARIMA:                '#a64d4d',
};

let DATA = null;     // resultados.json
let estado = {
  serie: 'NVO_INGRESO',
  carrera: null,
  modelo: '__mejor__',
};

// ---------- Utilidades ----------
function $(id) { return document.getElementById(id); }

function fmtInt(x) {
  if (x === null || x === undefined || !isFinite(x)) return '—';
  return Math.round(x).toLocaleString('es-MX');
}

function colorParaModelo(nombre) {
  if (nombre in COLORS) return COLORS[nombre];
  if (nombre.startsWith('ARIMA')) return COLORS.ARIMA;
  return '#999';
}

function nombreCarreraCorta(s) {
  // Acorta "Licenciatura en/como X" para los dropdowns
  return s
    .replace(/^Licenciatura como /i, '')
    .replace(/^Licenciatura en /i, '')
    .replace(/^Licenciatura /i, '');
}

// ---------- Cargar datos ----------
async function cargarDatos() {
  const r = await fetch('resultados.json');
  if (!r.ok) throw new Error('No se pudo cargar resultados.json');
  DATA = await r.json();
  $('meta-version').textContent =
    `Datos: ${Object.keys(DATA.columnas).length} indicadores · ` +
    `Horizonte: ${DATA.horizonte} años`;
}

// ---------- Inicialización de selectores ----------
function poblarCarreras() {
  const sel = $('sel-carrera');
  const programas = DATA.columnas[estado.serie].programas
    .filter(p => p in DATA.columnas[estado.serie].datos);
  programas.sort((a, b) => nombreCarreraCorta(a).localeCompare(nombreCarreraCorta(b), 'es'));
  sel.innerHTML = '';
  for (const p of programas) {
    const opt = document.createElement('option');
    opt.value = p;
    opt.textContent = nombreCarreraCorta(p);
    sel.appendChild(opt);
  }
  // Mantener selección si existe
  if (programas.includes(estado.carrera)) {
    sel.value = estado.carrera;
  } else {
    estado.carrera = programas[0];
    sel.value = estado.carrera;
  }
}

function poblarModelos() {
  const sel = $('sel-modelo');
  // limpiar todo excepto "Mejor automático"
  sel.innerHTML = '<option value="__mejor__">Mejor automático ★</option>';
  const info = DATA.columnas[estado.serie].datos[estado.carrera];
  if (!info) return;
  const modelos = Object.keys(info.modelos)
    .sort((a, b) => info.modelos[a].rmse - info.modelos[b].rmse);
  for (const m of modelos) {
    const opt = document.createElement('option');
    opt.value = m;
    opt.textContent = m;
    sel.appendChild(opt);
  }
  // si el modelo guardado no existe para esta carrera, volver a "mejor"
  if (estado.modelo !== '__mejor__' && !modelos.includes(estado.modelo)) {
    estado.modelo = '__mejor__';
  }
  sel.value = estado.modelo;
}

// ---------- Render ----------
function modeloElegido(info) {
  return estado.modelo === '__mejor__' ? info.mejor : estado.modelo;
}

function renderResumen(info) {
  const elegido = modeloElegido(info);
  const m = info.modelos[elegido];
  $('summary-mejor').textContent = elegido;
  $('summary-criterio').textContent =
    estado.modelo === '__mejor__'
      ? `Elegido por ${info.criterio}`
      : `Modelo seleccionado manualmente`;

  const nAnios = info.anios_hist.length;
  $('summary-anios').textContent = nAnios;
  const a0 = info.anios_hist[0], a1 = info.anios_hist[info.anios_hist.length - 1];
  const tarjetaAnios = $('summary-anios').parentElement;
  if (nAnios < 6) {
    $('summary-rango').textContent = `${a0} – ${a1} · serie corta ⚠`;
    tarjetaAnios.classList.add('card-warn');
  } else {
    $('summary-rango').textContent = `${a0} – ${a1}`;
    tarjetaAnios.classList.remove('card-warn');
  }

  const prox = m.prediccion[0];
  const ultimo = info.valores_hist[info.valores_hist.length - 1];
  const ultimoFut = m.prediccion[m.prediccion.length - 1];
  $('summary-prox').textContent = fmtInt(prox);
  $('summary-prox-label').textContent = `Año ${info.anios_fut[0]}`;

  const delta = ultimoFut - ultimo;
  const pct = ultimo ? (delta / ultimo) * 100 : 0;
  const signo = delta > 0 ? '+' : '';
  $('summary-delta').textContent = `${signo}${fmtInt(delta)}`;
  $('summary-delta-label').textContent =
    `${signo}${pct.toFixed(1)}% vs. ${ultimo} (${a1})`;
}

function renderGrafico(info) {
  const elegido = modeloElegido(info);

  if (typeof Plotly === 'undefined') {
    const g = document.getElementById('grafico');
    g.innerHTML = `
      <div style="display:flex;align-items:center;justify-content:center;
                  height:100%;padding:2rem;text-align:center;color:#8b8275;
                  font-family:'IBM Plex Mono',monospace;font-size:0.85rem;
                  border:1px dashed #d3c6ad;border-radius:4px;">
        El gráfico requiere conexión a internet<br>
        (Plotly.js se carga desde un CDN).<br><br>
        Los datos y tablas funcionan sin conexión.
      </div>`;
    return;
  }

  const traces = [];

  // 1. Histórico
  traces.push({
    x: info.anios_hist,
    y: info.valores_hist,
    type: 'scatter',
    mode: 'lines+markers',
    name: 'Histórico',
    line: { color: COLORS.hist, width: 2.5 },
    marker: { size: 7, color: COLORS.hist },
    hovertemplate: '%{x}: <b>%{y:.0f}</b><extra></extra>',
    legendrank: 1,
  });

  // 2. Cada modelo (línea desde último histórico hasta el pronóstico)
  const xUlt = info.anios_hist[info.anios_hist.length - 1];
  const yUlt = info.valores_hist[info.valores_hist.length - 1];

  const orden = Object.keys(info.modelos)
    .sort((a, b) => info.modelos[a].rmse - info.modelos[b].rmse);

  for (const nombre of orden) {
    const m = info.modelos[nombre];
    const esElegido = nombre === elegido;
    const color = esElegido ? COLORS.ganador : colorParaModelo(nombre);

    traces.push({
      x: [xUlt, ...info.anios_fut],
      y: [yUlt, ...m.prediccion],
      type: 'scatter',
      mode: 'lines+markers',
      name: `${nombre} <span style="font-family:'IBM Plex Mono';font-size:10px;color:#888">RMSE=${m.rmse.toFixed(1)}</span>`,
      line: {
        color,
        width: esElegido ? 2.8 : 1.4,
        dash: 'dot',
      },
      marker: {
        size: esElegido ? 8 : 5,
        symbol: esElegido ? 'star' : 'circle',
        color,
        line: esElegido ? { color: '#1a1612', width: 1 } : undefined,
      },
      opacity: esElegido ? 1 : (estado.modelo === '__mejor__' ? 0.4 : 0.18),
      hovertemplate: `<b>${nombre}</b><br>%{x}: %{y:.0f}<extra></extra>`,
      legendrank: esElegido ? 2 : 3 + orden.indexOf(nombre),
    });
  }

  const layout = {
    margin: { l: 60, r: 30, t: 20, b: 50 },
    paper_bgcolor: '#faf5e8',
    plot_bgcolor: '#faf5e8',
    font: {
      family: "'IBM Plex Sans', sans-serif",
      size: 12,
      color: '#1a1612',
    },
    xaxis: {
      title: { text: 'Año', standoff: 12, font: { size: 12, color: '#4a443c' } },
      showgrid: true,
      gridcolor: '#e3d8c1',
      gridwidth: 1,
      linecolor: '#1a1612',
      tickformat: 'd',
      zeroline: false,
    },
    yaxis: {
      title: { text: SERIE_LABEL[estado.serie], standoff: 12, font: { size: 12, color: '#4a443c' } },
      showgrid: true,
      gridcolor: '#e3d8c1',
      gridwidth: 1,
      linecolor: '#1a1612',
      zeroline: false,
      rangemode: 'tozero',
    },
    legend: {
      orientation: 'h',
      x: 0,
      y: -0.22,
      bgcolor: 'rgba(0,0,0,0)',
      font: { size: 11 },
    },
    hovermode: 'x unified',
    hoverlabel: {
      bgcolor: '#1a1612',
      bordercolor: '#1a1612',
      font: { family: "'IBM Plex Mono', monospace", size: 11, color: '#faf5e8' },
    },
    shapes: [
      // Línea vertical entre histórico y pronóstico
      {
        type: 'line',
        x0: xUlt + 0.5,
        x1: xUlt + 0.5,
        y0: 0,
        y1: 1,
        yref: 'paper',
        line: { color: '#d3c6ad', width: 1, dash: 'dash' },
      },
    ],
    annotations: [
      {
        x: xUlt + 0.5,
        y: 1,
        xref: 'x',
        yref: 'paper',
        text: 'Pronóstico →',
        showarrow: false,
        font: {
          family: "'IBM Plex Mono', monospace",
          size: 10,
          color: '#8b8275',
        },
        xanchor: 'left',
        yanchor: 'top',
        xshift: 4,
        yshift: -4,
      },
    ],
  };

  Plotly.newPlot('grafico', traces, layout, {
    responsive: true,
    displaylogo: false,
    modeBarButtonsToRemove: ['lasso2d', 'select2d', 'autoScale2d'],
  });
}

function renderRanking(info) {
  const tbody = $('tabla-ranking').querySelector('tbody');
  tbody.innerHTML = '';
  const orden = Object.keys(info.modelos)
    .sort((a, b) => info.modelos[a].rmse - info.modelos[b].rmse);
  const elegido = modeloElegido(info);

  orden.forEach((nombre, i) => {
    const m = info.modelos[nombre];
    const tr = document.createElement('tr');
    if (nombre === elegido) tr.classList.add('is-best');
    const pred = m.prediccion.map(p => fmtInt(p)).join(' · ');
    const detalle = m.info ? `<span class="mono">${m.info}</span>` : '';
    tr.innerHTML = `
      <td>${i + 1}</td>
      <td>${nombre}</td>
      <td class="num">${m.rmse.toFixed(2)}</td>
      <td>${detalle}</td>
      <td class="num"><span class="pred-list">${pred}</span></td>
    `;
    tbody.appendChild(tr);
  });

  $('ranking-criterio').textContent =
    `Ordenados por ${info.criterio} · ${orden.length} modelos`;
}

function renderPredicciones(info) {
  const elegido = modeloElegido(info);
  const m = info.modelos[elegido];
  const tbody = $('tabla-predicciones').querySelector('tbody');
  tbody.innerHTML = '';
  info.anios_fut.forEach((anio, i) => {
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td>${anio}</td>
      <td class="num">${fmtInt(m.prediccion[i])}</td>
    `;
    tbody.appendChild(tr);
  });
  $('pred-modelo').textContent = `Según ${elegido}`;
}

function render() {
  const info = DATA.columnas[estado.serie].datos[estado.carrera];
  if (!info) return;
  renderResumen(info);
  renderGrafico(info);
  renderRanking(info);
  renderPredicciones(info);
}

// ---------- Descarga CSV ----------
function descargarCSV() {
  const info = DATA.columnas[estado.serie].datos[estado.carrera];
  if (!info) return;
  const elegido = modeloElegido(info);
  const m = info.modelos[elegido];

  const filas = [
    ['Licenciatura', estado.carrera],
    ['Indicador', SERIE_LABEL[estado.serie]],
    ['Modelo', elegido],
    ['Criterio de selección', info.criterio],
    [],
    ['Año', 'Valor estimado'],
  ];
  info.anios_fut.forEach((a, i) => filas.push([a, Math.round(m.prediccion[i])]));
  filas.push([]);
  filas.push(['---', 'Comparativa de modelos (RMSE)']);
  Object.entries(info.modelos)
    .sort((a, b) => a[1].rmse - b[1].rmse)
    .forEach(([nombre, mm]) => filas.push([nombre, mm.rmse.toFixed(3)]));

  const csv = filas
    .map(f => f.map(c => `"${String(c).replace(/"/g, '""')}"`).join(','))
    .join('\n');
  const blob = new Blob(['\ufeff' + csv], { type: 'text/csv;charset=utf-8' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  const slug = nombreCarreraCorta(estado.carrera)
    .replace(/[^a-zA-Z0-9]+/g, '_').slice(0, 40);
  a.download = `prediccion_${estado.serie}_${slug}.csv`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

// ---------- Event listeners ----------
function bindControls() {
  $('sel-serie').addEventListener('change', e => {
    estado.serie = e.target.value;
    poblarCarreras();
    poblarModelos();
    render();
  });
  $('sel-carrera').addEventListener('change', e => {
    estado.carrera = e.target.value;
    poblarModelos();
    render();
  });
  $('sel-modelo').addEventListener('change', e => {
    estado.modelo = e.target.value;
    render();
  });
  $('btn-descargar').addEventListener('click', descargarCSV);
  window.addEventListener('resize', () => {
    if (DATA && estado.carrera && typeof Plotly !== 'undefined') {
      try { Plotly.Plots.resize($('grafico')); } catch (e) {}
    }
  });
}

// ---------- Boot ----------
(async function init() {
  try {
    await cargarDatos();
    poblarCarreras();
    poblarModelos();
    bindControls();
    render();
  } catch (e) {
    console.error(e);
    document.querySelector('main').innerHTML =
      `<div style="padding:2rem;color:#b54a2f">Error: ${e.message}</div>`;
  }
})();
