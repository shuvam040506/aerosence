const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => Array.from(document.querySelectorAll(sel));

const fmt = (n, d=1) => (n == null || isNaN(n)) ? '—' : Number(n).toFixed(d);

function pmBadge(value) {
  if (value == null) return '<span class="pm-badge">—</span>';
  if (value <= 12) return '<span class="pm-badge good">Good</span>';
  if (value <= 35.4) return '<span class="pm-badge moderate">Moderate</span>';
  if (value <= 150.4) return '<span class="pm-badge unhealthy">Unhealthy</span>';
  return '<span class="pm-badge hazard">Hazardous</span>';
}

function renderWeatherCards(current) {
  return `
    <div class="card">
      <h3>Temperature</h3>
      <div class="kpi">${fmt(current.temperature_2m, 1)} °C</div>
      <div class="meta">Feels like ${fmt(current.apparent_temperature, 1)} °C</div>
    </div>
    <div class="card">
      <h3>Humidity</h3>
      <div class="kpi">${fmt(current.relative_humidity_2m, 0)}%</div>
      <div class="meta">Surface pressure ${fmt(current.surface_pressure, 0)} hPa</div>
    </div>
    <div class="card">
      <h3>Wind</h3>
      <div class="kpi">${fmt(current.wind_speed_10m, 1)} m/s</div>
      <div class="meta">Direction ${fmt(current.wind_direction_10m, 0)}°</div>
    </div>
    <div class="card">
      <h3>Precipitation</h3>
      <div class="kpi">${fmt(current.precipitation, 2)} mm</div>
      <div class="meta">Updated at ${current.time?.replace('T',' ') || '—'}</div>
    </div>
    <div class="card compact">
      <h3>Conditions</h3>
      <div class="kpi" style="font-size:18px">${(current.is_day? 'Day':'Night')} • Code ${fmt(current.weather_code,0)}</div>
      <div class="meta">WMO weather_code</div>
    </div>
  `;
}

function latestHourly(weather, key) {
  const times = weather.hourly?.time;
  const arr = weather.hourly?.[key];
  if (!times || !arr || !arr.length) return null;
  // Find most recent non-null value from the end
  for (let i = arr.length - 1; i >= 0; i--) {
    const v = arr[i];
    if (v != null && !Number.isNaN(Number(v))) return v;
  }
  return null;
}

function renderPredictionCard(pm25, usingModel) {
  return `
    <div class="card">
      <h3>Predicted PM2.5</h3>
      <div class="kpi">${fmt(pm25, 1)} μg/m³</div>
      <div class="meta">Model: ${usingModel}</div>
      <div class="pm-scale">
        <div class="bar"></div>
        <div class="bar"></div>
        <div class="bar"></div>
        <div class="bar"></div>
        <div class="bar"></div>
      </div>
      <div class="meta" style="margin-top:8px">Air quality: ${pmBadge(pm25)}</div>
      <div style="margin-top:14px">
        <canvas id="aqPie" height="120"></canvas>
      </div>
      <div class="legend-chips" id="aqLegend"></div>
    </div>
  `;
}

async function fetchWeatherByQuery(q) {
  const r = await fetch(`/api/weather?q=${encodeURIComponent(q)}`);
  if (!r.ok) throw new Error(`Failed: ${r.status}`);
  return r.json();
}

async function fetchWeatherByCoords(lat, lon, label) {
  const r = await fetch(`/api/weather?lat=${encodeURIComponent(lat)}&lon=${encodeURIComponent(lon)}&label=${encodeURIComponent(label||'')}`);
  if (!r.ok) throw new Error(`Failed: ${r.status}`);
  return r.json();
}

async function geocode(q) {
  const r = await fetch(`/api/geocode?q=${encodeURIComponent(q)}`);
  if (!r.ok) throw new Error(`Geocode failed: ${r.status}`);
  return r.json();
}

async function predict(input) {
  const r = await fetch('/api/predict', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(input) });
  if (!r.ok) throw new Error('Prediction error');
  return r.json();
}

let pieChartInstance = null;

function clearChart() {
  if (pieChartInstance) {
    pieChartInstance.destroy();
    pieChartInstance = null;
  }
}

$('#searchForm').addEventListener('submit', async (e) => {
  e.preventDefault();
  const q = $('#query').value.trim();
  if (!q) return;

  $('#location').textContent = 'Searching...';
  $('#weather').innerHTML = '<div class="card"><div class="skeleton title"></div><div class="skeleton line"></div><div class="skeleton line"></div></div>';
  $('#prediction').innerHTML = '<div class="card placeholder"><div class="skeleton title"></div><div class="skeleton line"></div><div class="skeleton line"></div></div>';
  clearChart();
  const geoWrap = $('#geoResults');
  if (geoWrap) geoWrap.innerHTML = '';

  try {
    // 1) Geocode to offer choices
    const g = await geocode(q);
    if (Array.isArray(g?.results) && g.results.length > 0) {
      const list = document.createElement('ul');
      list.className = 'geo-list';
      g.results.slice(0, 5).forEach((it) => {
        const li = document.createElement('li');
        li.className = 'geo-item';
        li.innerHTML = `
          <div class="left">
            <span class="name">${it.name}${it.admin1 ? ', ' + it.admin1 : ''}</span>
            <span class="sub">${it.country} • ${it.latitude.toFixed(3)}, ${it.longitude.toFixed(3)}</span>
          </div>
          <span class="badge">${it.timezone || ''}</span>
        `;
        li.addEventListener('click', async () => {
          // When selected, fetch weather by coords
          $('#location').textContent = 'Loading weather...';
          $('#weather').innerHTML = '<div class="card"><div class="skeleton title"></div><div class="skeleton line"></div><div class="skeleton line"></div></div>';
          $('#prediction').innerHTML = '<div class="card placeholder"><div class="skeleton title"></div><div class="skeleton line"></div><div class="skeleton line"></div></div>';
          clearChart();
          try {
            const label = `${it.name}${it.admin1 ? ', ' + it.admin1 : ''}`;
            const data = await fetchWeatherByCoords(it.latitude, it.longitude, label);
            renderAll(data);
          const gw = document.getElementById('geoResults');
          if (gw) gw.innerHTML = '';
          } catch (err) {
            showError(err);
          }
        });
        list.appendChild(li);
      });
      if (geoWrap) {
        geoWrap.innerHTML = '<div class="meta" style="padding:0 2px 6px">Select a location:</div>';
        geoWrap.appendChild(list);
      }
      $('#location').textContent = 'Pick a location from the list above.';
      $('#weather').innerHTML = '';
      $('#prediction').innerHTML = '';
      return; // Wait for user selection
    }

    // Fallback: direct weather by query if no geocode results
    const data = await fetchWeatherByQuery(q);
    renderAll(data);

  } catch (err) {
    showError(err);
  }
});

function showError(err) {
  const msg = err?.message || 'Error';
  $('#location').textContent = msg;
  $('#weather').innerHTML = '';
  $('#prediction').innerHTML = `<div class="card"><h3>Error</h3><div class="meta">${msg}</div></div>`;
}

async function renderAll(data) {
  const { geocoding, weather } = data;
  const cur = weather.current || {};
  const parts = [geocoding?.name, geocoding?.country].filter(Boolean);
  const place = parts.join(', ');
  $('#location').innerHTML = `<span class="badge"><span class="dot"></span> ${place} • ${geocoding?.timezone || ''}</span>`;
  $('#weather').innerHTML = renderWeatherCards(cur);

  // Build features for prediction
  const features = {
    temperature_2m: cur.temperature_2m,
    relative_humidity_2m: cur.relative_humidity_2m,
    apparent_temperature: cur.apparent_temperature,
    precipitation: cur.precipitation,
    wind_speed_10m: cur.wind_speed_10m,
    surface_pressure: cur.surface_pressure,
    uv_index: latestHourly(weather, 'uv_index'),
    dew_point_2m: latestHourly(weather, 'dew_point_2m'),
    pm10: latestHourly(weather, 'pm10')
  };

  const pred = await predict(features);
  if (typeof pred?.pm25 === 'number') {
    $('#prediction').innerHTML = renderPredictionCard(pred.pm25, pred.model);
    // Build pollutant composition
    const pm25Val = pred.pm25;
    const pm10Val = latestHourly(weather, 'pm10') ?? 0;
    const no2Val = latestHourly(weather, 'nitrogen_dioxide') ?? 0;
    const so2Val = latestHourly(weather, 'sulphur_dioxide') ?? 0;
    const o3Val = latestHourly(weather, 'ozone') ?? 0;
    const values = [pm25Val, pm10Val, no2Val, so2Val, o3Val].map(v => Math.max(0, Number(v) || 0));
    const pieData = values.map(v => v);
    const ctx = document.getElementById('aqPie');
    if (ctx && window.Chart) {
      const baseColors = ['#6ee7ff','#a78bfa','#f59e0b','#10b981','#ef4444'];
      const labels = ['PM2.5','PM10','NO2','SO2','O3'];
      clearChart();
      pieChartInstance = new window.Chart(ctx, {
        type: 'doughnut',
        data: { labels, datasets: [{ data: pieData, backgroundColor: baseColors, borderColor: '#0f121a', borderWidth: 1 }] },
        options: {
          responsive: true,
          maintainAspectRatio: true,
          plugins: {
            legend: { display: false },
            tooltip: { callbacks: { label: (c) => `${c.label}: ${fmt(c.parsed,1)} μg/m³` } }
          },
          cutout: '58%'
        }
      });

      // Legend chips
      const legend = document.getElementById('aqLegend');
      if (legend) {
        legend.innerHTML = labels.map((name, i) => `
          <span class="chip"><span class="swatch" style="background:${baseColors[i]}"></span>${name}</span>
        `).join('');
      }
    }
  } else {
    $('#prediction').innerHTML = `<div class="card"><h3>No prediction</h3><div class="meta">The server did not return a PM2.5 value.</div></div>`;
  }
}
