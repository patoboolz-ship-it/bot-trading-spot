const chartEl = document.getElementById('chart');
const logBox = document.getElementById('logBox');
const equityEl = document.getElementById('equity');
const balancesEl = document.getElementById('balances');
const ordersTbody = document.querySelector('#orders tbody');
const tradesTbody = document.querySelector('#trades tbody');
const stateEl = document.getElementById('botState');

const chart = LightweightCharts.createChart(chartEl, {
  layout: { background: { color: '#020617' }, textColor: '#cbd5e1' },
  grid: { vertLines: { color: '#1e293b' }, horzLines: { color: '#1e293b' } },
  rightPriceScale: { borderColor: '#334155' }, timeScale: { borderColor: '#334155' }
});

const candleSeries = chart.addCandlestickSeries();
const buyMarkers = [];
const sellMarkers = [];
let liveMode = true;
let pending = [];

function log(msg) {
  const ts = new Date().toLocaleTimeString();
  logBox.textContent = `[${ts}] ${msg}\n` + logBox.textContent;
}

function setMarkers() {
  candleSeries.setMarkers([...buyMarkers, ...sellMarkers].sort((a,b)=>a.time-b.time));
}

function updateAccount(a) {
  equityEl.innerHTML = `
    <div><b>Equity:</b> ${a.equity?.toFixed?.(2) ?? a.equity} USDT</div>
    <div><b>PnL:</b> ${a.pnl_usdt?.toFixed?.(2)} USDT (${a.pnl_pct?.toFixed?.(2)}%)</div>
    <div><b>USDT disponible:</b> ${(a.balances?.USDT ?? 0).toFixed?.(2)}</div>
  `;
  const assets = (a.assets || []).map(x => `<div>${x.asset}: ${Number(x.qty).toFixed(4)} (~${Number(x.est_usdt).toFixed(2)} USDT)</div>`).join('');
  balancesEl.innerHTML = assets || '<div>Sin posiciones spot</div>';
}

function renderTable(tbody, rows, cols) {
  tbody.innerHTML = rows.slice(-20).reverse().map(r => `<tr>${cols.map(c => `<td>${r[c] ?? ''}</td>`).join('')}</tr>`).join('');
}

function applySnapshot(s) {
  document.getElementById('symbol').value = s.symbol;
  document.getElementById('interval').value = s.interval;
  stateEl.textContent = s.bot_state;
  candleSeries.setData((s.market_state.candles || []).map(c => ({
    time: c.time, open: c.open, high: c.high, low: c.low, close: c.close
  })));
  updateAccount(s.account_state);
  renderTable(ordersTbody, s.orders_state || [], ['status', 'side', 'origQty', 'time']);
  renderTable(tradesTbody, s.trades_state || [], ['isBuyer', 'price', 'qty', 'commission', 'time']);
  chart.timeScale().scrollToRealTime();
}

function onMarket(payload) {
  const items = payload.candles || [];
  if (!liveMode) { pending.push(...items); return; }
  items.forEach(c => candleSeries.update({ time: c.time, open: c.open, high: c.high, low: c.low, close: c.close }));
}

async function post(url, body={}) {
  await fetch(url, { method:'POST', headers:{'content-type':'application/json'}, body: JSON.stringify(body) });
}

['play','pause','stop'].forEach(id => {
  document.getElementById(id).onclick = () => post(`/control/${id}`);
});

document.getElementById('symbol').onchange = async e => {
  await post('/symbol', { symbol: e.target.value });
  log(`Símbolo => ${e.target.value}`);
};

document.getElementById('interval').onchange = async e => {
  await post('/timeframe', { interval: e.target.value });
  log(`Temporalidad => ${e.target.value}`);
};

document.getElementById('liveMode').onchange = e => {
  liveMode = e.target.checked;
  log(`Modo ${liveMode ? 'LIVE' : 'HISTÓRICO'}`);
};

document.getElementById('goLive').onclick = () => {
  liveMode = true;
  document.getElementById('liveMode').checked = true;
  pending.forEach(c => candleSeries.update({ time: c.time, open: c.open, high: c.high, low: c.low, close: c.close }));
  pending = [];
  chart.timeScale().scrollToRealTime();
};

function connectWS() {
  const wsProto = location.protocol === 'https:' ? 'wss' : 'ws';
  const ws = new WebSocket(`${wsProto}://${location.host}/stream`);
  ws.onopen = () => log('WS conectado');
  ws.onclose = () => { log('WS desconectado, reconectando...'); setTimeout(connectWS, 1500); };
  ws.onmessage = ev => {
    const msg = JSON.parse(ev.data);
    if (msg.type === 'snapshot') applySnapshot(msg.payload);
    if (msg.type === 'market.candle_update') onMarket(msg.payload);
    if (msg.type === 'account.update') updateAccount(msg.payload);
    if (msg.type === 'orders.update') renderTable(ordersTbody, msg.payload.orders || [], ['status', 'side', 'origQty', 'time']);
    if (msg.type === 'trades.update') renderTable(tradesTbody, msg.payload.trades || [], ['isBuyer', 'price', 'qty', 'commission', 'time']);
    if (msg.type === 'bot.state') stateEl.textContent = msg.payload.state;
  };
}

connectWS();
