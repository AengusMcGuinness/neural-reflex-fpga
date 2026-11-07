const token = document.querySelector('meta[name="arm-token"]').content;
const connection = document.getElementById('connection');
const torque = document.getElementById('torque');
const modeState = document.getElementById('mode-state');
const notice = document.getElementById('notice');
const timer = document.getElementById('timer');
const limits = document.getElementById('limits');
const updated = document.getElementById('updated');
const connectionHelp = document.getElementById('connection-help');
const faultPanel = document.getElementById('fault-panel');
const faultText = document.getElementById('fault-text');
const joints = document.getElementById('joints');
const armButton = document.getElementById('arm');
const freeButton = document.getElementById('free');
const stopButton = document.getElementById('stop');
const retryButton = document.getElementById('retry');
const stepSelect = document.getElementById('step');

let current = null;
let busy = false;
let polling = false;
let requestSequence = 0;
let localError = '';
const rows = new Map();

function label(name) {
  return name === 'gripper' ? 'Handle / gripper' : name.replaceAll('_', ' ');
}

function currentJoint(name) {
  return current?.joints?.find((joint) => joint.name === name);
}

function safeRange(joint) {
  const lower = joint.target_minimum;
  const upper = joint.target_maximum;
  if (!Number.isInteger(lower) || !Number.isInteger(upper) || lower > upper ||
      lower < joint.minimum || upper > joint.maximum) return null;
  return { lower, upper };
}

function clamp(value, lower, upper) {
  return Math.max(lower, Math.min(upper, value));
}

function metric(value, unit) {
  return typeof value === 'number' && Number.isFinite(value)
    ? `${value.toFixed(1)} ${unit}` : '—';
}

function createRow(joint) {
  const row = document.createElement('article');
  row.className = 'joint';
  const top = document.createElement('div');
  top.className = 'joint-top';
  const identity = document.createElement('div');
  const name = document.createElement('span');
  name.className = 'joint-name';
  name.textContent = label(joint.name);
  const position = document.createElement('span');
  position.className = 'joint-position';
  identity.append(name, position);
  const toggle = document.createElement('button');
  toggle.type = 'button';
  toggle.className = 'torque-toggle';
  toggle.setAttribute('role', 'switch');
  toggle.setAttribute('aria-label', `${label(joint.name)} motor torque`);
  toggle.addEventListener('click', () => {
    const latest = currentJoint(joint.name);
    if (latest && typeof latest.torque_enabled === 'boolean') {
      action('/api/torque', { joint: joint.name, enabled: !latest.torque_enabled });
    }
  });
  top.append(identity, toggle);

  const metrics = document.createElement('div');
  metrics.className = 'joint-metrics';
  const motorLimits = document.createElement('span');
  const operatingMode = document.createElement('span');
  const voltage = document.createElement('span');
  const temperature = document.createElement('span');
  const torqueLimit = document.createElement('span');
  metrics.append(motorLimits, operatingMode, voltage, temperature, torqueLimit);

  const motion = document.createElement('div');
  motion.className = 'joint-motion';
  const buttons = document.createElement('div');
  buttons.className = 'joint-buttons';
  const stepButtons = [];
  for (const sign of [-1, 1]) {
    const button = document.createElement('button');
    button.type = 'button';
    button.textContent = sign < 0 ? '−' : '+';
    button.setAttribute('aria-label',
      `${sign < 0 ? 'Decrease' : 'Increase'} ${label(joint.name)} by selected step`);
    button.addEventListener('click', () => action('/api/jog', {
      joint: joint.name, delta: sign * Number(stepSelect.value),
    }));
    stepButtons.push(button);
    buttons.append(button);
  }

  const targetControl = document.createElement('div');
  targetControl.className = 'target-control';
  const targetLabel = document.createElement('label');
  const range = document.createElement('input');
  range.type = 'range';
  range.step = '1';
  range.id = `target-${joint.name}`;
  range.setAttribute('aria-label', `${label(joint.name)} target in raw ticks`);
  targetLabel.htmlFor = range.id;
  targetLabel.textContent = 'Target';
  const number = document.createElement('input');
  number.type = 'number';
  number.step = '1';
  number.setAttribute('aria-label', `${label(joint.name)} target in raw ticks`);
  const move = document.createElement('button');
  move.type = 'button';
  move.textContent = 'Move';
  move.setAttribute('aria-label', `Move ${label(joint.name)} to target`);

  const ui = {
    row, position, toggle, motorLimits, operatingMode, voltage, temperature, torqueLimit,
    stepButtons, range, number, move, targetRange: null, dirty: false,
  };
  range.addEventListener('input', () => {
    ui.dirty = true;
    number.value = range.value;
  });
  range.addEventListener('change', () => sendTarget(joint.name, Number(range.value)));
  number.addEventListener('input', () => {
    ui.dirty = true;
    const value = Number(number.value);
    const bounds = safeRange(currentJoint(joint.name) || joint);
    if (number.value !== '' && Number.isInteger(value) && bounds &&
        value >= bounds.lower && value <= bounds.upper) range.value = String(value);
  });
  number.addEventListener('keydown', (event) => {
    if (event.key === 'Enter') {
      event.preventDefault();
      sendTarget(joint.name, Number(number.value));
    } else if (event.key === 'Escape') {
      ui.dirty = false;
      if (current) render(current);
      number.blur();
    }
  });
  move.addEventListener('click', () => sendTarget(joint.name, Number(number.value)));
  targetControl.append(targetLabel, range, number, move);
  motion.append(buttons, targetControl);
  const targetRange = document.createElement('p');
  targetRange.className = 'target-range';
  row.append(top, metrics, motion, targetRange);
  joints.append(row);
  ui.targetRange = targetRange;
  rows.set(joint.name, ui);
  return ui;
}

function render(state) {
  current = state;
  const jointList = Array.isArray(state.joints) ? state.joints : [];
  const enabledCount = jointList.filter((joint) => joint.torque_enabled === true).length;
  const allHeld = jointList.length > 0 && enabledCount === jointList.length;
  const controlMode = state.control_mode === 'position' || state.control_mode === 'manual'
    ? state.control_mode : enabledCount ? 'position' : 'manual';

  connection.textContent = state.fault ? 'Motor fault' : 'Arm connected';
  connection.className = state.fault ? 'badge error' : 'badge on';
  torque.textContent = state.torque_unknown ? 'Torque unverified'
    : allHeld ? 'All motors on'
    : enabledCount ? `${enabledCount} of ${jointList.length} motors on`
    : 'Motors off';
  torque.className = state.torque_unknown ? 'badge error'
    : allHeld ? 'badge on' : enabledCount ? 'badge mixed' : 'badge off';
  modeState.textContent = state.torque_unknown ? 'Mode unknown'
    : controlMode === 'position' ? 'Position control' : 'Free/manual';
  modeState.className = state.torque_unknown ? 'badge error'
    : controlMode === 'position' ? 'badge on' : 'badge off';
  armButton.setAttribute('aria-pressed', String(controlMode === 'position'));
  freeButton.setAttribute('aria-pressed', String(controlMode === 'manual'));
  timer.textContent = enabledCount || state.armed
    ? `Auto stop in ${state.remaining_seconds ?? 0}s` : '';
  notice.textContent = localError || state.notice || '';
  notice.className = localError || state.fault ? 'notice error' : 'notice';
  connectionHelp.hidden = true;
  faultPanel.hidden = !state.fault;
  faultText.textContent = state.fault || '';
  retryButton.disabled = busy;
  armButton.disabled = busy || Boolean(state.fault) || state.torque_unknown ||
    controlMode === 'position';
  freeButton.disabled = busy || Boolean(state.fault) || state.torque_unknown ||
    controlMode === 'manual';
  // Keep Stop available even while another request is in progress.
  stopButton.disabled = !busy && !state.armed && !enabledCount && !state.torque_unknown;
  stepSelect.disabled = Boolean(state.fault) || state.torque_unknown;
  const maxStep = Number.isInteger(state.max_step) ? state.max_step : 20;
  for (const option of stepSelect.options) option.disabled = Number(option.value) > maxStep;
  if (Number(stepSelect.value) > maxStep) {
    const allowed = [...stepSelect.options].filter((option) => !option.disabled);
    if (allowed.length) stepSelect.value = allowed[allowed.length - 1].value;
  }
  limits.textContent = `Up to ${maxStep} ticks per step · ${state.max_travel ?? '—'} ticks travel per joint from start.`;
  updated.textContent = `Updated ${new Date().toLocaleTimeString()}`;

  const seen = new Set();
  for (const joint of jointList) {
    seen.add(joint.name);
    const ui = rows.get(joint.name) || createRow(joint);
    const bounds = safeRange(joint);
    const movable = !busy && !state.fault && !state.torque_unknown &&
      joint.torque_enabled === true && (joint.operating_mode == null || joint.operating_mode === 0);
    ui.position.textContent = `${state.stale ? 'Last read' : 'Measured'} ${joint.position} ticks`;
    ui.toggle.textContent = joint.torque_enabled == null ? 'Torque unknown'
      : joint.torque_enabled ? 'Turn motor off' : 'Enable motor';
    ui.toggle.setAttribute('aria-checked', String(joint.torque_enabled === true));
    ui.toggle.disabled = busy || Boolean(state.fault) || state.torque_unknown ||
      typeof joint.torque_enabled !== 'boolean';
    ui.motorLimits.textContent = `Motor limits ${joint.minimum}–${joint.maximum} ticks`;
    ui.operatingMode.textContent = joint.operating_mode == null ? 'Operating mode —'
      : joint.operating_mode === 0 ? 'Operating mode 0 · position'
      : `Operating mode ${joint.operating_mode}`;
    ui.voltage.textContent = `Voltage ${metric(joint.voltage_v, 'V')}`;
    ui.temperature.textContent = `Temperature ${metric(joint.temperature_c, '°C')}`;
    ui.torqueLimit.textContent = joint.torque_limit == null
      ? 'Torque limit —'
      : `Torque limit ${joint.torque_limit}/${joint.max_torque_limit ?? '—'}`;
    for (const button of ui.stepButtons) button.disabled = !movable;
    ui.range.disabled = !movable || !bounds;
    ui.number.disabled = !movable || !bounds;
    ui.move.disabled = !movable || !bounds;
    if (bounds) {
      ui.range.min = ui.number.min = String(bounds.lower);
      ui.range.max = ui.number.max = String(bounds.upper);
      ui.targetRange.textContent = joint.torque_enabled === true
        ? `Target range ${bounds.lower}–${bounds.upper} ticks · slider sends on release`
        : `Click Enable motor to unlock this slider · target range ${bounds.lower}–${bounds.upper} ticks`;
      if (!ui.dirty) {
        const value = clamp(joint.position, bounds.lower, bounds.upper);
        ui.range.value = ui.number.value = String(value);
      }
    } else {
      ui.targetRange.textContent = 'Safe target range unavailable';
      if (!ui.dirty) ui.number.value = '';
    }
  }
  for (const [name, ui] of rows) {
    if (!seen.has(name)) {
      ui.row.remove();
      rows.delete(name);
    }
  }
}

function disconnected(error) {
  current = null;
  connection.textContent = 'Connection lost';
  connection.className = 'badge error';
  torque.textContent = 'Torque unknown';
  torque.className = 'badge error';
  modeState.textContent = 'Mode unknown';
  modeState.className = 'badge error';
  timer.textContent = '';
  notice.textContent = error.message || String(error);
  notice.className = 'notice error';
  connectionHelp.hidden = false;
  faultPanel.hidden = true;
  armButton.disabled = true;
  freeButton.disabled = true;
  stopButton.disabled = true;
  retryButton.disabled = true;
  stepSelect.disabled = true;
  updated.textContent = '';
  for (const ui of rows.values()) {
    ui.toggle.disabled = true;
    ui.range.disabled = true;
    ui.number.disabled = true;
    ui.move.disabled = true;
    for (const button of ui.stepButtons) button.disabled = true;
  }
}

async function refresh() {
  if (busy || polling) return;
  polling = true;
  const sequence = requestSequence;
  try {
    const response = await fetch('/api/status', { cache: 'no-store' });
    const state = await response.json();
    if (!response.ok) throw new Error(state.error || 'Could not read arm');
    if (sequence === requestSequence && !busy) render(state);
  } catch (error) {
    if (sequence === requestSequence && !busy) disconnected(error);
  } finally {
    polling = false;
  }
}

async function action(path, payload = {}) {
  if (busy && path !== '/api/stop') return;
  const sequence = ++requestSequence;
  busy = true;
  localError = '';
  if (current) render(current);
  try {
    const response = await fetch(path, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'X-Arm-Token': token },
      body: JSON.stringify(payload),
    });
    const state = await response.json();
    if (!response.ok) throw new Error(state.error || 'Arm command failed');
    if (sequence === requestSequence) {
      busy = false;
      render(state);
    }
  } catch (error) {
    if (sequence === requestSequence) {
      busy = false;
      await refresh();
      localError = error.message || String(error);
      notice.textContent = localError;
      notice.className = 'notice error';
      if (/token/i.test(localError)) connectionHelp.hidden = false;
    }
  }
}

async function sendTarget(name, target) {
  const joint = currentJoint(name);
  const bounds = joint && safeRange(joint);
  if (!bounds || !Number.isInteger(target) || target < bounds.lower || target > bounds.upper) {
    localError = bounds
      ? `Enter a whole-number target from ${bounds.lower} to ${bounds.upper} ticks.`
      : 'The arm has not provided a safe target range.';
    notice.textContent = localError;
    notice.className = 'notice error';
    return;
  }
  await action('/api/target', { joint: name, target });
  const ui = rows.get(name);
  const latest = currentJoint(name);
  if (ui && latest) {
    ui.dirty = false;
    const latestBounds = safeRange(latest);
    if (latestBounds) {
      const value = clamp(latest.position, latestBounds.lower, latestBounds.upper);
      ui.range.value = ui.number.value = String(value);
    }
  }
}

armButton.addEventListener('click', () => action('/api/mode', { mode: 'position' }));
freeButton.addEventListener('click', () => action('/api/mode', { mode: 'manual' }));
stopButton.addEventListener('click', () => action('/api/stop'));
retryButton.addEventListener('click', () => action('/api/retry'));
refresh();
setInterval(refresh, 1000);
document.addEventListener('visibilitychange', () => {
  if (!document.hidden) refresh();
});
