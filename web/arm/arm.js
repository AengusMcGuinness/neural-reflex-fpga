const token = document.querySelector('meta[name="arm-token"]').content;
const connection = document.getElementById('connection');
const torque = document.getElementById('torque');
const modeState = document.getElementById('mode-state');
const motionState = document.getElementById('motion-state');
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
const pauseButton = document.getElementById('pause');
const stopButton = document.getElementById('stop');
const retryButton = document.getElementById('retry');
const stepSelect = document.getElementById('step');
const speedSelect = document.getElementById('speed');
const rangeView = document.getElementById('range-view');
const recenterButton = document.getElementById('recenter');

const rows = new Map();
const pendingTargets = new Map();
let current = null;
let motionEpoch = null;
let busy = false;
let polling = false;
let heartbeatInFlight = false;
let motionLease = false;
let pausing = false;
let criticalRequest = false;
let focusStopPending = false;
let reloadRequired = false;
let requestSequence = 0;
let postSequence = 0;
let motionGeneration = 0;
let localError = '';
let targetTimer = null;
let targetInFlight = null;
let nextTargetAt = 0;
let activeHoldCancel = null;

function label(name) {
  return name === 'gripper' ? 'Handle / gripper' : name.replaceAll('_', ' ');
}

function currentJoint(name) {
  return current?.joints?.find((joint) => joint.name === name);
}

function safeRange(joint) {
  if (!joint) return null;
  const lower = joint.target_minimum;
  const upper = joint.target_maximum;
  if (!Number.isInteger(lower) || !Number.isInteger(upper) || lower > upper ||
      lower < joint.minimum || upper > joint.maximum) return null;
  return { lower, upper };
}

function sliderRange(joint, ui) {
  const full = safeRange(joint);
  if (!full) return null;
  if (rangeView.value !== 'fine') return full;
  if (!Number.isInteger(ui.fineCenter)) ui.fineCenter = joint.position;
  return {
    lower: Math.max(full.lower, ui.fineCenter - 160),
    upper: Math.min(full.upper, ui.fineCenter + 160),
  };
}

function clamp(value, lower, upper) {
  return Math.max(lower, Math.min(upper, value));
}

function metric(value, unit) {
  return typeof value === 'number' && Number.isFinite(value)
    ? value.toFixed(1) + ' ' + unit : '—';
}

function showError(error) {
  localError = error.message || String(error);
  notice.textContent = localError;
  notice.className = 'notice error';
  if (/token/i.test(localError)) connectionHelp.hidden = false;
}

function requireReload(message) {
  reloadRequired = true;
  disconnected(new Error(message));
  connectionHelp.textContent = 'The control server changed. Reload this page to get its new control token.';
}

function clearPendingMotion() {
  motionGeneration += 1;
  pendingTargets.clear();
  if (targetTimer !== null) {
    clearTimeout(targetTimer);
    targetTimer = null;
  }
  if (targetInFlight) targetInFlight.abort.abort();
  if (activeHoldCancel) activeHoldCancel(false);
  motionLease = false;
  for (const ui of rows.values()) {
    ui.dragging = false;
    ui.dirty = false;
    ui.localTarget = null;
  }
}

async function post(path, payload, signal) {
  const response = await fetch(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', 'X-Arm-Token': token },
    body: JSON.stringify(payload),
    signal,
  });
  const state = await response.json();
  if (!response.ok) throw new Error(state.error || 'Arm command failed');
  return state;
}

function queueTarget(name, target, flush = false) {
  if (reloadRequired) return false;
  const joint = currentJoint(name);
  const bounds = safeRange(joint);
  if (!bounds || !Number.isInteger(target) || target < bounds.lower || target > bounds.upper) {
    showError(new Error(bounds
      ? 'Enter a whole-number target from ' + bounds.lower + ' to ' + bounds.upper + ' ticks.'
      : 'The arm has not provided a safe target range.'));
    return false;
  }
  if (busy || current.fault || current.torque_unknown || joint.torque_enabled !== true ||
      motionEpoch === null || document.hidden || !document.hasFocus()) return false;
  const ui = rows.get(name);
  if (ui) {
    ui.localTarget = target;
    ui.dirty = true;
    ui.number.value = String(target);
    const visible = sliderRange(joint, ui);
    if (visible) ui.range.value = String(clamp(target, visible.lower, visible.upper));
  }
  localError = '';
  pendingTargets.set(name, { target, flush });
  motionLease = true;
  pauseButton.disabled = false;
  scheduleTarget(flush);
  return true;
}

function sendNumberTarget(name, number) {
  if (number.value.trim() === '') {
    showError(new Error('Enter a target tick before moving.'));
    return;
  }
  queueTarget(name, Number(number.value), true);
}

function scheduleTarget(flush = false) {
  if (targetInFlight || busy || pendingTargets.size === 0) return;
  if (targetTimer !== null) {
    if (!flush) return;
    clearTimeout(targetTimer);
  }
  const next = pendingTargets.values().next().value;
  const delay = flush || next.flush ? 0 : Math.max(0, nextTargetAt - Date.now());
  targetTimer = setTimeout(sendNextTarget, delay);
}

async function sendNextTarget() {
  targetTimer = null;
  if (targetInFlight || busy || pendingTargets.size === 0 || motionEpoch === null) return;
  const [name, item] = pendingTargets.entries().next().value;
  pendingTargets.delete(name);
  const epoch = motionEpoch;
  const generation = motionGeneration;
  const abort = new AbortController();
  targetInFlight = { name, abort, generation };
  nextTargetAt = Date.now() + 100;
  const sequence = ++postSequence;
  try {
    const state = await post('/api/target',
      { joint: name, target: item.target, motion_epoch: epoch }, abort.signal);
    if (generation !== motionGeneration) return;
    const acknowledged = state.joints?.find((joint) => joint.name === name);
    const ui = rows.get(name);
    if (sequence === postSequence && !busy) {
      if (ui && acknowledged?.target === item.target && !pendingTargets.has(name) &&
          !ui.dragging && ui.localTarget === item.target) {
        ui.dirty = false;
        ui.localTarget = null;
      }
      render(state);
    }
  } catch (error) {
    if (generation === motionGeneration && error.name !== 'AbortError') {
      if (/token/i.test(error.message || '')) {
        requireReload('Control token expired. Reload this page.');
        return;
      }
      if (sequence !== postSequence) return;
      clearPendingMotion();
      showError(error);
      refresh();
    }
  } finally {
    if (targetInFlight?.abort === abort) targetInFlight = null;
    scheduleTarget();
  }
}

function repeatHeldTarget(name, sign) {
  const joint = currentJoint(name);
  const bounds = safeRange(joint);
  if (!bounds || joint.torque_enabled !== true) return;
  const speed = Number(current?.max_speed_ticks_per_second) ||
    ({ fine: 60, normal: 180, fast: 360 }[speedSelect.value] || 60);
  const commanded = Number.isInteger(joint.commanded_position)
    ? joint.commanded_position : joint.position;
  const lead = Math.max(1, Math.floor(speed * 0.25));
  const target = clamp(commanded + sign * lead, bounds.lower, bounds.upper);
  if (target === joint.target && !pendingTargets.has(name)) return;
  queueTarget(name, target);
}

function bindStepButton(button, name, sign) {
  let holdTimer = null;
  let repeatTimer = null;
  let pressing = false;
  let held = false;
  let suppressClick = false;

  function end(pause) {
    if (holdTimer !== null) clearTimeout(holdTimer);
    if (repeatTimer !== null) clearInterval(repeatTimer);
    holdTimer = repeatTimer = null;
    const wasHeld = held;
    pressing = held = false;
    if (activeHoldCancel === end) activeHoldCancel = null;
    if (wasHeld) {
      suppressClick = true;
      setTimeout(() => { suppressClick = false; }, 0);
      if (pause) pauseMotion();
    }
  }

  button.addEventListener('pointerdown', (event) => {
    if (button.disabled || event.button !== 0 || activeHoldCancel) return;
    pressing = true;
    activeHoldCancel = end;
    button.setPointerCapture?.(event.pointerId);
    holdTimer = setTimeout(() => {
      if (!pressing || button.disabled) return;
      held = true;
      repeatHeldTarget(name, sign);
      repeatTimer = setInterval(() => repeatHeldTarget(name, sign), 100);
    }, 300);
  });
  button.addEventListener('pointerup', () => end(true));
  button.addEventListener('pointercancel', () => end(true));
  button.addEventListener('lostpointercapture', () => {
    if (pressing) end(true);
  });
  button.addEventListener('click', (event) => {
    if (suppressClick) {
      event.preventDefault();
      suppressClick = false;
      return;
    }
    if (activeHoldCancel && activeHoldCancel !== end) return;
    if (!button.disabled && motionEpoch !== null) {
      motionLease = true;
      action('/api/jog', {
        joint: name,
        delta: sign * Number(stepSelect.value),
        motion_epoch: motionEpoch,
      });
    }
  });
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
  toggle.setAttribute('aria-label', label(joint.name) + ' motor torque');
  toggle.addEventListener('click', () => {
    const latest = currentJoint(joint.name);
    if (!latest || typeof latest.torque_enabled !== 'boolean') return;
    const enabled = !latest.torque_enabled;
    action('/api/torque',
      { joint: joint.name, enabled, motion_epoch: motionEpoch },
      { critical: !enabled });
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
    const lift = joint.name === 'shoulder_lift';
    button.textContent = lift ? (sign < 0 ? '↑ Raise' : '↓ Lower') : (sign < 0 ? '−' : '+');
    if (lift) button.classList.add('directional');
    button.setAttribute('aria-label', (lift
      ? (sign < 0 ? 'Raise ' : 'Lower ')
      : (sign < 0 ? 'Decrease ' : 'Increase ')) + label(joint.name) + ' by selected step');
    bindStepButton(button, joint.name, sign);
    stepButtons.push(button);
    buttons.append(button);
  }

  const targetControl = document.createElement('div');
  targetControl.className = 'target-control';
  const targetLabel = document.createElement('label');
  const range = document.createElement('input');
  range.type = 'range';
  range.step = '1';
  range.id = 'target-' + joint.name;
  range.setAttribute('aria-label', label(joint.name) + ' live target in raw ticks');
  targetLabel.htmlFor = range.id;
  targetLabel.textContent = 'Target';
  const number = document.createElement('input');
  number.type = 'number';
  number.step = '1';
  number.setAttribute('aria-label', label(joint.name) + ' target in raw ticks');
  const move = document.createElement('button');
  move.type = 'button';
  move.textContent = 'Move';
  move.setAttribute('aria-label', 'Move ' + label(joint.name) + ' to target');

  const ui = {
    row, position, toggle, motorLimits, operatingMode, voltage, temperature, torqueLimit,
    stepButtons, range, number, move, targetRange: null, dirty: false,
    dragging: false, localTarget: null, fineCenter: null,
  };
  range.addEventListener('pointerdown', () => { ui.dragging = true; });
  range.addEventListener('pointercancel', () => pauseMotion());
  range.addEventListener('input', () => {
    ui.dragging = true;
    queueTarget(joint.name, Number(range.value));
  });
  range.addEventListener('change', () => {
    ui.dragging = false;
    queueTarget(joint.name, Number(range.value), true);
  });
  number.addEventListener('input', () => {
    ui.dirty = true;
    const value = Number(number.value);
    const visible = sliderRange(currentJoint(joint.name) || joint, ui);
    if (number.value !== '' && Number.isInteger(value) && visible) {
      range.value = String(clamp(value, visible.lower, visible.upper));
    }
  });
  number.addEventListener('keydown', (event) => {
    if (event.key === 'Enter') {
      event.preventDefault();
      sendNumberTarget(joint.name, number);
    } else if (event.key === 'Escape') {
      ui.dirty = false;
      ui.localTarget = null;
      if (current) render(current);
      number.blur();
    }
  });
  move.addEventListener('click', () => sendNumberTarget(joint.name, number));
  targetControl.append(targetLabel, range, number, move);
  motion.append(buttons, targetControl);
  const targetRange = document.createElement('p');
  targetRange.className = 'target-range';
  row.append(top, metrics, motion);
  if (joint.name === 'shoulder_lift') {
    const directionHint = document.createElement('p');
    directionHint.className = 'direction-hint';
    directionHint.textContent = '← Raise: smaller ticks · Lower: larger ticks →';
    row.append(directionHint);
  }
  row.append(targetRange);
  joints.append(row);
  ui.targetRange = targetRange;
  rows.set(joint.name, ui);
  return ui;
}

function render(state) {
  if (reloadRequired) return;
  if (Number.isInteger(state.motion_epoch)) {
    if (motionEpoch !== null && state.motion_epoch < motionEpoch) {
      requireReload('Control server restarted. Reload this page.');
      return;
    }
    if (motionEpoch !== null && state.motion_epoch > motionEpoch) clearPendingMotion();
    motionEpoch = state.motion_epoch;
  }
  if (state.fault) clearPendingMotion();
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
    : enabledCount ? enabledCount + ' of ' + jointList.length + ' motors on'
    : 'Motors off';
  torque.className = state.torque_unknown ? 'badge error'
    : allHeld ? 'badge on' : enabledCount ? 'badge mixed' : 'badge off';
  modeState.textContent = state.torque_unknown ? 'Mode unknown'
    : controlMode === 'position' ? 'Position control' : 'Free/manual';
  modeState.className = state.torque_unknown ? 'badge error'
    : controlMode === 'position' ? 'badge on' : 'badge off';
  motionState.textContent = state.moving ? 'Moving' : 'Still';
  motionState.className = state.moving ? 'badge mixed' : 'badge off';
  armButton.setAttribute('aria-pressed', String(controlMode === 'position'));
  freeButton.setAttribute('aria-pressed', String(controlMode === 'manual'));
  timer.textContent = enabledCount || state.armed
    ? 'Auto stop in ' + (state.remaining_seconds ?? 0) + 's' : '';
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
  pauseButton.disabled = criticalRequest || !shouldPause() ||
    Boolean(state.fault) || state.torque_unknown;
  stopButton.disabled = !busy && !state.armed && !enabledCount && !state.torque_unknown;
  stepSelect.disabled = Boolean(state.fault) || state.torque_unknown;
  speedSelect.disabled = busy || Boolean(state.fault) || state.torque_unknown;
  if (['fine', 'normal', 'fast'].includes(state.speed)) speedSelect.value = state.speed;
  recenterButton.disabled = rangeView.value !== 'fine';
  const maxStep = Number.isInteger(state.max_step) ? state.max_step : 20;
  for (const option of stepSelect.options) option.disabled = Number(option.value) > maxStep;
  if (Number(stepSelect.value) > maxStep) {
    const allowed = [...stepSelect.options].filter((option) => !option.disabled);
    if (allowed.length) stepSelect.value = allowed[allowed.length - 1].value;
  }
  limits.textContent = 'Tap up to ' + maxStep + ' ticks · speed ' +
    (state.max_speed_ticks_per_second ?? '—') + ' ticks/s.';
  updated.textContent = 'Updated ' + new Date().toLocaleTimeString();

  const seen = new Set();
  for (const joint of jointList) {
    seen.add(joint.name);
    const ui = rows.get(joint.name) || createRow(joint);
    const full = safeRange(joint);
    const visible = sliderRange(joint, ui);
    const movable = !busy && !state.fault && !state.torque_unknown &&
      joint.torque_enabled === true &&
      (joint.operating_mode == null || joint.operating_mode === 0);
    ui.row.classList.toggle('moving', Boolean(joint.moving));
    const actual = state.stale ? 'Last read' : 'Actual';
    ui.position.textContent = actual + ' ' + joint.position +
      ' · target ' + (joint.target ?? '—') +
      ' · commanded ' + (joint.commanded_position ?? '—') +
      ' ticks · ' + (joint.moving ? 'moving' : 'still');
    ui.toggle.textContent = joint.torque_enabled == null ? 'Torque unknown'
      : joint.torque_enabled ? 'Turn motor off' : 'Enable motor';
    ui.toggle.setAttribute('aria-checked', String(joint.torque_enabled === true));
    ui.toggle.disabled = busy || Boolean(state.fault) || state.torque_unknown ||
      typeof joint.torque_enabled !== 'boolean';
    ui.motorLimits.textContent = 'Motor limits ' + joint.minimum + '–' +
      joint.maximum + ' ticks';
    ui.operatingMode.textContent = joint.operating_mode == null ? 'Operating mode —'
      : joint.operating_mode === 0 ? 'Operating mode 0 · position'
      : 'Operating mode ' + joint.operating_mode;
    ui.voltage.textContent = 'Voltage ' + metric(joint.voltage_v, 'V');
    ui.temperature.textContent = 'Temperature ' + metric(joint.temperature_c, '°C');
    ui.torqueLimit.textContent = joint.torque_limit == null ? 'Torque limit —'
      : 'Torque limit ' + joint.torque_limit + '/' + (joint.max_torque_limit ?? '—');
    for (const button of ui.stepButtons) button.disabled = !movable;
    ui.range.disabled = !movable || !visible;
    ui.number.disabled = !movable || !full;
    ui.move.disabled = !movable || !full;
    if (full && visible) {
      ui.range.min = String(visible.lower);
      ui.range.max = String(visible.upper);
      ui.number.min = String(full.lower);
      ui.number.max = String(full.upper);
      ui.targetRange.textContent = joint.torque_enabled === true
        ? 'Live slider ' + visible.lower + '–' + visible.upper +
          ' ticks · full target ' + full.lower + '–' + full.upper
        : 'Click Enable motor · full target ' + full.lower + '–' + full.upper + ' ticks';
      if (!ui.dirty) {
        const value = Number.isInteger(joint.target) ? joint.target : joint.position;
        ui.range.value = String(clamp(value, visible.lower, visible.upper));
        ui.number.value = String(clamp(value, full.lower, full.upper));
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
  clearPendingMotion();
  current = null;
  connection.textContent = 'Connection lost';
  connection.className = 'badge error';
  torque.textContent = 'Torque unknown';
  torque.className = 'badge error';
  modeState.textContent = 'Mode unknown';
  modeState.className = 'badge error';
  motionState.textContent = 'Unknown';
  motionState.className = 'badge error';
  timer.textContent = '';
  showError(error);
  connectionHelp.hidden = false;
  faultPanel.hidden = true;
  armButton.disabled = true;
  freeButton.disabled = true;
  pauseButton.disabled = true;
  stopButton.disabled = true;
  retryButton.disabled = true;
  stepSelect.disabled = true;
  speedSelect.disabled = true;
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
  if (reloadRequired || busy || polling || document.hidden) return;
  polling = true;
  const sequence = postSequence;
  const generation = motionGeneration;
  try {
    const response = await fetch('/api/status', { cache: 'no-store' });
    const state = await response.json();
    if (!response.ok) throw new Error(state.error || 'Could not read arm');
    if (sequence === postSequence && generation === motionGeneration && !busy) {
      render(state);
      if (!state.moving && !pendingTargets.size && !targetInFlight && !activeHoldCancel) {
        motionLease = false;
      }
    }
  } catch (error) {
    if (sequence === postSequence && generation === motionGeneration && !busy) {
      disconnected(error);
    }
  } finally {
    polling = false;
  }
}

async function action(path, payload = {}, options = {}) {
  if (reloadRequired) return;
  if (busy && !options.urgent) return;
  if (options.critical) {
    criticalRequest = true;
    clearPendingMotion();
  }
  const sequence = ++requestSequence;
  postSequence += 1;
  busy = true;
  localError = '';
  if (current) render(current);
  try {
    const state = await post(path, payload);
    if (sequence === requestSequence) {
      busy = false;
      render(state);
    }
  } catch (error) {
    if (sequence === requestSequence) {
      busy = false;
      if (/token/i.test(error.message || '')) {
        requireReload('Control token expired. Reload this page.');
        return;
      }
      showError(error);
      await refresh();
      showError(error);
    }
  } finally {
    if (sequence === requestSequence) {
      criticalRequest = false;
      scheduleTarget();
    }
  }
}

async function pauseMotion() {
  if (pausing || criticalRequest || motionEpoch === null) return;
  pausing = true;
  const epoch = motionEpoch;
  try {
    await action('/api/pause', { motion_epoch: epoch }, { critical: true, urgent: true });
  } finally {
    pausing = false;
  }
}

function shouldPause() {
  return motionLease || pendingTargets.size > 0 || targetInFlight !== null ||
    activeHoldCancel !== null || Boolean(current?.moving);
}

async function heartbeat() {
  if (reloadRequired || heartbeatInFlight || busy || document.hidden || !document.hasFocus() ||
      motionEpoch === null || !shouldPause()) return;
  heartbeatInFlight = true;
  const generation = motionGeneration;
  const epoch = motionEpoch;
  const sequence = postSequence;
  try {
    const state = await post('/api/heartbeat', { motion_epoch: epoch });
    if (generation === motionGeneration) {
      if (state.fault || (Number.isInteger(state.motion_epoch) &&
          state.motion_epoch > motionEpoch)) {
        postSequence += 1;
        render(state);
      } else if (state.motion_epoch === motionEpoch &&
          sequence === postSequence && !busy && current &&
          state.joints?.every((joint) => currentJoint(joint.name)?.target === joint.target)) {
        render(state);
      }
    }
  } catch (error) {
    if (generation === motionGeneration) {
      if (/token/i.test(error.message || '')) {
        requireReload('Control token expired. Reload this page.');
        return;
      }
      clearPendingMotion();
      showError(error);
      refresh();
    }
  } finally {
    heartbeatInFlight = false;
  }
}

async function stopOnFocusLoss() {
  if (focusStopPending || reloadRequired || !current) return;
  if (!current.armed && !current.torque_unknown && !shouldPause() && !busy) return;
  focusStopPending = true;
  try {
    await action('/api/stop', { motion_epoch: motionEpoch },
      { critical: true, urgent: true });
  } finally {
    focusStopPending = false;
  }
}

armButton.addEventListener('click', () =>
  action('/api/mode', { mode: 'position', motion_epoch: motionEpoch }));
freeButton.addEventListener('click', () =>
  action('/api/mode', { mode: 'manual', motion_epoch: motionEpoch },
    { critical: true }));
pauseButton.addEventListener('click', pauseMotion);
stopButton.addEventListener('click', () =>
  action('/api/stop', { motion_epoch: motionEpoch },
    { critical: true, urgent: true }));
retryButton.addEventListener('click', () =>
  action('/api/retry', {}, { critical: true }));
speedSelect.addEventListener('change', () =>
  action('/api/speed', { speed: speedSelect.value }));
rangeView.addEventListener('change', () => {
  if (rangeView.value === 'fine') {
    for (const [name, ui] of rows) ui.fineCenter = currentJoint(name)?.position ?? null;
  }
  if (current) render(current);
});
recenterButton.addEventListener('click', () => {
  for (const [name, ui] of rows) ui.fineCenter = currentJoint(name)?.position ?? null;
  if (current) render(current);
});

refresh();
setInterval(refresh, 1000);
setInterval(heartbeat, 200);
document.addEventListener('visibilitychange', () => {
  if (document.hidden) {
    stopOnFocusLoss();
  } else {
    refresh();
  }
});
window.addEventListener('blur', stopOnFocusLoss);
