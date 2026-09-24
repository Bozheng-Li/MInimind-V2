/* ==========================================================================
   MiniMind 训练控制台
   --------------------------------------------------------------------------
   三栏逻辑：

   ① 选算法（16 个，各带一句「它到底在做什么」）
   ② 选配置文件（阶段配置 + 架构扫描预设），并**从服务端取回该组合真正生效的默认值**
   ③ 覆盖超参 —— 表单里只预填「与默认值不同」的项会被标出来，命令预览实时同步

   关键设计：

   - **默认值不是前端猜的**。`/api/train/defaults` 让训练入口自己的 argparse 解析一遍，
     因此展示的 lr / batch / 数据路径与真实训练完全一致。前端只负责把用户改过的项
     发回去，没改的一律不发 —— 这样配置更新后界面自动跟上，不会「钉死」成旧快照。
   - **预览与执行共用同一个 `build_command`**（后端 jobs.py），预览里看到的命令就是
     真正会跑的命令。
   - **安全**：参数经白名单校验、不经过 shell、独立进程组。前端不做任何「自己拼命令」
     的事，只传键值对。
   ========================================================================== */
'use strict';

(function () {
'use strict';

const { mkEl, escapeHtml, fmtNum, fmtInt, fmtPct, fmtK, fmtDur, fmtBytes,
        token, seriesColor, showTip, hideTip, tipHead, tipRow } = window.Viz;

const ALGO_ORDER = ['pretrain', 'sft', 'lora', 'qlora', 'distill',
  'dpo', 'ipo', 'simpo', 'cpo', 'orpo', 'kto', 'grpo', 'dapo', 'rloo', 'ppo', 'agent'];
const ALGO_COLOR = Object.fromEntries(ALGO_ORDER.map((a, i) => [a, seriesColor(i)]));

const STATUS_META = {
  running:     { label: '运行中', cls: 'acc' },
  ok:          { label: '已完成', cls: 'ok' },
  failed:      { label: '启动失败', cls: 'bad' },
  stopped:     { label: '已停止', cls: 'warn' },
  interrupted: { label: '被中断', cls: 'warn' },
};

const state = {
  catalog: null,
  algo: 'sft',
  config: null,          // 配置文件相对路径；null = 用算法默认阶段配置
  defaults: null,        // 服务端解析出的真实默认值
  overrides: {},         // 用户改动的项（只发这些）
  resources: null,
  jobs: [],
  logs: {},              // job_id -> {offset, text}
  metrics: {},           // job_id -> 从日志增量解析出的实时指标
  openLogs: new Set(),   // 用户手动展开日志的 job_id（重绘后仍保持展开）
  expRuns: null,         // /api/experiments 的 runs 列表（懒加载，供作业卡关联登记）
  previewCmd: '',
  busy: false,
  timers: [],
};

/* ======================================================================== */
/*  表单规格                                                                 */
/* ======================================================================== */
function fieldsFor(fields, algo) {
  return fields.filter((f) => {
    if (!f.algos) return true;                       // 无 algos = 所有算法都可用
    if (f.algos === 'all') return true;
    return Array.isArray(f.algos) && f.algos.includes(algo);
  });
}
function byGroup(fields) {
  const g = new Map();
  for (const f of fields) {
    if (!g.has(f.group)) g.set(f.group, []);
    g.get(f.group).push(f);
  }
  return g;
}

/** 当前值 = 用户覆盖 ≥ 服务端默认 ≥ 空。 */
function valueOf(key) {
  if (key in state.overrides) return state.overrides[key];
  const d = state.defaults && state.defaults.params ? state.defaults.params[key] : undefined;
  return d === undefined ? '' : d;
}
function isChanged(key) {
  return key in state.overrides;
}

/* ======================================================================== */
/*  渲染：算法选择                                                           */
/* ======================================================================== */
function algoPicker() {
  const algos = (state.catalog && state.catalog.algos) || [];
  const box = mkEl('div', 'algo-grid');
  for (const a of algos) {
    if (!ALGO_ORDER.includes(a.key)) continue;
    const b = mkEl('button', 'algo-card');
    b.type = 'button';
    b.setAttribute('aria-pressed', state.algo === a.key ? 'true' : 'false');
    b.innerHTML = `<span class="nm"><span class="dotmark" style="background:${ALGO_COLOR[a.key]}"></span>${escapeHtml(a.label)}</span>`
      + `<span class="st">${escapeHtml(a.stage)} · ${escapeHtml(a.kind)}</span>`
      + `<span class="ds">${escapeHtml(a.desc)}</span>`;
    b.addEventListener('click', () => {
      state.algo = a.key;
      state.config = null;          // 换算法 → 回到该算法的阶段配置
      state.overrides = {};
      state.defaults = null;
      loadDefaults();
      render();
    });
    box.appendChild(b);
  }
  return box;
}

/* ======================================================================== */
/*  渲染：配置选择                                                           */
/* ======================================================================== */
function configPicker() {
  const cfgs = (state.catalog && state.catalog.configs) || {};
  const algos = (state.catalog && state.catalog.algos) || [];
  const info = algos.find((a) => a.key === state.algo) || {};
  const stage = info.stage || '';
  const stageCfg = (cfgs.stage || []).filter((c) => {
    // 阶段配置与算法匹配：SFT 家族用 sft*.yaml，RL 用 rl*.yaml，预训练用 pretrain*.yaml
    const n = c.name;
    if (state.algo === 'pretrain') return n.startsWith('pretrain');
    if (['sft', 'lora', 'qlora', 'distill'].includes(state.algo)) return n.startsWith('sft');
    return n.startsWith('rl');
  });
  const list = mkEl('div', 'cfg-list');

  const auto = mkEl('button', 'cfg-pick');
  auto.type = 'button';
  auto.setAttribute('aria-pressed', state.config === null ? 'true' : 'false');
  auto.innerHTML = '<span class="rad"></span><span class="txt">'
    + `<span class="nm">（默认）按算法自动选择</span>`
    + `<span class="ds">${escapeHtml(stage ? `使用 configs/${stage === '预训练' ? 'pretrain' : stage === '监督微调' ? 'sft' : 'rl'}.yaml` : '阶段配置')}</span>`
    + '<span class="pt">stage config</span></span>';
  auto.addEventListener('click', () => pickConfig(null));
  list.appendChild(auto);

  for (const c of stageCfg) {
    const b = mkEl('button', 'cfg-pick');
    b.type = 'button';
    b.setAttribute('aria-pressed', state.config === c.path ? 'true' : 'false');
    b.innerHTML = '<span class="rad"></span><span class="txt">'
      + `<span class="nm">${escapeHtml(c.name)}</span>`
      + `<span class="ds">${escapeHtml(c.arch.summary)}</span>`
      + `<span class="pt">${escapeHtml(c.path)}</span>`
      + `<span class="slotrow">${Object.entries(c.arch.slots || {}).map(([k, v]) => `<span>${escapeHtml(String(v.type))}</span>`).join('')}</span>`
      + '</span>';
    b.addEventListener('click', () => pickConfig(c.path));
    list.appendChild(b);
  }
  return list;
}
function pickConfig(path) {
  state.config = path;
  state.overrides = {};
  state.defaults = null;
  loadDefaults();
  render();
}

function sweepPicker() {
  const cfgs = (state.catalog && state.catalog.configs) || {};
  const sweep = cfgs.sweep || [];
  if (!sweep.length) return null;
  const sel = mkEl('select');
  const opt0 = mkEl('option', null, '— 不用预设 —');
  opt0.value = '';
  sel.appendChild(opt0);
  for (const c of sweep) {
    const o = mkEl('option', null, c.name);
    o.value = c.path;
    sel.appendChild(o);
  }
  sel.value = sweep.some((c) => c.path === state.config) ? state.config : '';
  sel.addEventListener('change', () => { if (sel.value) pickConfig(sel.value); });
  return sel;
}

/* ======================================================================== */
/*  渲染：参数表单                                                           */
/* ======================================================================== */
function paramForm() {
  const cat = state.catalog;
  if (!cat) return mkEl('div', 'empty-note', '正在读取目录…');
  const fields = fieldsFor(cat.fields || [], state.algo);
  const groups = byGroup(fields);
  const wrap = mkEl('div');

  for (const gname of cat.field_groups || []) {
    const items = groups.get(gname);
    if (!items || !items.length) continue;
    const g = mkEl('div', 'pgroup');
    const ttl = mkEl('div', 'ttl');
    ttl.appendChild(document.createTextNode(gname));
    const changed = items.filter((f) => isChanged(f.key)).length;
    if (changed) ttl.appendChild(mkEl('span', 'hint', `已改 ${changed} 项`));
    g.appendChild(ttl);
    const pg = mkEl('div', 'pgrid');
    for (const f of items) pg.appendChild(fieldEl(f));
    g.appendChild(pg);
    wrap.appendChild(g);
  }
  return wrap;
}

function fieldEl(f) {
  const cur = valueOf(f.key);
  const box = mkEl('div', `field${f.type === 'flag' ? ' switch' : ''}`);
  const changed = isChanged(f.key);

  if (f.type === 'flag') {
    const id = `f-${f.key}`;
    const cb = mkEl('input');
    cb.type = 'checkbox'; cb.id = id;
    cb.checked = cur === true || cur === 1 || cur === '1' || cur === 'true';
    cb.addEventListener('change', () => setVal(f.key, cb.checked));
    const lb = mkEl('label', null, escapeHtml(f.label));
    lb.htmlFor = id;
    box.appendChild(cb); box.appendChild(lb);
    return box;
  }

  const lab = mkEl('label');
  lab.appendChild(document.createTextNode(f.label));
  if (f.help) lab.appendChild(mkEl('span', 'def', ''));
  lab.appendChild(mkEl('span', 'flag', `--${f.key}`));
  box.appendChild(lab);

  let ctl;
  if (f.choices && f.choices.length) {
    ctl = mkEl('select');
    const blank = mkEl('option', null, '（用配置里的值）');
    blank.value = '';
    if (cur === '') blank.selected = true;
    ctl.appendChild(blank);
    for (const c of f.choices) {
      const o = mkEl('option', null, c.label);
      o.value = c.value;
      if (String(cur) === String(c.value)) o.selected = true;
      ctl.appendChild(o);
    }
    ctl.addEventListener('change', () => setVal(f.key, ctl.value === '' ? null : ctl.value));
  } else if (f.type === 'number') {
    ctl = mkEl('input');
    ctl.type = 'number';
    ctl.step = f.step || 'any';
    ctl.value = cur === '' ? '' : String(cur);
    ctl.placeholder = f.phasize || '留空 = 用配置的值';
    ctl.addEventListener('input', () => setVal(f.key, ctl.value === '' ? null : ctl.value));
  } else {
    ctl = mkEl('input');
    ctl.type = 'text';
    ctl.value = cur === '' ? '' : String(cur);
    ctl.placeholder = f.phasize || '留空 = 用配置的值';
    ctl.addEventListener('input', () => setVal(f.key, ctl.value === '' ? null : ctl.value));
  }
  if (changed) ctl.classList.add('changed');
  box.appendChild(ctl);
  if (f.help) box.appendChild(mkEl('div', 'note', escapeHtml(f.help)));
  return box;
}

function setVal(key, v) {
  const def = state.defaults && state.defaults.params ? state.defaults.params[key] : undefined;
  // 与默认值相同就撤掉覆盖，保持「只发差异」的语义
  if (v === null || v === undefined || String(v) === String(def === undefined ? '' : def) || v === '') {
    delete state.overrides[key];
  } else {
    state.overrides[key] = v;
  }
  schedulePreview();
  // 只更新该字段的「已改」标记，避免整表重建导致输入框失焦
  refreshChangedMarks();
}
function refreshChangedMarks() {
  document.querySelectorAll('.field').forEach((el) => {
    const flag = el.querySelector('.flag');
    if (!flag) return;
    const key = flag.textContent.replace(/^--/, '');
    const ctl = el.querySelector('input, select');
    if (!ctl) return;
    ctl.classList.toggle('changed', isChanged(key));
  });
  const badge = document.getElementById('tc-changed');
  if (badge) {
    const n = Object.keys(state.overrides).length;
    badge.textContent = n ? `已覆盖 ${n} 项` : '全部用配置默认值';
    badge.className = `badge ${n ? 'acc' : 'mute'}`;
  }
}

/* ======================================================================== */
/*  命令预览                                                                 */
/* ======================================================================== */
let previewTimer = null;
function schedulePreview() {
  clearTimeout(previewTimer);
  previewTimer = setTimeout(doPreview, 220);
}
function doPreview() {
  const params = { ...state.overrides };
  if (state.config) params.config = state.config;
  return fetch('/api/train/preview', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ algo: state.algo, params }),
  }).then((r) => r.json()).then((j) => {
    state.previewCmd = j.ok ? j.pretty : `# 参数不合法：${j.error}`;
    const body = document.getElementById('preview-body');
    if (body) body.innerHTML = renderCmd(j.ok ? j.cmd : [], j.ok ? '' : j.error);
  }).catch((e) => {
    const body = document.getElementById('preview-body');
    if (body) body.textContent = String(e);
  });
}
/** 命令着色：只把「旗标」和「值」区分开，不做花哨语法高亮。 */
function renderCmd(cmd, error) {
  if (error) return `<span class="ex"># ${escapeHtml(error)}</span>`;
  return cmd.map((c, i) => {
    if (i === 0) return `<span class="ex">${escapeHtml(c)}</span>`;
    return c.startsWith('--') ? `<span class="fl">${escapeHtml(c)}</span>` : `<span class="va">${escapeHtml(c)}</span>`;
  }).join(' ');
}

/* ======================================================================== */
/*  启动 / 监控                                                              */
/* ======================================================================== */
function startJob() {
  const params = { ...state.overrides };
  if (state.config) params.config = state.config;
  state.busy = true;
  renderAside();
  fetch('/api/train/start', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ algo: state.algo, params }),
  }).then((r) => r.json()).then((j) => {
    if (!j.ok) { toast(j.error || '启动失败', true); return; }
    toast(`已启动 ${j.job.id}`);
    state.logs[j.job.id] = { offset: 0, text: '' };
    refreshJobs().then(() => { startJobPoll(j.job.id); });
  }).catch((e) => toast(String(e), true))
    .finally(() => { state.busy = false; renderAside(); });
}

function stopJob(id) {
  if (!confirm(`停止任务 ${id}？会先发 SIGTERM 让当前 step 走完，超时再强杀。`)) return;
  fetch(`/api/train/jobs/${encodeURIComponent(id)}/stop`, { method: 'POST' })
    .then((r) => r.json()).then((j) => {
      if (!j.ok) { toast(j.error || '停止失败', true); return; }
      toast('已发送停止信号');
      refreshJobs();
    }).catch((e) => toast(String(e), true));
}

function refreshJobs() {
  return fetch('/api/train/jobs').then((r) => r.json()).then((j) => {
    state.jobs = j.jobs || [];
    renderAside();
  }).catch(() => {});
}

/** 增量拉取某个任务的日志（记住字节偏移，不重复传已看过的内容）。 */
function pollJobLog(id) {
  const cur = state.logs[id] || { offset: 0, text: '' };
  // 已结束的任务读到文件末尾后就没必要再请求了（轮询 tick 仍会调用这里，
  // 好让首次展开的历史任务能补上一拉）。注意必须带上「任务已结束」这个条件：
  // 刚启动的训练日志文件可能是空的，那时 eof 为真但内容还没开始写。
  const job = state.jobs.find((x) => x.id === id);
  if (cur.eof && job && job.status !== 'running') return Promise.resolve();
  return fetch(`/api/train/jobs/${encodeURIComponent(id)}/log?offset=${cur.offset}&limit=120000`)
    .then((r) => r.json()).then((j) => {
      if (j.text) {
        // 只保留尾部 200 KB —— 日志正文留在磁盘上，浏览器不必全量持有
        let text = cur.text + j.text;
        if (text.length > 200000) text = '…（更早的内容已省略，完整日志见磁盘）\n' + text.slice(-200000);
        state.logs[id] = { offset: j.offset, text, size: j.size, eof: j.eof };
        const el = document.querySelector(`[data-log="${CSS.escape(id)}"]`);
        if (el) {
          const atBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 60;
          el.textContent = text;
          if (atBottom) el.scrollTop = el.scrollHeight;
        }
        // 新到的那一段立刻解析成指标（上面那段 200 KB 截断只影响日志正文，
        // 解析走的是「这次新拿到的 chunk」，所以不会因为截断漏掉或重复计点）
        ingestMetrics(id, j.text);
        // 只就地替换指标块，不整栏重绘 —— 否则每 2.5 秒都会把用户展开的日志、
        // 滚动位置和卡片状态全部重置一遍。
        const slot = document.querySelector(`[data-metrics="${CSS.escape(id)}"]`);
        if (slot) {
          const fresh = metricsBlock(id);
          if (fresh) slot.replaceWith(fresh);
        } else if (state.metrics[id] && state.metrics[id].keys.size) {
          scheduleAside();
        }
      }
    }).catch(() => {});
}

/* ------------------------------------------------------------------ */
/*  日志 → 实时指标                                                      */
/* ------------------------------------------------------------------ */
// 训练 stdout 的每行都由算法自己格式化（`trainer/algos/*.py::format_log`），
// 字段集**每个算法都不一样** —— 这正是「不同算法记不同的 log」最直接的证据。
// 下面用一个通用的 `键: 值` 解析器把它们统一成 `{key: [{x: step, y}]}`：
//
//   Epoch:[1/2](5/113151), loss: 2.0594, logits_loss: 2.0594, aux_loss: 0.0000,
//   lr: 0.00000868, epoch_time: 614.0min | grad_norm: 2.561, tokens/s: 18,851, gpu_mem: 3,715MB
//   Epoch:[1/1](7/500), Reward: 1.2345, KL_ref: 0.0021, Adv Std: 0.43, Actor Loss: ... （GRPO）
//   Epoch:[1/1](7/500), Reward: ..., Approx KL: ..., Critic Loss: ..., Actor LR: ... （PPO）
//
// 零服务端改动：日志本来就通过 `/api/train/jobs/<id>/log` 增量送过来了。
const STEP_RE = /Epoch:\[(\d+)\/(\d+)\]\((\d+)\/(\d+)\)/;
const KV_RE = /([A-Za-z][A-Za-z0-9_/ ]*?):\s*(-?\d[\d,]*(?:\.\d+)?(?:[eE][+-]?\d+)?)/g;

// 只有「同一个量的两种写法」才归一。别的键一律原样保留 —— 例如 Agent 打的 `KL`
// 与 GRPO 的 `KL_ref` 并不保证同源，硬合并成一条曲线就是编造口径。
const KEY_ALIAS = {
  tokens_s: 'tokens_per_sec',
  learning_rate: 'lr',
};
/** 这些键是「行的骨架」不是指标，不画也不列。 */
const SKIP_KEYS = new Set(['epoch', 'step', 'iters', 'epoch_time']);

function normKey(raw) {
  const k = String(raw).trim().toLowerCase().replace(/[\s/]+/g, '_').replace(/_+/g, '_');
  return KEY_ALIAS[k] || k;
}

/** 可画的指标 + 显示名 + 小数位。**按这个顺序取，取到几个画几个（最多 4 个）** ——
 *  不同量纲绝不合并到一张图，每个指标一张自己的小图（单条曲线，所以不需要图例）。 */
const PLOT_SPEC = [
  ['loss', 'loss', 4],
  ['reward', '奖励', 4],
  ['grad_norm', '梯度范数', 3],
  ['lr', '学习率', 8],
  ['tokens_per_sec', '吞吐 tok/s', 0],
  ['gpu_mem', '显存 MB', 0],
  ['critic_loss', 'critic 损失', 4],
  ['approx_kl', 'approx KL', 5],
  ['clipfrac', 'clipfrac', 4],
  ['preference_loss', '偏好损失', 4],
  ['dpo_loss', 'DPO 损失', 4],
  ['margin', '偏好间隔', 4],
  ['pref_acc', '偏好准确率', 4],
  ['policy_loss', '策略损失', 4],
  ['kl_ref', 'KL（相对参考模型）', 4],
  ['avg_response_len', '回复长度', 2],
  ['pass_rate', '任务通过率', 4],
  ['turns_mean', '平均对话轮数', 2],
  ['valid_call_rate', '工具调用合法率', 4],
  ['aux_loss', 'aux_loss', 4],
  ['moe_load_maxmin', 'MoE 负载 max/min', 2],
];
const PLOT_MAX = 4;

function metricsOf(id) {
  let m = state.metrics[id];
  if (!m) m = state.metrics[id] = { pending: '', series: {}, keys: new Set(), last: {}, lastStep: null };
  return m;
}

/** 把新拿到的日志块解析进该任务的指标状态（保留最后一段半截行到下一块）。 */
function ingestMetrics(id, chunk) {
  const st = metricsOf(id);
  const lines = (st.pending + chunk).split('\n');
  st.pending = lines.pop() || '';
  for (const line of lines) ingestLogLine(st, line);
}

function ingestLogLine(st, line) {
  const m = STEP_RE.exec(line);
  if (!m) return;
  const step = Number(m[3]);
  KV_RE.lastIndex = 0;
  let hit;
  while ((hit = KV_RE.exec(line)) !== null) {
    const key = normKey(hit[1]);
    if (!key || SKIP_KEYS.has(key)) continue;
    const val = Number(hit[2].replace(/,/g, ''));
    if (!Number.isFinite(val)) continue;
    st.keys.add(key);
    st.last[key] = val;
    let s = st.series[key];
    if (!s) s = st.series[key] = [];
    const tail = s[s.length - 1];
    // 同一个 step 可能打多行（如 RL 的 rollout 与 update）→ 后者覆盖前者，不重复计点
    if (tail && tail.x === step) tail.y = val;
    else {
      s.push({ x: step, y: val });
      if (s.length > 400) s.shift();      // 只留最近 400 个点：长训练也不撑爆内存
    }
  }
  st.lastStep = step;
}

/** 作业卡里的实时指标块：最多 4 张单曲线小图 + 一行最新值。 */
function metricsBlock(id) {
  const st = state.metrics[id];
  if (!st || !st.keys.size) return null;
  const picked = PLOT_SPEC.filter(([k]) => (st.series[k] || []).length > 1).slice(0, PLOT_MAX);
  if (!picked.length) return null;
  const box = mkEl('div', 'jobmetrics');
  box.dataset.metrics = id;
  const charts = mkEl('div', 'jobcharts');
  // 画图要等节点进 DOM：lineChart 按 host.clientWidth 定 viewBox，此刻还是 0，
  // 会退到兜底宽度、画完再插入就整张图缩在角落（这个块的两条插入路径 —— 整栏重绘
  // 与就地 replaceWith —— 都是「先构造后插入」，所以统一延后一帧再画）。
  const draws = [];
  for (const [key, label, nd] of picked) {
    const host = mkEl('div', 'jobchart');
    host.appendChild(mkEl('div', 'jobchart-t', `${escapeHtml(label)} · 最新 ${fmtNum(st.last[key], nd)}`));
    const h = mkEl('div');
    host.appendChild(h);
    const points = st.series[key].map((p) => ({ x: p.x, y: p.y }));
    // 单条曲线 → 直接标出它的名字，不需要图例（一条线不靠颜色区分身份）
    draws.push(() => window.Viz.lineChart(h, [{ key, name: label, color: token('--accent'), points }],
      { height: 104, area: true, yTickNd: nd, xTickFmt: (v) => fmtK(v) }));
    charts.appendChild(host);
  }
  box.appendChild(charts);
  requestAnimationFrame(() => { for (const d of draws) d(); });

  // 本算法**实际打出来的**指标名 —— 「不同算法记不同 log」在控制台上的正面体现。
  // 注意口径：这里只统计 stdout 的 `键: 值` 行；逐 step 的完整列数（100+）在指标 CSV 里。
  const names = [...st.keys].sort();
  const chips = mkEl('div', 'keychips');
  for (const k of names) {
    const label = (PLOT_SPEC.find(([x]) => x === k) || [])[1] || k;
    chips.appendChild(mkEl('span', 'keychip', `<b>${escapeHtml(label)}</b><span>${escapeHtml(k)}</span>`));
  }
  const det = mkEl('details', 'jobkeys');
  det.appendChild(mkEl('summary', null,
    `本次算法在 stdout 声明的指标 · ${names.length} 个`));
  det.appendChild(chips);
  det.appendChild(mkEl('div', 'tile-foot',
    '口径：日志里的 <code>键: 值</code> 行（各算法由 <code>format_log</code> 自己格式化，字段集互不相同）。'
    + '逐 step 的完整列（100+ 列）落在指标 CSV 里，见「实验台 → 强化学习 / 监督微调」。'));
  box.appendChild(det);
  return box;
}

/** 作业 → 训练登记：用 save_weight / run_name 在登记表里做**字符串精确匹配**，命中才给链接。 */
function runForJob(j) {
  const runs = state.expRuns || [];
  const keys = [j.save_weight, j.run_name, (j.params || {}).run_name, (j.params || {}).save_weight]
    .filter((x) => typeof x === 'string' && x);
  if (!keys.length) return null;
  return runs.find((r) => keys.includes(r.run_name)) || null;
}
function ensureExpRuns() {
  if (state.expRuns !== null || state._runsLoading) return;
  state._runsLoading = true;
  fetch('/api/experiments').then((r) => r.json()).then((j) => {
    state.expRuns = ((j.runs || {}).runs) || [];
    renderAside();
  }).catch(() => { state.expRuns = []; });
}
let jobPollTimer = null;
function startJobPoll(focusId) {
  stopJobPoll();
  const tick = () => {
    refreshJobs();
    const running = state.jobs.filter((j) => j.status === 'running');
    const ids = running.length ? running.map((j) => j.id) : (focusId ? [focusId] : []);
    ids.forEach(pollJobLog);
    fetch('/api/resources').then((r) => r.json()).then((j) => { state.resources = j; renderLive(); }).catch(() => {});
    if (!running.length && state._polledOnce) { /* 保留定时器：可能还会有新任务 */ }
    state._polledOnce = true;
  };
  tick();
  jobPollTimer = setInterval(tick, 2500);
}
function stopJobPoll() { if (jobPollTimer) { clearInterval(jobPollTimer); jobPollTimer = null; } }

/* ======================================================================== */
/*  右侧栏：资源 + 任务                                                      */
/* ======================================================================== */
function renderLive() {
  const el = document.getElementById('tc-live');
  if (!el || !state.resources) return;
  const r = state.resources;
  const g = (r.gpus || [])[0];
  const parts = [];
  if (g && g.util_pct != null) parts.push(`GPU ${g.util_pct.toFixed(0)}%`);
  if (g && g.mem_used_pct != null) parts.push(`显存 ${g.mem_used_pct.toFixed(0)}%`);
  if (r.memory && r.memory.used_pct != null) parts.push(`内存 ${r.memory.used_pct.toFixed(0)}%`);
  const act = (r.jobs || []).length;
  parts.push(act ? `${act} 个训练在跑` : '无训练进程');
  el.innerHTML = `<span class="live-dot"></span>${escapeHtml(parts.join(' · '))}`;
}

function renderAside() {
  const host = document.getElementById('tc-aside');
  if (!host) return;
  host.innerHTML = '';
  host.appendChild(resourcePanel());
  host.appendChild(jobsPanel());
  renderLive();
}

function resourcePanel() {
  const r = state.resources || {};
  const gpus = r.gpus || [];
  const box = mkEl('div', 'lcard');
  const head = mkEl('div', 'lcard-head');
  head.appendChild(mkEl('h3', null, '实时资源'));
  head.appendChild(mkEl('span', 'sub', '5 秒刷新'));
  box.appendChild(head);

  // 选卡：把 GPU 做成可点的设备下拉源
  for (const g of gpus) {
    const row = mkEl('div', 'meter-row');
    const top = mkEl('div', 'meter-row-top');
    top.appendChild(mkEl('span', null, `GPU ${g.index} · ${escapeHtml(short(g.name, 22))}`));
    top.appendChild(mkEl('span', 'v', g.mem_used_mb != null && g.mem_total_mb
      ? `${fmtInt(g.mem_used_mb)}/${fmtInt(g.mem_total_mb)} MB` : '—'));
    row.appendChild(top);
    const track = mkEl('div', 'meter-track');
    const fill = mkEl('div', 'meter-fill');
    const f = g.mem_used_pct != null ? g.mem_used_pct / 100 : 0;
    fill.style.width = `${Math.max(1, Math.min(100, f * 100))}%`;
    fill.style.background = f >= 0.92 ? token('--critical') : f >= 0.75 ? token('--warning') : token('--accent');
    track.appendChild(fill);
    row.appendChild(track);
    const facts = mkEl('div', 'meter-row-top');
    facts.style.marginTop = '3px';
    facts.innerHTML = `<span>利用率 <b style="font-family:var(--mono)">${g.util_pct != null ? g.util_pct.toFixed(0) + '%' : '—'}</b></span>`
      + `<span>温度 <b style="font-family:var(--mono)">${g.temp_c != null ? g.temp_c + '°C' : '—'}</b></span>`
      + `<span>空闲 <b style="font-family:var(--mono)">${g.mem_free_mb != null ? fmtK(g.mem_free_mb) + ' MB' : '—'}</b></span>`;
    facts.style.color = 'var(--ink-3)';
    facts.style.fontSize = '10.5px';
    row.appendChild(facts);
    box.appendChild(row);
  }
  if (!gpus.length) {
    box.appendChild(mkEl('div', 'empty-note', '未检测到 GPU。训练会退化到 CPU（极慢）。'));
  }

  const m = r.memory || {}, cpu = r.cpu || {}, disk = r.disk || {};
  const mw = mkEl('div');
  mw.style.marginTop = '10px';
  if (m.total_gb) mw.appendChild(meterRow('内存', `${m.used_gb}/${m.total_gb} GB`, m.used_pct, memColor(m.used_pct / 100)));
  if (cpu.load_pct != null) mw.appendChild(meterRow('CPU 负载', `${cpu.load1} · ${cpu.cores} 核`, cpu.load_pct, utilColor(cpu.load_pct)));
  if (disk.total_gb) mw.appendChild(meterRow('磁盘空闲', `${disk.free_gb}/${disk.total_gb} GB`, disk.used_pct, memColor(disk.used_pct / 100)));
  if (mw.children.length) box.appendChild(mw);

  const sw = r.software || {};
  if (sw.torch) {
    box.appendChild(mkEl('div', 'tile-foot',
      `torch ${escapeHtml(sw.torch)} · CUDA ${escapeHtml(sw.cuda || '—')} · bf16 ${sw.bf16_supported ? '支持' : '不支持'}`));
  }
  return box;
}
const memColor = (f) => f >= 0.92 ? token('--critical') : f >= 0.75 ? token('--warning') : token('--accent');
const utilColor = (p) => p >= 90 ? token('--critical') : p >= 60 ? token('--warning') : token('--good');
const short = (s, n) => (s && s.length > n ? s.slice(0, n - 1) + '…' : s);
function meterRow(label, text, pct, color) {
  const row = mkEl('div', 'meter-row');
  const top = mkEl('div', 'meter-row-top');
  top.appendChild(mkEl('span', null, escapeHtml(label)));
  top.appendChild(mkEl('span', 'v', escapeHtml(text)));
  row.appendChild(top);
  const track = mkEl('div', 'meter-track');
  const fill = mkEl('div', 'meter-fill');
  fill.style.width = `${Math.max(0, Math.min(100, pct || 0))}%`;
  fill.style.background = color;
  track.appendChild(fill);
  row.appendChild(track);
  return row;
}

function jobsPanel() {
  const box = mkEl('div', 'lcard');
  const head = mkEl('div', 'lcard-head');
  head.appendChild(mkEl('h3', null, '训练任务'));
  head.appendChild(mkEl('span', 'sub', `${state.jobs.length} 个`));
  head.appendChild(mkEl('span', 'spacer'));
  const rf = mkEl('button', 'btn ghost', '刷新');
  rf.style.fontSize = '11px';
  rf.addEventListener('click', () => refreshJobs().then(() => toast('已刷新')));
  head.appendChild(rf);
  box.appendChild(head);

  if (!state.jobs.length) {
    box.appendChild(mkEl('div', 'empty-note', '还没有从控制台启动过任务。<br>启动后的进程与日志会出现在这里。'));
    return box;
  }
  const list = mkEl('div', 'jobs-list');
  for (const j of state.jobs) {
    list.appendChild(jobCard(j));
  }
  box.appendChild(list);
  return box;
}

function jobCard(j) {
  const meta = STATUS_META[j.status] || { label: j.status, cls: 'mute' };
  const card = mkEl('div', 'jobcard');
  const head = mkEl('div', 'jobcard-head');
  head.appendChild(mkEl('span', `badge ${meta.cls}`, `<span class="dot"></span>${escapeHtml(meta.label)}`));
  head.appendChild(mkEl('span', 'id', escapeHtml(j.id)));
  head.appendChild(mkEl('span', 'spacer'));
  if (j.status === 'running') {
    const sb = mkEl('button', 'btn danger', '停止');
    sb.style.cssText = 'font-size:11px;padding:3px 9px';
    sb.addEventListener('click', () => stopJob(j.id));
    head.appendChild(sb);
  }
  // 归档到实验库（训练进程自己写的 run 登记）
  const body = mkEl('div', 'jobcard-body');
  const facts = [
    ['算法', j.algo],
    ['设备', j.device || '默认'],
    ['实验名', j.run_name || '—'],
    ['开始', (j.started_at || '').replace('T', ' ').slice(5, 19)],
  ];
  for (const [k, v] of facts) {
    const row = mkEl('div', 'meter-row-top');
    row.appendChild(mkEl('span', null, k));
    row.appendChild(mkEl('span', 'v', escapeHtml(String(v || '—'))));
    body.appendChild(row);
  }
  // 关联的训练登记：能对上才给链接，对不上不出声（不猜）
  ensureExpRuns();
  const run = runForJob(j);
  if (run) {
    const a = mkEl('a', 'joblink');
    a.href = '/lab#runs';
    a.title = '在实验台的训练登记里展开这一条';
    a.textContent = `训练登记：${run.run_id}（${statusLabel(run.status)}${run.wall_seconds != null ? ` · ${fmtDur(run.wall_seconds)}` : ''}）`;
    body.appendChild(a);
  }
  // 实时指标（从日志增量解析，零服务端改动）——运行中直接可见，已结束的展开后也能看
  const mb = metricsBlock(j.id);
  if (mb) body.appendChild(mb);

  const cmdt = mkEl('div', 'cmdbox');
  cmdt.style.cssText = 'font-size:10.5px;max-height:80px';
  cmdt.textContent = j.cmd_pretty || '';
  body.appendChild(cmdt);
  card.appendChild(head);
  card.appendChild(body);

  const log = mkEl('pre', 'joblog');
  log.dataset.log = j.id;
  log.textContent = (state.logs[j.id] || {}).text || '';
  card.appendChild(log);
  // 展开/收起状态记在 state 里，而不是靠 DOM 上读 —— 右侧栏每 2.5 秒整体重绘一次，
  // 若把状态放在 style.display 上，用户展开的历史任务会在下次轮询时自动收起。
  const wantOpen = state.openLogs.has(j.id) || j.status === 'running';
  log.style.display = wantOpen ? 'block' : 'none';
  // 展开时才拉日志（列表里几十个任务全拉会浪费）。已结束的任务日志不再增长，
  // 但首次展开仍要拉一次 —— 这样才能把历史日志解析成指标曲线。
  card.addEventListener('click', (ev) => {
    if (ev.target.closest('button') || ev.target.closest('a')) return;
    if (state.openLogs.has(j.id)) state.openLogs.delete(j.id);
    else state.openLogs.add(j.id);
    const open = state.openLogs.has(j.id);
    log.style.display = open ? 'block' : 'none';
    if (open) { log.scrollTop = log.scrollHeight; pollJobLog(j.id).then(renderAside); }
  });
  if (wantOpen) {
    requestAnimationFrame(() => { log.scrollTop = log.scrollHeight; });
    // 这里**不发请求**：拉日志由轮询 tick 与点击事件负责。卡片渲染过程中再发请求
    // 会引出「渲染 → 请求 → 渲染」的递归链（下面 scheduleAside 也只能兜住一层）。
  }
  return card;
}
function statusLabel(s) { return (STATUS_META[s] || {}).label || s || '未知'; }

/** 合并重绘：一帧内多次调用只重绘一次，避免递归渲染。 */
let asideQueued = false;
function scheduleAside() {
  if (asideQueued) return;
  asideQueued = true;
  setTimeout(() => { asideQueued = false; renderAside(); }, 0);
}

/* ======================================================================== */
/*  主渲染                                                                   */
/* ======================================================================== */
function render() {
  const main = document.getElementById('tc-main');
  if (!main) return;
  main.innerHTML = '';

  if (!state.catalog) {
    main.appendChild(mkEl('div', 'empty-note', '正在读取配置目录与设备信息…'));
    return;
  }

  // ① 算法
  main.appendChild(section('1 · 选择训练算法', '16 个算法覆盖「预训练 → 对齐 → 强化」全流程；每个算法的日志列与产物都不同',
    algoPicker()));

  // ② 配置
  const c2 = mkEl('div');
  c2.appendChild(configPicker());
  const sp = sweepPicker();
  if (sp) {
    const row = mkEl('div');
    row.style.cssText = 'display:flex;align-items:center;gap:10px;margin-top:12px';
    row.appendChild(mkEl('span', 'tile-foot', '或从架构扫描预设里选：'));
    row.appendChild(sp);
    c2.appendChild(row);
  }
  main.appendChild(section('2 · 选择模型配置', '配置只提供默认值，下面的每一项都可以覆盖；命令行永远优先', c2));

  // ③ 参数
  const c3 = mkEl('div');
  const badge = mkEl('span', 'badge mute', '全部用配置默认值');
  badge.id = 'tc-changed';
  const reset = mkEl('button', 'btn ghost', '恢复全部默认');
  reset.style.fontSize = '11px';
  reset.addEventListener('click', () => {
    state.overrides = {};
    render();
    toast('已恢复为配置默认值');
  });
  const headRow = mkEl('div');
  headRow.style.cssText = 'display:flex;align-items:center;gap:9px;margin-bottom:12px';
  headRow.appendChild(badge);
  headRow.appendChild(reset);
  c3.appendChild(headRow);
  c3.appendChild(paramForm());
  main.appendChild(section('3 · 调整超参与训练策略',
    state.defaults
      ? `表单里的值来自 <code>${escapeHtml(state.defaults.config || `configs/${state.defaults.algo}.yaml`)}</code> —— 由训练入口自己解析，与应用到真实训练的一致`
      : '正在解析该组合的默认值…', c3));

  // ④ 预览 + 启动
  main.appendChild(section('4 · 确认命令并启动',
    '预览与实际执行共用同一个拼命令函数，因此这里看到的就是真正会跑的', previewPanel()));

  refreshChangedMarks();
  renderAside();
}

function section(title, sub, body) {
  const el = mkEl('div', 'lcard');
  const head = mkEl('div', 'lcard-head');
  head.appendChild(mkEl('h3', null, escapeHtml(title)));
  if (sub) head.appendChild(mkEl('span', 'sub', sub));
  el.appendChild(head);
  el.appendChild(body);
  el.style.marginBottom = '16px';
  return el;
}

function previewPanel() {
  const wrap = mkEl('div');
  const pv = mkEl('div', 'preview');
  const head = mkEl('div', 'preview-head');
  head.appendChild(mkEl('span', null, '将执行的命令'));
  head.appendChild(mkEl('span', 'spacer'));
  const copy = mkEl('button', 'mini', '复制');
  copy.addEventListener('click', () => {
    navigator.clipboard?.writeText(state.previewCmd).then(() => toast('已复制命令'), () => toast('复制失败', true));
  });
  head.appendChild(copy);
  pv.appendChild(head);
  const body = mkEl('pre', 'preview-body');
  body.id = 'preview-body';
  body.textContent = '正在生成…';
  pv.appendChild(body);
  wrap.appendChild(pv);

  const row = mkEl('div');
  row.style.cssText = 'display:flex;gap:10px;align-items:center;margin-top:12px';
  const run = mkEl('button', 'btn primary');
  run.innerHTML = '<svg viewBox="0 0 24 24" width="14" height="14"><path d="M8 5l11 7-11 7V5z" fill="currentColor"/></svg> 启动训练';
  run.addEventListener('click', () => {
    if (state.busy) return;
    const running = state.jobs.filter((j) => j.status === 'running');
    const g = (state.resources && state.resources.gpus) || [];
    const busyGpu = g.some((x) => (x.mem_used_pct || 0) > 60);
    if (running.length && !confirm(`当前还有 ${running.length} 个训练在跑，同时训练会争抢显存。仍然启动？`)) return;
    else if (!running.length && busyGpu && !confirm('检测到 GPU 显存占用较高（可能是别的进程在用）。仍然启动？')) return;
    startJob();
  });
  row.appendChild(run);
  row.appendChild(mkEl('span', 'tile-foot',
    '启动后进程与日志会出现在右侧；训练进程会自己往「实验台 → 训练登记」写一份结构化记录'));
  wrap.appendChild(row);

  schedulePreview();
  return wrap;
}

/* ======================================================================== */
/*  加载                                                                     */
/* ======================================================================== */
function loadDefaults() {
  const q = new URLSearchParams({ algo: state.algo });
  if (state.config) q.set('config', state.config);
  return fetch(`/api/train/defaults?${q}`).then((r) => r.json()).then((j) => {
    if (!j.ok) { toast(j.error || '解析默认值失败', true); return; }
    state.defaults = j;
    // 没给 save_weight 时用配置里的，方便识别产物
    render();
  }).catch((e) => toast(String(e), true));
}

function toast(msg, bad) {
  let wrap = document.querySelector('.toast-wrap');
  if (!wrap) { wrap = mkEl('div', 'toast-wrap'); document.body.appendChild(wrap); }
  const t = mkEl('div', `toast${bad ? ' bad' : ' ok'}`, escapeHtml(msg));
  wrap.appendChild(t);
  setTimeout(() => { t.style.opacity = '0'; setTimeout(() => t.remove(), 260); }, 3000);
}

function initTheme() {
  let t = 'light';
  try { t = localStorage.getItem('mm-theme') || (matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'); } catch (e) { /* ignore */ }
  document.documentElement.dataset.theme = t;
}

function boot() {
  initTheme();
  document.getElementById('tc-theme').addEventListener('click', () => {
    const next = document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark';
    document.documentElement.dataset.theme = next;
    try { localStorage.setItem('mm-theme', next); } catch (e) { /* ignore */ }
    render();          // 图表颜色从 CSS 变量读，主题变了要重画
  });
  Promise.all([
    fetch('/api/catalog').then((r) => r.json()),
    fetch('/api/resources').then((r) => r.json()).catch(() => ({})),
    fetch('/api/train/jobs').then((r) => r.json()).catch(() => ({ jobs: [] })),
  ]).then(([cat, res, jobs]) => {
    state.catalog = cat;
    state.resources = res;
    state.jobs = jobs.jobs || [];
    render();
    loadDefaults();
    startJobPoll(null);
  }).catch((e) => {
    document.getElementById('tc-main').innerHTML = '';
    document.getElementById('tc-main').appendChild(mkEl('pre', 'cmdbox err', String(e)));
  });
}
document.addEventListener('DOMContentLoaded', boot);
})();
