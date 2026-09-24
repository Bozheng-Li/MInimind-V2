/* ==========================================================================
   MiniMind 实验台 —— 页面渲染
   --------------------------------------------------------------------------
   设计原则（与 dataviz 规范一致）：

   - **一种图型只干一件事**，同一页刻意混用不同图型：折线看趋势、横向条看量值、
     发散条看「相对基线的增减」、热力图看矩阵、环图看构成、散点看权衡、
     时间轴看「谁在什么时候跑了多久」、管线图看流程。**不把什么数据都画成折线**。
   - **颜色跟着实体走**（算法/架构/模型分到固定色位），不随排名或筛选而重排。
   - **≥2 条序列必有图例**；≤4 条再叠直接标注，身份永不只靠颜色。
   - **绝不双 Y 轴**：量纲不同就拆两张图或分面。
   - 每张图旁边（或下方）都有可读的**表视图**兜底，颜色只是加速阅读、不是唯一信息。
   ========================================================================== */
'use strict';

(function () {
'use strict';

const { mkEl, escapeHtml, fmtNum, fmtInt, fmtSigned, fmtPct, fmtK, fmtDur, fmtBytes,
        token, seriesColor, seqColor, divColor, showTip, hideTip, tipHead, tipRow } = window.Viz;

/* --------------------------- 实体 → 颜色（固定） --------------------------- */
// 算法：固定色位。顺序写死，绝不按出现的先后重新分配。
const ALGO_ORDER = ['dpo', 'ipo', 'simpo', 'cpo', 'orpo', 'kto',
  'grpo', 'dapo', 'rloo', 'ppo', 'agent'];
const ALGO_SLOT = Object.fromEntries(ALGO_ORDER.map((a, i) => [a, i]));
const algoColor = (a) => seriesColor(ALGO_SLOT[a] ?? 7);
// 架构：Dense / MoE 两个色位（在只比架构的图里用）
const ARCH_COLOR = { dense: seriesColor(0), moe: seriesColor(1) };
// 评测阶段 → 色位
const STAGE_SLOT = { '预训练': 0, 'SFT': 1, 'DPO': 2, 'GRPO': 3, 'PPO': 4, 'Agentic': 5, '其它': 6 };
const stageColor = (s) => seriesColor(STAGE_SLOT[s] ?? 6);

const STATUS_META = {
  ok:          { label: '成功',   cls: 'ok' },
  failed:      { label: '失败',   cls: 'bad' },
  running:     { label: '运行中', cls: 'acc' },
  interrupted: { label: '被中断', cls: 'warn' },
  stopped:     { label: '已停止', cls: 'warn' },
  unknown:     { label: '未收尾', cls: 'mute' },
};

// 数据行数：精确数出来的直接显示；按体积估算的加「≈」前缀，不冒充精确值。
// 后端对 >256MB 的语料改用「字节 / 110」估算（见 catalog._count_rows）。
function rowsText(rows, exact) {
  if (rows === null || rows === undefined) return '—';
  return `${exact === false ? '≈' : ''}${fmtInt(rows)}`;
}

const state = {
  data: null, catalog: null, resources: null, models: null,
  page: 'library',
  sort: {},            // 表排序：{tableId: {key, dir}}
  rlArch: 'dense',     // RL 页当前分面
  logPath: null, logOffset: 0, logText: '',
  timers: [],
};

/* ======================================================================== */
/*  通用小件                                                                 */
/* ======================================================================== */
function card(title, sub, body, headExtra) {
  const el = mkEl('div', 'lcard');
  if (title) {
    const head = mkEl('div', 'lcard-head');
    head.appendChild(mkEl('h3', null, escapeHtml(title)));
    if (sub) head.appendChild(mkEl('span', 'sub', sub));
    head.appendChild(mkEl('span', 'spacer'));
    if (headExtra) head.appendChild(headExtra);
    el.appendChild(head);
  }
  if (body) el.appendChild(body);
  return el;
}
function stack(children) { const d = mkEl('div', 'stack'); children.filter(Boolean).forEach((c) => d.appendChild(c)); return d; }
function grid(cls, children) { const d = mkEl('div', `grid ${cls}`); children.filter(Boolean).forEach((c) => d.appendChild(c)); return d; }
function subTitle(text) { return mkEl('div', 'section-title', escapeHtml(text)); }

function tile(label, value, opts = {}) {
  const t = mkEl('div', 'tile');
  t.appendChild(mkEl('div', 'tile-label', escapeHtml(label)));
  t.appendChild(mkEl('div', 'tile-value',
    `${escapeHtml(String(value))}${opts.unit ? `<span class="unit">${escapeHtml(opts.unit)}</span>` : ''}`));
  if (opts.foot) t.appendChild(mkEl('div', 'tile-foot', opts.foot));
  if (opts.spark && opts.spark.length > 1) {
    // 宽度写死（不读 clientWidth）：此刻宿主还没进 DOM，clientWidth 恒为 0，
    // 读它会退化成默认值 —— 写死反而让牌子里的小图尺寸稳定、彼此对齐。
    const h = mkEl('div', 'tile-spark');
    t.appendChild(h);
    window.Viz.sparkline(h, opts.spark, { width: 104, height: 22, color: opts.color });
  }
  return t;
}
function tiles(items) { const d = mkEl('div', 'tiles'); items.forEach((x) => d.appendChild(x)); return d; }

function badge(text, cls = 'mute') { return mkEl('span', `badge ${cls}`, escapeHtml(text)); }
function statusBadge(status) {
  const m = STATUS_META[status] || { label: status || '未知', cls: 'mute' };
  return mkEl('span', `badge ${m.cls}`, `<span class="dot"></span>${escapeHtml(m.label)}`);
}

function kv(pairs) {
  const d = mkEl('div', 'kv');
  for (const [k, v, wide] of pairs) {
    if (v === undefined || v === null || v === '') continue;
    const row = mkEl('div', 'kv-row');
    row.appendChild(mkEl('span', 'kv-k', escapeHtml(k)));
    row.appendChild(mkEl('span', `kv-v${wide ? ' wide' : ''}`, escapeHtml(String(v))));
    d.appendChild(row);
  }
  return d;
}

/** 图表宿主：宽度变化时重建（SVG 用 viewBox，但刻度密度依赖像素宽度）。 */
function chartHost(build, height) {
  const h = mkEl('div');
  if (height) h.style.minHeight = `${height}px`;
  const draw = () => { build(h); };
  draw();
  pendingCharts.push(draw);
  return h;
}
let pendingCharts = [];
let _resizeTimer = null;
function onResize() {
  clearTimeout(_resizeTimer);
  _resizeTimer = setTimeout(() => { pendingCharts.forEach((f) => f()); }, 160);
}
window.addEventListener('resize', onResize);

function legendHost(series, hidden, onToggle) {
  const h = mkEl('div');
  h.style.cssText = 'display:flex;flex-wrap:wrap;gap:6px;margin-top:10px';
  window.Viz.legend(h, series, hidden, onToggle);
  return h;
}

function notice(text, kind = '') {
  return mkEl('div', `notice ${kind}`, `<span class="ic">ℹ</span><span>${text}</span>`);
}

/** 分组小标题：用在 kv 明细里把「优化动力学 / MoE / 系统」等分组分开。 */
function kvGroup(text) { return mkEl('div', 'kv-group', escapeHtml(text)); }

/**
 * 「系列名 → 曲线」的通用折线卡。
 *
 * 数据层已经把「没有数据 / 恒为水平线」的列滤掉了，所以这里拿到几条就画几条 ——
 * 一条都没有就返回 null，由调用方 `.filter(Boolean)` 丢掉，不会留下一张空卡片。
 *
 * colSeries: `{列名: {step:[], value:[]}}`（后端 ``_curves`` 的输出格式）
 * spec:      `{列名: {label, fmt, color?, dash?}}` —— 只画 spec 里出现的列
 */
function curveCard(title, note, colSeries, spec, opts = {}) {
  const keys = Object.keys(spec).filter((k) => colSeries && colSeries[k] && colSeries[k].value.length);
  if (!keys.length) return null;
  const series = keys.map((k, i) => {
    const sp = spec[k];
    return {
      key: k, name: sp.label, shortName: sp.short || sp.label, color: sp.color || seriesColor(i),
      dashed: sp.dash, points: (colSeries[k].value || []).map((y, j) => ({ x: colSeries[k].step[j], y })),
    };
  });
  const hidden = new Set();
  const wrap = mkEl('div');
  wrap.appendChild(chartHost((h) => window.Viz.lineChart(h, series.filter((s) => !hidden.has(s.key)), {
    height: opts.height || 180, yTickNd: opts.yTickNd === undefined ? 4 : opts.yTickNd,
    yTickFmt: opts.yTickFmt, yLabel: opts.yLabel, xTickFmt: (v) => fmtK(v),
    zeroLine: opts.zeroLine, area: series.length === 1, markY: opts.markY,
    markYLabel: opts.markYLabel, endLabels: series.length <= 4,
  }), opts.height || 180));
  if (series.length > 1) {
    wrap.appendChild(legendHost(series, hidden, (k) => { hidden.has(k) ? hidden.delete(k) : hidden.add(k); rerender(); }));
  }
  return card(title, note, wrap);
}

/**
 * 逐层 / 逐专家矩阵 → 热力图卡。
 * 矩阵的行是系列自身（后端按行归一化成 0~1 只用于配色），浮层里读的是原值。
 */
function matrixCard(title, note, m, opts = {}) {
  if (!m || !m.labels || !m.labels.length) return null;
  const rows = m.labels.map((lbl) => labelOfSeries(lbl, opts.prefix));
  const cols = m.step.map((s) => fmtK(s));
  return card(title, note, chartHost((h) => window.Viz.heatmap(h, rows, cols, m.values, {
    raw: m.cols, mode: opts.mode || 'seq', rowH: 17, labelW: opts.labelW || 62,
    colW: opts.colW || 17, colLabelH: 34, legend: opts.legend !== false,
    rawFmt: opts.rawFmt || ((v) => fmtNum(v, opts.nd === undefined ? 3 : opts.nd)),
    cellFmt: (v) => v.toFixed(2),
    valueLabel: opts.valueLabel || '值',
    hint: opts.hint,
  }), opts.height || 240));
}
/** `hidden_rms_L3` → `L3`，`moe_load_e7` → `e7`；认不出来的列名原样保留。 */
function labelOfSeries(col, prefix) {
  if (prefix !== undefined) return col.replace(prefix, '');
  const i = col.indexOf('_L');
  if (i >= 0 && /^\d+$/.test(col.slice(i + 2))) return col.slice(i + 1);
  const j = col.indexOf('_e');
  if (j >= 0 && /^\d+$/.test(col.slice(j + 2))) return col.slice(j + 1);
  return col;
}

/* 表格：支持排序。cols=[{key,label,num,width,fmt,cls}] */
function table(id, cols, rows, opts = {}) {
  const s = state.sort[id];
  let data = rows;
  if (s) {
    const col = cols.find((c) => c.key === s.key);
    data = [...rows].sort((a, b) => {
      const x = a[s.key], y = b[s.key];
      const nx = x === null || x === undefined, ny = y === null || y === undefined;
      if (nx && ny) return 0;
      if (nx) return 1;                    // 空值永远排在最后（不是当成 0）
      if (ny) return -1;
      const r = typeof x === 'number' && typeof y === 'number' ? x - y : String(x).localeCompare(String(y));
      return s.dir === 'asc' ? r : -r;
    });
  }
  const wrap = mkEl('div', 'tw');
  const t = mkEl('table', `dtable${opts.sortable === false ? '' : ' sortable'}`);
  const thead = mkEl('thead');
  const tr = mkEl('tr');
  for (const c of cols) {
    const th = mkEl('th', null, escapeHtml(c.label));
    if (c.width) th.style.width = c.width;
    if (c.num) th.style.textAlign = 'right';
    if (opts.sortable !== false) {
      th.dataset.sort = c.key;
      if (s && s.key === c.key) th.setAttribute('aria-sort', s.dir === 'asc' ? 'ascending' : 'descending');
      th.addEventListener('click', () => {
        const cur = state.sort[id];
        state.sort[id] = { key: c.key, dir: cur && cur.key === c.key && cur.dir === 'asc' ? 'desc' : 'asc' };
        rerender();
      });
    }
    tr.appendChild(th);
  }
  thead.appendChild(tr);
  t.appendChild(thead);
  const tb = mkEl('tbody');
  for (const r of data) {
    const row = mkEl('tr');
    for (const c of cols) {
      const td = mkEl('td', `${c.num ? 'num ' : ''}${c.cls || ''}`);
      const v = r[c.key];
      // 列的三条出路，顺序不能颠倒：
      //   1) cell  —— 要拼子元素（徽标、色标）
      //   2) fmt   —— 显示值由整行推出来，`v` 本身可能是 undefined（例如「末 loss」读
      //      r.summary.loss_last，「参数量」读 r.arch.params_total）。所以 fmt 必须在
      //      「空值」判断**之前**拿到控制权，否则这些列会永远显示「—」。
      //   3) 原值  —— 直接渲染，空值才落灰写「—」。
      if (c.cell) {
        c.cell(td, v, r);
      } else if (typeof c.fmt === 'function') {
        const txt = c.fmt(v, r);
        const empty = txt === null || txt === undefined || txt === '' || txt === '—';
        td.textContent = empty ? '—' : txt;
        if (empty) td.classList.add('dim');
      } else if (v === null || v === undefined || v === '') {
        td.textContent = '—';
        td.classList.add('dim');
      } else {
        td.textContent = String(v);
      }
      row.appendChild(td);
    }
    tb.appendChild(row);
  }
  t.appendChild(tb);
  wrap.appendChild(t);
  return wrap;
}

/* ======================================================================== */
/*  页面：实验库                                                             */
/* ======================================================================== */
function renderLibrary(d) {
  const out = [];
  out.push(notice(
    `实验数据根目录 <code>${escapeHtml(d.meta.artifact_root)}</code> · 生成于 ${escapeHtml(d.meta.generated_at)}。`
    + '本页只读扫描，不写入任何文件；点击左侧分区可下钻到每个训练阶段。', ''));

  const evalModels = (d.eval && d.eval.models) || [];
  const runs = (d.runs && d.runs.runs) || [];
  const rlRuns = (d.rl && d.rl.runs) || [];
  const sweep = (d.sweep && d.sweep.configs) || [];

  // ---- Hero：一句话结论（一屏只有一个主数字）----
  const withScore = evalModels.filter((m) => m[Object.keys(d.eval.random || {})[0]] != null);
  const best = [...evalModels].sort((a, b) => avgScore(b, d.eval.tasks) - avgScore(a, d.eval.tasks))[0];
  const hero = mkEl('div', 'hero');
  if (best) {
    const bm = mkEl('div', 'hero-main');
    bm.appendChild(mkEl('div', 'hero-num',
      `${(avgScore(best, d.eval.tasks) * 100).toFixed(1)}<span class="unit">% 平均得分</span>`));
    bm.appendChild(mkEl('div', 'hero-cap',
      `全流程里综合表现最好的权重是 <b>${escapeHtml(best.model)}</b>（${escapeHtml(best.stage)} 阶段）。`
      + `综合分 = ${(d.eval.tasks || []).length} 个任务的算术平均，仅用于横向排序，不等于任何单一能力。`));
    hero.appendChild(bm);
  } else {
    hero.appendChild(mkEl('div', 'hero-main', '<div class="hero-num">—</div>'
      + '<div class="hero-cap">还没有评测结果（<code>test/storage/report/eval/summary.csv</code> 为空）。</div>'));
  }

  // Hero 右侧：评测集构成环图（构成用环，不用折线）
  const byStage = {};
  for (const m of evalModels) byStage[m.stage] = (byStage[m.stage] || 0) + 1;
  const donutWrap = mkEl('div', 'viz-donut-wrap');
  const slices = Object.entries(byStage).map(([k, v]) => ({ label: k, value: v, color: stageColor(k) }));
  window.Viz.donut(donutWrap, slices, {
    size: 132, center: { value: String(evalModels.length), label: '个评测权重' }, valueLabel: '模型数',
  });
  const donutBox = mkEl('div', 'hero-split');
  donutBox.appendChild(donutWrap);
  donutBox.appendChild(legendHost(slices.map((s, i) => ({ key: s.label, name: s.label, color: s.color })), null, () => {}));
  hero.appendChild(donutBox);
  out.push(hero);

  // ---- 统计牌 ----
  const okCount = runs.filter((r) => r.status === 'ok').length;
  const failCount = runs.filter((r) => r.status === 'failed').length;
  out.push(tiles([
    tile('训练登记', String((d.runs && d.runs.total) || 0), { unit: '次 run', foot: `${okCount} 成功 · ${failCount} 失败` }),
    tile('强化学习 run', String(rlRuns.length), { unit: '组', foot: '4 算法 × 2 架构' }),
    tile('架构配置', String(sweep.length), { unit: '组', foot: `最优 ${escapeHtml(sweep[0] ? sweep[0].name : '—')}` }),
    tile('评测权重', String(evalModels.length), { unit: '个', foot: `${(d.eval.tasks || []).length} 个任务` }),
    tile('原始日志', String((d.logs || []).length), { unit: '个文件', foot: '可逐行浏览' }),
  ]));

  // ---- 阶段最佳：横向条（量值比较用条，不用折线）----
  const taskKey = (d.eval.tasks || [])[0] ? d.eval.tasks[0].key : 'ceval';
  const bestByStage = {};
  for (const m of evalModels) {
    const v = m[taskKey];
    if (v == null) continue;
    if (!bestByStage[m.stage] || bestByStage[m.stage].v < v) bestByStage[m.stage] = { v, m };
  }
  const barItems = Object.entries(bestByStage)
    .sort((a, b) => b[1].v - a[1].v)
    .map(([st, o]) => ({
      label: `${st} · ${o.m.model}`,
      value: o.v,
      color: stageColor(st),
      text: fmtPct(o.v),
      sub: `${taskKey.toUpperCase()} · 随机 ${fmtPct((d.eval.random || {})[taskKey] ?? 0)}`,
    }));
  if (barItems.length) {
    out.push(card(
      `各阶段最强模型的 ${taskKey.toUpperCase()} 得分`,
      '每个阶段取该阶段内该项得分最高的权重；虚线是随机猜的水平',
      chartHost((h) => window.Viz.barChart(h, barItems, {
        labelW: 190, rowH: 20, minRef: (d.eval.random || {})[taskKey] ?? 0,
        tickFmt: (v) => `${(v * 100).toFixed(0)}%`, valueLabel: '准确率',
      }), 120),
    ));
  }

  // ---- 训练时间轴：每个 RL run 跑了多久（时间轴看「何时/多久」，折线做不了）----
  const timed = rlRuns.filter((r) => r.wall_hours != null);
  if (timed.length) {
    out.push(card('RL 各算法的墙钟耗时', '按状态着色；斜纹 = 未正常结束',
      timeline(timed.map((r) => ({
        name: `${r.algo_label} · ${r.arch}`,
        hours: r.wall_hours,
        color: algoColor(r.algo),
        status: r.status,
        detail: `${r.arch} · ${r.status}`,
      })))));
  }

  // ---- MoE 健康度横比：跨 run 一眼看出哪次训练最健康 ----
  // 三张同量纲的横向条并排（负载 CV / 路由熵 / 死专家），每张的条形颜色按「架构」
  // 而不是按排名 —— 颜色跟着实体走，排序变了也不会重新上色。
  const moeRows = [];
  const pushMoe = (name, arch, obj) => {
    if (!obj) return;
    const cv = obj.moe_load_cv_tail, ent = obj.moe_entropy_tail, dead = obj.dead_experts_max;
    if (cv == null && ent == null && dead == null) return;
    moeRows.push({ name, arch, cv, ent, dead });
  };
  const pm = (d.pretrain || {}).moe || {};
  pushMoe('预训练 MoE', 'moe', {
    moe_load_cv_tail: pm.moe_load_cv_last, moe_entropy_tail: pm.entropy_min, dead_experts_max: pm.dead_experts_max,
  });
  for (const r of (d.sft || {}).runs || []) pushMoe(`SFT · ${r.label}`, r.label.includes('MoE') ? 'moe' : 'dense', r);
  for (const r of rlRuns) pushMoe(`${r.algo_label} · ${r.arch}`, r.arch, r);

  if (moeRows.length) {
    // 三张条的量纲各不相同，所以各自一张图（绝不双轴）。每张**按「越健康越在上」
    // 排序**：CV 与死专家数越小越好，路由熵越大越好 —— 三张图的第一行因此都是
    // 「这次最健康」，读者不用为每张图反转一次直觉。
    const bars = [
      ['负载变异系数 load_cv', (x) => x.cv, (v) => fmtNum(v, 4), 1,
       '越小越均衡；0 = 每个专家接到的 token 完全一样多'],
      ['死专家数（整段窗口最大）', (x) => x.dead, (v) => fmtNum(v, 0), 1,
       '几乎收不到 token 的专家数量；> 0 说明有专家被饿死'],
      ['路由熵（归一化）', (x) => x.ent, (v) => fmtNum(v, 4), -1,
       '1 = 路由完全均匀，趋 0 = 塌成少数专家'],
    ].map(([title, get, fmt, dir, note]) => {
      const items = moeRows.filter((x) => get(x) != null)
        .sort((a, b) => dir * (get(a) - get(b)))
        .map((x) => ({ label: x.name, value: get(x), text: fmt(get(x)),
                       color: token('--accent'), sub: note }));
      if (!items.length) return null;
      return card(title, note, chartHost((h) => window.Viz.barChart(h, items, {
        labelW: 176, rowH: 17, valueW: 64, tickFmt: (v) => fmtNum(v, 3), valueLabel: title,
      }), 60 + items.length * 23));
    }).filter(Boolean);
    if (bars.length) {
      out.push(subTitle('MoE 健康度横比 · 哪一次训练的路由最健康'));
      out.push(grid('c2', bars));
      out.push(notice('只有 <b>MoE 架构</b>的 run 有这几列（Dense 天然缺失，不列入横比）；'
        + '三个量纲各一张图，每张都把「最健康的那次」排在最上面。'
        + '数值取尾部均值 —— 个别 step 的抖动不该被读成结论。'
        + '逐 step 的完整演化见「预训练」与「强化学习」两页。', ''));
    }
  }

  // ---- 最近 run 表 ----
  if (runs.length) {
    const recent = runs.slice(0, 12);
    out.push(card('最近训练登记', '完整登记见「训练登记」页',
      table('lib-runs', [
        { key: 'started_at', label: '开始', width: '108px', fmt: (v) => shortTime(v) },
        { key: 'algo', label: '算法', cell: (td, v) => { td.appendChild(badge(v || '?', 'acc')); } },
        { key: 'run_name', label: '实验名', cls: 'name' },
        { key: 'arch', label: '架构', fmt: (_v, r) => (r.arch && r.arch.summary) || '—', cls: 'dim' },
        { key: 'status', label: '状态', cell: (td, v) => { td.appendChild(statusBadge(v)); } },
        { key: 'wall', label: '耗时', num: true, fmt: (_v, r) => fmtDur(r.wall_seconds) },
        { key: 'loss', label: '末 loss', num: true, fmt: (_v, r) => fmtNum((r.summary || {}).loss_tail ?? (r.summary || {}).loss_last, 4) },
      ], recent, { })));
  }
  return stack(out);
}
function avgScore(m, tasks) {
  const ks = (tasks || []).map((t) => t.key);
  const vs = ks.map((k) => m[k]).filter((v) => v != null);
  return vs.length ? vs.reduce((a, b) => a + b, 0) / vs.length : -1;
}
function shortTime(iso) {
  if (!iso) return '—';
  return String(iso).replace('T', ' ').slice(5, 16);
}

/* 时间轴：一行一条，跨度按小时。 */
function timeline(items) {
  const max = Math.max(...items.map((i) => i.hours), 0.001);
  const host = mkEl('div');
  const tl = mkEl('div', 'tl');
  for (const it of items) {
    const row = mkEl('div', 'tl-row');
    row.appendChild(mkEl('div', 'tl-name', escapeHtml(it.name)));
    const track = mkEl('div', 'tl-track');
    const span = mkEl('div', `tl-span${it.status && it.status !== 'OK' && it.status !== 'ok' ? ' fail' : ''}`);
    span.style.left = '0%';
    span.style.width = `${Math.max(1.2, (it.hours / max) * 100)}%`;
    span.style.background = it.color || token('--accent');
    span.addEventListener('mousemove', (ev) => showTip(
      tipHead(it.name) + tipRow(it.color || token('--accent'), '耗时', fmtDur(it.hours * 3600))
      + (it.status ? tipRow(null, '状态', it.status) : '')
      + (it.detail ? `<div class="viz-row"><span class="viz-k">${escapeHtml(it.detail)}</span></div>` : ''), ev));
    span.addEventListener('mouseleave', hideTip);
    track.appendChild(span);
    row.appendChild(track);
    row.appendChild(mkEl('div', 'tl-val', fmtDur(it.hours * 3600)));
    tl.appendChild(row);
  }
  host.appendChild(tl);
  host.appendChild(mkEl('div', 'tl-axis',
    `<span>0</span><span>${escapeHtml(fmtDur(max * 3600 / 2))}</span><span>${escapeHtml(fmtDur(max * 3600))}</span>`));
  return host;
}

/* ======================================================================== */
/*  页面：全流程                                                             */
/* ======================================================================== */
/**
 * 某个算法在管线图节点上的悬停明细：奖励 + 该算法**独有的健康度指标**。
 *
 * 每个算法「该看什么」并不一样（DPO 看偏好准确率、PPO 看 clipfrac、Agent 看工具
 * 调用合法率），所以这一列不是同一个字段换个名字，而是真的按算法给不同口径 ——
 * 这正是「不同算法记录不同 log」在管线图上的落点。
 */
function rlNodeDetail(list, extraLabel) {
  return (list || []).flatMap((r) => {
    const rows = [[r.arch, `奖励 ${fmtNum(r.reward_last)}`]];
    const extra = {
      '偏好准确率': r.preference_acc_tail, 'EOS 正收尾率': r.eos_rate_tail,
      'clipfrac': r.clipfrac_tail, '工具调用合法率': r.valid_call_rate_tail,
    }[extraLabel];
    if (extra != null) rows.push([`${r.arch} ${extraLabel}`, fmtNum(extra, 4)]);
    if (r.gpu_mem_peak_mb != null) rows.push([`${r.arch} 峰值显存`, `${fmtK(r.gpu_mem_peak_mb)} MB`]);
    return rows;
  });
}

function renderPipeline(d) {
  const pr = d.pretrain || {}, sft = d.sft || {}, rl = d.rl || {}, ev = d.eval || {};
  const rlBy = {};
  for (const r of (rl.runs || [])) {
    if (!rlBy[r.algo]) rlBy[r.algo] = [];
    rlBy[r.algo].push(r);
  }
  const n = (k) => (rlBy[k] || []).filter((r) => r.status === 'OK' || r.steps).length;
  const st = (k) => (n(k) ? 'done' : 'todo');

  const nodes = [
    { id: 'tok', label: '分词器', sub: 'vocab 6400', state: 'done', group: '数据', badge: 'BPE',
      detail: [['产物', 'tokenizer.json'], ['说明', '中英双语 BPE，词表 6400']] },
    { id: 'pre', label: '预训练', sub: `${(pr.dense && pr.dense.steps_total) || 0} 步 Dense`, state: (pr.dense && pr.dense.steps_total) ? 'done' : 'todo',
      group: '数据', badge: pr.moe && pr.moe.steps_total ? `MoE ${pr.moe.steps_total} 步` : '',
      detail: [['Dense 末 loss', pr.dense ? fmtNum(pr.dense.loss_last) : '—'],
               ['MoE 末 loss', pr.moe ? fmtNum(pr.moe.loss_last) : '—'],
               ['MoE 负载 CV', pr.moe ? fmtNum(pr.moe.moe_load_cv_last, 4) : '—'],
               ['MoE 死专家', pr.moe && pr.moe.dead_experts_max != null ? `${fmtNum(pr.moe.dead_experts_max, 0)} 个` : '—'],
               ['MoE 吞吐', pr.moe && pr.moe.tokens_per_sec ? `${fmtK(pr.moe.tokens_per_sec)} tok/s` : '—'],
               ['MoE 峰值显存', pr.moe && pr.moe.gpu_mem_peak_mb ? `${fmtK(pr.moe.gpu_mem_peak_mb)} MB` : '—']] },
    { id: 'sft', label: '监督微调', sub: `${(sft.runs || []).length} 组实验`, state: (sft.runs || []).length ? 'done' : 'todo',
      group: '对齐', badge: 'full_sft',
      // 三组 run 的关键数字全放进来：吞吐/峰值显存是「同预算下选哪个架构」的直接依据
      detail: (sft.runs || []).flatMap((r) => [
        [r.label, `val ${fmtNum(r.val_loss_last)}`],
        [`${r.label} 吞吐`, r.tokens_per_sec ? `${fmtK(r.tokens_per_sec)} tok/s` : '—'],
        [`${r.label} 峰值显存`, r.gpu_mem_peak_mb ? `${fmtK(r.gpu_mem_peak_mb)} MB` : '—'],
      ]) },
    { id: 'lora', label: 'LoRA / 蒸馏', sub: '低成本垂域 / 分布对齐', state: 'partial', group: '对齐', badge: '2 种',
      detail: [['LoRA', '冻结主体，只训低秩分支'], ['蒸馏', 'CE + KL 双损失']] },
    { id: 'dpo', label: 'DPO', sub: '离线偏好优化', state: st('dpo'), group: '强化', badge: `${n('dpo')} run`,
      detail: rlNodeDetail(rlBy.dpo, '偏好准确率') },
    { id: 'grpo', label: 'GRPO / CISPO', sub: '组内相对优势', state: st('grpo'), group: '强化', badge: `${n('grpo')} run`,
      detail: rlNodeDetail(rlBy.grpo, 'EOS 正收尾率') },
    { id: 'ppo', label: 'PPO', sub: 'Actor-Critic + GAE', state: st('ppo'), group: '强化', badge: `${n('ppo')} run`,
      detail: rlNodeDetail(rlBy.ppo, 'clipfrac') },
    { id: 'agent', label: 'Agentic RL', sub: '多轮工具调用', state: st('agent'), group: '强化', badge: `${n('agent')} run`,
      detail: rlNodeDetail(rlBy.agent, '工具调用合法率') },
    { id: 'eval', label: '标准评测', sub: `${(ev.models || []).length} 个权重`, state: (ev.models || []).length ? 'done' : 'todo',
      group: '交付', badge: `${(ev.tasks || []).length} 任务`,
      detail: [['任务', (ev.tasks || []).map((t) => `${t.label}${t.group ? `（${t.group}）` : ''}`).join(' / ')]] },
    { id: 'serve', label: '部署服务', sub: 'FastAPI + SSE', state: 'done', group: '交付', badge: 'OpenAI 兼容',
      detail: [['接口', '/v1/chat/completions'], ['界面', '对话页 + 训练控制台']] },
  ];

  const out = [];
  out.push(notice('从分词器到部署的完整链路。节点上的徽标是该项实际跑出的产物数量，'
    + '<b>灰色 = 尚未产出</b>，<b>橙色 = 已实现但本次实验未产出</b>。悬停节点看关键数字。', ''));
  out.push(card('全流程管线', '按「数据 → 对齐 → 强化 → 交付」分组',
    chartHost((h) => window.Viz.pipeline(h, nodes, { colW: 128, gap: 24, rowH: 76 }), 300)));

  // 每阶段的产物清单（表视图：颜色之外的准确信息）
  const rows = [
    { stage: '预训练', data: 'pretrain', artifact: 'pretrain.log / pretrain_moe_metrics.csv', n: `${(pr.dense && pr.dense.steps_total) || 0} + ${(pr.moe && pr.moe.steps_total) || 0}`, note: 'Dense 走 stdout 正则，MoE 走 139 列 CSV' },
    { stage: '监督微调', data: 'sft', artifact: 'sft_{dense,moe,moe_full}_metrics.csv', n: String((sft.runs || []).length), note: '含留出集 val_loss' },
    { stage: '强化学习', data: 'rl', artifact: 'rl/<algo>_<arch>_metrics.csv', n: String((rl.runs || []).length), note: '每算法一套专属列' },
    { stage: '架构对比', data: 'sweep', artifact: 'storage/report/summary.csv', n: String(((d.sweep || {}).configs || []).length), note: '18 组配置统一口径' },
    { stage: '标准评测', data: 'eval', artifact: 'storage/report/eval/{summary.csv,tasks.json}', n: String((ev.models || []).length), note: `${(ev.tasks || []).length} 任务 × ${String((ev.models || []).length)} 权重` },
    { stage: '训练登记', data: 'runs', artifact: 'log/runs/<run_id>/meta.json', n: String(((d.runs || {}).total) || 0), note: '本次新增：机器可读的超参与策略' },
  ];
  out.push(card('各阶段产物清单', '这些是「实验台」其余页面各自的数据来源',
    table('pipe-art', [
      { key: 'stage', label: '阶段', cls: 'name' },
      { key: 'data', label: '分区', fmt: (v) => `d.${v}` },
      { key: 'artifact', label: '落盘文件', cls: 'dim' },
      { key: 'n', label: '记录数', num: true },
      { key: 'note', label: '口径备注', cls: 'dim' },
    ], rows)));
  return stack(out);
}

/* ======================================================================== */
/*  页面：训练登记                                                           */
/* ======================================================================== */
function renderRuns(d) {
  const runs = (d.runs && d.runs.runs) || [];
  const out = [];
  if (!runs.length) {
    out.push(card('还没有训练登记', '当用 <code>trainer/train.py</code> 或训练控制台起一次训练后，'
      + '会自动在 <code>test/log/runs/&lt;run_id&gt;/meta.json</code> 落一份机器可读的登记。',
      mkEl('div', 'empty-note', '登记内容包括：实际生效的超参、训练策略、架构槽位、数据规模、环境与 git 版本、以及结果摘要。')));
    return stack(out);
  }
  const counts = (d.runs && d.runs.counts) || {};

  out.push(notice('每一次训练都会落一份 <code>meta.json</code>：记录的是 <b>argparse 解析后真正生效的值</b>'
    + '（命令行 > YAML），因此界面上的超参就是训练循环拿到的那一份。点任意一行展开完整登记。', ''));

  const tileRow = [tile('登记总数', String(d.runs.total), { unit: '次' })];
  for (const [k, label] of [['ok', '成功'], ['failed', '失败'], ['running', '运行中'], ['interrupted', '被中断']]) {
    if (counts[k]) tileRow.push(tile(label, String(counts[k]), { unit: '次' }));
  }
  out.push(tiles(tileRow));

  // 各算法的 run 数：横向条
  const byAlgo = {};
  for (const r of runs) byAlgo[r.algo] = (byAlgo[r.algo] || 0) + 1;
  const algoItems = Object.entries(byAlgo).sort((a, b) => b[1] - a[1])
    .map(([a, c]) => ({ label: a, value: c, text: String(c), color: token('--accent'), sub: '次 run' }));
  const timed = runs.filter((r) => r.wall_seconds != null);
  const side = [];
  side.push(card('各算法训练次数', null, chartHost((h) => window.Viz.barChart(h, algoItems,
    { labelW: 78, rowH: 18, valueW: 40, tickFmt: (v) => String(v), valueLabel: 'run 数' }))));
  if (timed.length) {
    side.push(card('墙钟耗时', '每行一次 run，按算法着色',
      timeline(timed.slice(0, 14).map((r) => ({
        name: `${shortTime(r.started_at)} · ${r.algo}`,
        hours: r.wall_seconds / 3600, color: algoColor(r.algo), status: r.status,
        detail: r.run_name,
      })))));
  }
  out.push(grid('c2', side));

  // 主表：可展开详情
  const host = mkEl('div');
  const cols = [
    { key: 'started_at', label: '开始时间', width: '118px', fmt: (v) => shortTime(v) },
    { key: 'algo', label: '算法', cell: (td, v) => td.appendChild(badge(v, 'acc')) },
    { key: 'run_name', label: '实验名', cls: 'name' },
    { key: 'arch', label: '架构', cls: 'dim', fmt: (_v, r) => (r.arch && (r.arch.summary || '')) || '—' },
    { key: 'params', label: '参数量', num: true, fmt: (_v, r) => {
      const a = r.arch || {};
      if (!a.params_total) return '—';
      const tot = fmtK(a.params_total);
      return a.params_activated ? `${tot} / 激活 ${fmtK(a.params_activated)}` : tot;
    } },
    { key: 'data_rows', label: '数据', num: true, fmt: (_v, r) => rowsText((r.data || {}).rows, (r.data || {}).rows_exact) },
    { key: 'lr', label: 'lr', num: true, fmt: (_v, r) => sci((r.strategy || {}).learning_rate) },
    { key: 'status', label: '状态', cell: (td, v) => td.appendChild(statusBadge(v)) },
    { key: 'wall_seconds', label: '耗时', num: true, fmt: (v) => fmtDur(v) },
    { key: 'loss', label: '末 loss', num: true, fmt: (_v, r) => fmtNum((r.summary || {}).loss_last, 4) },
    // 「奖励」只在 RL 的 run 上有值（走 reward_tail）。SFT/预训练没有这一项，
    // 整列恒显「—」的话就是一条死列 —— 不如整列省掉，少一格噪声。
    ...(runs.some((r) => (r.summary || {}).reward_tail != null)
      ? [{ key: 'reward', label: '奖励', num: true, fmt: (_v, r) => fmtNum((r.summary || {}).reward_tail, 4) }]
      : []),
  ];
  const tw = table('runs-main', cols, runs);
  host.appendChild(tw);

  // 展开：把该 run 的完整登记渲染在表下方（避免表格里塞大块内容）
  const detail = mkEl('div');
  detail.style.marginTop = '14px';
  host.appendChild(detail);
  tw.querySelectorAll('tbody tr').forEach((tr, i) => {
    tr.style.cursor = 'pointer';
    tr.addEventListener('click', () => {
      const open = tr.getAttribute('aria-expanded') === 'true';
      tw.querySelectorAll('tbody tr').forEach((x) => x.removeAttribute('aria-expanded'));
      detail.innerHTML = '';
      if (open) return;
      tr.setAttribute('aria-expanded', 'true');
      detail.appendChild(runDetail(runs[i]));
      detail.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
    });
  });
  out.push(card('训练登记明细', '点击任意一行展开；同一张表里按列排序',
    host));
  return stack(out);
}
function sci(v) {
  if (v === null || v === undefined) return '—';
  return Math.abs(v) < 0.001 ? Number(v).toExponential(1).replace('e-', 'e−') : String(v);
}
function runDetail(r) {
  const box = mkEl('div', 'run-detail');
  const a = r.arch || {}, s = r.strategy || {}, e = r.env || {}, g = r.git || {};
  const sum = r.summary || {};
  const art = r.artifacts || {};

  // 「结果摘要」放第一格：点开一次登记最想知道的是「这次跑成什么样」。
  // 这些字段由训练侧 summarize_run() 写进 meta.json，此前前端一个都没渲染。
  // 按「优化 / MoE / 算法专属 / 系统」分组 —— 与 #pretrain、#rl 的卡片分组同口径，
  // 这样从曲线页跳到登记页时找的是同一批数字。
  const c0 = mkEl('div');
  c0.appendChild(mkEl('h4', null, '结果摘要'));
  const sub0 = mkEl('div');
  // 分组只在**本组至少有一个真实值**时出现 —— 判断必须看原值，不能看格式化后的
  // 字符串（fmtNum(null) 返回的是字符串 '—'，用它判断会让空分组照样渲染出标题）。
  const group = (title, pairs) => {
    const live = pairs.filter((p) => p[1] !== null && p[1] !== undefined && p[1] !== '');
    if (!live.length) return;
    sub0.appendChild(kvGroup(title));
    sub0.appendChild(kv(live));
  };
  group('优化', [
    ['步数', sum.steps != null ? fmtInt(sum.steps) : null],
    ['末 loss', sum.loss_last != null ? fmtNum(sum.loss_last) : null],
    ['尾部 loss', sum.loss_tail != null ? fmtNum(sum.loss_tail) : null],
    ['最低 loss', sum.loss_min != null ? fmtNum(sum.loss_min) : null],
    ['末 val_loss', sum.val_loss_last != null ? fmtNum(sum.val_loss_last) : null],
    ['最低 val_loss', sum.val_loss_min != null ? fmtNum(sum.val_loss_min) : null],
    ['最优 val 步', sum.best_val_step != null ? fmtK(sum.best_val_step) : null],
    ['最大梯度范数', sum.grad_norm_max != null ? fmtNum(sum.grad_norm_max, 3) : null],
  ]);
  group('MoE 健康度', [['MoE 负载 CV（尾部）', sum.moe_load_cv_tail != null ? fmtNum(sum.moe_load_cv_tail) : null]]);
  group('算法专属', [
    ['首奖励', sum.reward_first != null ? fmtNum(sum.reward_first) : null],
    ['尾部奖励', sum.reward_tail != null ? fmtNum(sum.reward_tail) : null],
    ['KL（尾部）', sum.kl_ref_tail != null ? fmtNum(sum.kl_ref_tail) : null],
    ['通过率（尾部）', sum.pass_rate_tail != null ? fmtPct(sum.pass_rate_tail) : null],
    ['偏好准确率（尾部）', sum.preference_acc_tail != null ? fmtPct(sum.preference_acc_tail) : null],
  ]);
  group('系统与产物', [
    ['吞吐', sum.tokens_per_sec != null ? `${fmtK(sum.tokens_per_sec)} tok/s` : null],
    ['峰值显存', sum.gpu_mem_peak_mb != null ? `${fmtK(sum.gpu_mem_peak_mb)} MB` : null],
    ['墙钟', r.wall_seconds != null ? fmtDur(r.wall_seconds) : null],
    ['权重', art.weight, true], ['指标 CSV', art.metrics_csv, true],
  ]);
  c0.appendChild(sub0);
  box.appendChild(c0);

  const c1 = mkEl('div');
  c1.appendChild(mkEl('h4', null, '训练策略（本次实际生效）'));
  c1.appendChild(kv(Object.entries(s).map(([k, v]) => [k, v])));
  box.appendChild(c1);

  const c2 = mkEl('div');
  c2.appendChild(mkEl('h4', null, '架构槽位'));
  c2.appendChild(kv([
    ['架构摘要', a.summary || '—', true],
    ['层数', a.n_layers], ['hidden', a.hidden_size], ['vocab', a.vocab_size],
    ['注意力', a.attention], ['前馈', a.feedforward],
    ['归一化', a.norm], ['位置编码', a.positional_encoding],
    ['路由专家', a.n_experts],
    ['总参数', a.params_total ? fmtInt(a.params_total) : '—'],
    ['激活参数', a.params_activated ? fmtInt(a.params_activated) : '—'],
  ]));
  box.appendChild(c2);

  const c3 = mkEl('div');
  c3.appendChild(mkEl('h4', null, '数据与产物'));
  c3.appendChild(kv([
    ['数据', (r.data || {}).path || '—', true],
    ['数据行数', rowsText((r.data || {}).rows, (r.data || {}).rows_exact)],
    ['max_seq_len', (r.data || {}).max_seq_len],
    ['权重', art.weight, true],
    ['指标 CSV', art.metrics_csv, true],
    ['断点目录', art.checkpoints || art.checkpoint, true],
  ]));
  box.appendChild(c3);

  const c4 = mkEl('div');
  c4.appendChild(mkEl('h4', null, '环境'));
  c4.appendChild(kv([
    ['主机', e.hostname], ['平台', e.platform, true],
    ['Python', e.python], ['torch', e.torch], ['CUDA', e.cuda],
    ['GPU', (e.gpus || []).map((x) => `${x.name} ${x.total_mem_mb}MB`).join(' / ') || '—', true],
    ['git', g.commit ? `${g.commit}${g.dirty ? ' (dirty)' : ''}` : '—', true],
  ]));
  box.appendChild(c4);

  if (r.error || (r.notes || []).length) {
    const c5 = mkEl('div');
    c5.style.gridColumn = '1 / -1';
    c5.appendChild(mkEl('h4', null, r.error ? '失败原因' : '备注'));
    const pre = mkEl('pre', `cmdbox${r.error ? ' err' : ''}`);
    pre.textContent = r.error || (r.notes || []).join('\n');
    c5.appendChild(pre);
    box.appendChild(c5);
  }
  return box;
}

/* ======================================================================== */
/*  页面：预训练                                                             */
/* ======================================================================== */
function renderPretrain(d) {
  const dense = (d.pretrain && d.pretrain.dense) || {};
  const moe = (d.pretrain && d.pretrain.moe) || {};
  const out = [];
  if (!dense.loss && !moe.loss) {
    out.push(card('没有预训练记录', '需要 <code>test/log/pretrain.log</code> 或 '
      + '<code>test/log/pretrain_moe_metrics.csv</code>。', mkEl('div', 'empty-note', '—')));
    return stack(out);
  }
  out.push(notice('Dense 走 stdout 日志（每 100 步一行），MoE 走 139 列逐 50 步指标 CSV。'
    + '两者实验口径不同，因此曲线只做「各自收敛」的观察，不做等预算结论。', ''));

  out.push(tiles([
    tile('Dense 末 loss', fmtNum(dense.loss_last), { unit: '', foot: `首 ${fmtNum(dense.loss_first)} · tail100 ${fmtNum(dense.loss_tail100)}` }),
    tile('MoE 末 loss', fmtNum(moe.loss_last), { unit: '', foot: `首 ${fmtNum(moe.loss_first)} · tail200 ${fmtNum(moe.loss_tail100)}` }),
    tile('MoE 吞吐', moe.tokens_per_sec ? fmtK(moe.tokens_per_sec) : '—', { unit: 'tok/s', foot: 'tail200 均值' }),
    tile('MoE 峰值显存', moe.gpu_mem_peak_mb ? fmtK(moe.gpu_mem_peak_mb) : '—', {
      unit: 'MB', foot: `nvidia-smi 采样窗口最大值；窗口均值 ${moe.gpu_mem_mb ? fmtK(moe.gpu_mem_mb) : '—'} MB`,
    }),
    // 这一格原来的 foot 填的是 moe_maxmin_last（负载 max/min），label 却写「梯度范数」，
    // 标签与数据不符。梯度范数走 moe.grad_norm，负载比单独另起一格。
    tile('MoE 梯度范数', fmtNum(moe.grad_norm, 3), { unit: '', foot: 'tail200 均值（裁剪前）' }),
    tile('MoE 负载 max/min', fmtNum(moe.moe_maxmin_last, 3), {
      unit: '', foot: `首 ${fmtNum(moe.moe_maxmin_first, 2)} · 1.0 = 完全均衡`,
    }),
  ]));

  // 训练/验证 loss：两条曲线同轴（同一量纲），图例 + 端点直标
  const lossSeries = [];
  if (dense.loss && dense.loss.value && dense.loss.value.length) {
    lossSeries.push({ key: 'dense', name: 'Dense 训练 loss', shortName: 'Dense', color: ARCH_COLOR.dense,
                      points: dencePoints(dense.loss) });
  }
  if (moe.loss && moe.loss.value && moe.loss.value.length) {
    lossSeries.push({ key: 'moe', name: 'MoE 训练 loss', shortName: 'MoE', color: ARCH_COLOR.moe,
                      points: dencePoints(moe.loss) });
  }
  const hidden = new Set();
  const wrap = mkEl('div');
  const host = chartHost((h) => window.Viz.lineChart(h, lossSeries.filter((s) => !hidden.has(s.key)), {
    height: 250, xLabel: 'step', yLabel: 'loss', yTickNd: 2, xTickFmt: (v) => fmtK(v), unit: '',
  }), 250);
  wrap.appendChild(host);
  wrap.appendChild(legendHost(lossSeries, hidden, (k) => { hidden.has(k) ? hidden.delete(k) : hidden.add(k); rerender(); }));
  out.push(card('训练 loss', 'Dense 与 MoE 各一条；点图例可单独查看', wrap));

  if (moe.loss && moe.loss.value && moe.loss.value.length) {
    // MoE 健康度：四个量纲不同 → 四个小倍数，绝不挤进一张双轴图
    const panels = [
      ['负载变异系数 load_cv', moe.moe_load_cv, '越小越均衡；0 = 每个专家接的 token 完全一样多', (v) => fmtNum(v, 3), seriesColor(0)],
      ['max/min 负载比', moe.moe_load_maxmin, '最忙专家 / 最闲专家；1.0 = 完全均衡', (v) => fmtNum(v, 2), seriesColor(1)],
      ['死专家数', moe.moe_dead_experts, '整段统计窗口内几乎没收过 token 的专家数', (v) => fmtNum(v, 1), seriesColor(6)],
      ['路由熵（归一化）', moe.moe_entropy_norm, '1 = 路由完全均匀，趋 0 = 塌成少数专家', (v) => fmtNum(v, 3), seriesColor(3)],
    ];
    const boxes = panels.map(([title, curve, note, fmt, color], i) => {
      if (!curve || !curve.value || !curve.value.length) return null;
      return card(title, note, chartHost((h) => window.Viz.lineChart(h,
        [{ key: 'x', name: title, color, points: dencePoints(curve) }],
        { height: 150, area: true, yTickFmt: fmt, xTickFmt: (v) => fmtK(v), endLabels: false, yTickNd: 4 }), 150));
    });
    out.push(subTitle('MoE 健康度 · 四个独立量纲分面'));
    out.push(grid('c2', boxes));
  }

  // 共享专家占比：相对基线的增减 → 发散条
  if (moe.shared_share_first != null && moe.shared_share_last != null) {
    const rows = [
      { label: '起始', value: moe.shared_share_first, text: fmtPct(moe.shared_share_first) },
      { label: '结束（tail200）', value: moe.shared_share_last, text: fmtPct(moe.shared_share_last) },
    ];
    out.push(card('共享专家承接的 token 占比', '共享专家不吃路由，占比升高说明路由专家在「卸载」',
      chartHost((h) => window.Viz.barChart(h, rows, {
        labelW: 108, rowH: 22, tickFmt: (v) => `${(v * 100).toFixed(0)}%`, valueLabel: '占比',
      }), 90)));
  }

  if (moe.loss && (!moe.loss.value || !moe.loss.value.length) && dense.loss) {
    out.push(notice('MoE 只有摘要、没有逐点曲线数据。', 'warn'));
  }

  // ---- 优化动力学：哪一组参数还在学、哪一组已经停滞 ----
  // update_ratio 的口径是「该组学习率 / 该组权重的逐元素 rms」（见 trainer/common/metrics.py
  // 的 snapshot()，那里明确警告过不能写成 lr·‖g‖/‖w‖ —— 后者差 5 个数量级）。
  // 四组参数共用同一个量纲，可以同轴比较。
  const optCard = curveCard('优化动力学 · 各组参数的相对更新量', 'update_ratio = lr / 权重逐元素 RMS；四条同量纲可直接比较。'
    + '持续掉到 1e-5 以下的组等于停学',
  moe.opt_curves, {
    update_ratio_attn: { label: '注意力', short: 'attn', color: seriesColor(0) },
    update_ratio_ffn: { label: '前馈', short: 'ffn', color: seriesColor(1) },
    update_ratio_embed: { label: '词嵌入', short: 'embed', color: seriesColor(2) },
    update_ratio_norm: { label: '归一化', short: 'norm', color: seriesColor(3) },
  }, { height: 200, yTickNd: 6 });
  const gradCard = curveCard('梯度范数 · 按参数组', '在梯度裁剪**之前**记录，因此保留真实的尖峰；'
    + '某个组长期高一个量级 = 该组主导了更新方向',
  moe.grad_curves, {
    grad_norm_attn: { label: '注意力', short: 'attn', color: seriesColor(0) },
    grad_norm_ffn: { label: '前馈', short: 'ffn', color: seriesColor(1) },
    grad_norm_embed: { label: '词嵌入', short: 'embed', color: seriesColor(2) },
    grad_norm_norm: { label: '归一化', short: 'norm', color: seriesColor(3) },
    grad_norm_sum_groups: { label: '分组之和', short: 'sum', color: token('--ink-3'), dash: true },
  }, { height: 200, yTickNd: 3 });
  const wCard = curveCard('权重范数 · 按参数组', '与 update_ratio 合看：范数在涨但更新量在掉，说明这一组已经进入平台期',
    moe.weight_curves, {
      weight_norm_attn: { label: '注意力', color: seriesColor(0) },
      weight_norm_ffn: { label: '前馈', color: seriesColor(1) },
      weight_norm_embed: { label: '词嵌入', color: seriesColor(2) },
    }, { height: 180, yTickNd: 2 });
  if (optCard || gradCard || wCard) {
    out.push(subTitle('优化动力学 · 按参数组'));
    out.push(grid('c2', [optCard, gradCard, wCard]));
  }

  // ---- 门控健康：gate_sat_* 上升说明门控在饱和（路由退化成硬开关）----
  const gateCard = curveCard('注意力门控健康度', 'gate_mean 偏离 0.5 = 门控有偏；gate_sat_lo/hi 是落在'
    + '饱和区的比例，持续上升说明门控退化成硬开关',
  moe.gate_curves, {
    gate_mean: { label: '门控均值', short: 'mean', color: seriesColor(0) },
    gate_std: { label: '门控标准差', short: 'std', color: seriesColor(1) },
    gate_sat_lo: { label: '低饱和比例', short: 'sat_lo', color: seriesColor(6) },
    gate_sat_hi: { label: '高饱和比例', short: 'sat_hi', color: seriesColor(3) },
  }, { height: 200, yTickNd: 3, markY: 0.5, markYLabel: '门控均值中性点 0.5' });
  const biasCard = curveCard('MoE 路由偏置', '负载均衡 bias 的分布；bias 越分散说明均衡器在持续做功',
    moe.bias_curves, {
      moe_bias_std: { label: 'bias 标准差', color: seriesColor(0) },
      moe_bias_absmean: { label: 'bias 平均绝对值', color: seriesColor(1) },
    }, { height: 180, yTickNd: 4 });
  if (gateCard || biasCard) {
    out.push(subTitle('门控与路由偏置 · 健康度'));
    out.push(grid('c2', [gateCard, biasCard]));
  }

  // ---- 逐层激活量：层 × step 的矩阵（8 层逐条折线会糊成一团，用热力图）----
  const layerCards = [
    matrixCard('残差流逐层 RMS', '每一行的颜色按该行**自身**量程归一化，所以看的是「这一层有没有在放大」，'
      + '浮层里读的是原值。列是训练 step',
      moe.layer_hidden, { labelW: 56, colW: 16, height: 210, nd: 2, valueLabel: 'hidden_rms' }),
    matrixCard('注意力 Q 逐层 RMS', '与残差流合看：某一层突然跳起来通常是该层开始主导表征',
      moe.layer_q, { labelW: 56, colW: 16, height: 210, nd: 2, valueLabel: 'q_rms' }),
    matrixCard('注意力输出逐层 RMS', null, moe.layer_out, { labelW: 56, colW: 16, height: 210, nd: 2, valueLabel: 'out_rms' }),
    matrixCard('门控离散度逐层', null, moe.layer_gate_std, { labelW: 56, colW: 16, height: 210, nd: 3, valueLabel: 'gate_std' }),
  ].filter(Boolean);
  if (layerCards.length) {
    out.push(subTitle('逐层激活量 · 深度 × 时间'));
    out.push(grid('c2', layerCards));
  }

  // ---- 16 专家逐个负载：比 load_cv 一条线更能指出「谁被饿死」 ----
  const exCard = matrixCard('16 个路由专家的负载（MoE 最有诊断价值的一张图）',
    '每一行是一个专家，列是训练 step。整行长期偏白 = 这个专家几乎没被路由到（被饿死）；'
    + '反过来某一行长期最深 = 负载塌到了它身上',
    moe.expert_load, { labelW: 34, colW: 16, height: 330, nd: 4, valueLabel: '专家负载占比' });
  if (exCard) out.push(exCard);

  // ---- 系统侧：采样线程记录的真实利用率 ----
  const sysCard = curveCard('系统侧采样', '由 trainer/common/metrics.py 的 GpuSampler 后台线程轮询，'
    + '与训练步不是一一对应（按时间采样）',
  moe.sys_curves, {
    gpu_util_mean: { label: 'GPU 利用率均值', color: seriesColor(0) },
    gpu_util_min: { label: 'GPU 利用率最低', color: seriesColor(1) },
    ram_pct: { label: '内存占用率', color: seriesColor(2) },
  }, { height: 180, yTickNd: 2 });
  if (sysCard) out.push(sysCard);

  return stack(out);
}
const dencePoints = (curve) => (curve.value || []).map((y, i) => ({ x: curve.step[i], y }));

/* ======================================================================== */
/*  页面：监督微调                                                           */
/* ======================================================================== */
function renderSft(d) {
  const runs = (d.sft && d.sft.runs) || [];
  const cmp = (d.sft && d.sft.compare) || {};
  const out = [];
  if (!runs.length) {
    out.push(card('没有 SFT 记录', '需要 <code>test/log/sft_*_metrics.csv</code>。', mkEl('div', 'empty-note', '—')));
    return stack(out);
  }
  out.push(notice('三个 run 共用同一套数据口径（同 seq_len / 同等效 batch / 同 seed），'
    + '因此 <b>val_loss 可以直接横向比较</b>；Dense 与 MoE 的差异来自结构而非训练量。', ''));

  out.push(tiles(runs.map((r) => tile(r.label, fmtNum(r.val_loss_last), {
    unit: '', foot: `train ${fmtNum(r.loss_last)} · ${fmtInt(r.steps)} 步`,
    spark: r.val_loss_spark, color: seriesColor(runs.indexOf(r)),
  })).concat([
    tile('MoE/Dense 激活参数比', cmp.ratio_act ? `${cmp.ratio_act}×` : '—', { unit: '', foot: '等质量比算力的折算系数' }),
    tile('MoE 相对 Dense', cmp.val_loss_delta != null ? fmtSigned(cmp.val_loss_delta) : '—',
      { unit: 'val', foot: '负 = MoE 更低（更好）' }),
  ])));
  // 吞吐是「同预算下谁更快」的硬指标，单独一组统计牌（带迷你趋势线）
  const tpsTiles = runs.filter((r) => r.tps_spark).map((r) => tile(`${r.label} 吞吐`, fmtK(r.tokens_per_sec), {
    unit: 'tok/s', foot: 'tail100 均值 · 趋势见下方迷你折线', spark: r.tps_spark, color: seriesColor(runs.indexOf(r)),
  }));
  if (tpsTiles.length) out.push(tiles(tpsTiles));

  // val_loss 曲线：3 条 → 图例 + 端点直标
  const series = runs.filter((r) => r.val_loss && r.val_loss.value.length).map((r, i) => ({
    key: r.key, name: `${r.label} val_loss`, shortName: r.label.replace(' · ', '/'),
    color: seriesColor(i), points: dencePoints(r.val_loss),
  }));
  const hidden = new Set();
  const wrap = mkEl('div');
  wrap.appendChild(chartHost((h) => window.Viz.lineChart(h, series.filter((s) => !hidden.has(s.key)), {
    height: 260, xLabel: 'step', yLabel: 'val_loss', yTickNd: 3, xTickFmt: (v) => fmtK(v),
  }), 260));
  wrap.appendChild(legendHost(series, hidden, (k) => { hidden.has(k) ? hidden.delete(k) : hidden.add(k); rerender(); }));
  out.push(card('留出集 val_loss', '在训练数据尾部切出的留出集上评估 —— 这是唯一能判断「有没有过拟合」的曲线', wrap));

  // Dense vs MoE 的逐点差：正负相反的两个方向 → 发散条 + 零线折线
  if (cmp.points && cmp.points.length) {
    const pts = cmp.points;
    const hist = chartHost((h) => window.Viz.lineChart(h,
      [{ key: 'd', name: 'Δ val_loss（MoE − Dense）', color: token('--ink-2'), points: pts.map((p) => ({ x: p.step, y: p.delta })) }],
      { height: 200, zeroLine: true, area: false, yTickFmt: (v) => fmtSigned(v, 2), xTickFmt: (v) => fmtK(v),
        yTickNd: 4, markYLabel: '' }), 200);
    const recent = pts.slice(-10).map((p) => ({
      label: `step ${fmtK(p.step)}`, value: p.delta, text: fmtSigned(p.delta, 4),
    }));
    out.push(card('Δ val_loss：MoE 相对 Dense', '零线以上 = MoE 更差（红），以下 = 更好（蓝）；右侧是最近 10 个采样点',
      grid('c2', [hist, chartHost((h) => window.Viz.barChart(h, recent, {
        labelW: 92, rowH: 17, symmetric: true, valueW: 74, gap: 2,
        tickFmt: (v) => fmtSigned(v, 1), valueLabel: 'Δ val_loss',
      }), 220)])));
  }

  out.push(subTitle('优化动力学对照 · 三个 run 各自分面'));
  out.push(grid('c2', runs.map((r) => optCardFor(r, 'update_ratio'))));
  out.push(grid('c2', runs.map((r) => optCardFor(r, 'gate'))));
  out.push(grid('c2', runs.filter((r) => r.layer_hidden).map((r) => matrixCard(
    `${r.label} · 残差流逐层 RMS`,
    '同一套口径下对比 Dense 与 MoE 的深度维行为；颜色按行内量程归一化，浮层读原值',
    r.layer_hidden, { labelW: 56, colW: 12, height: 170, nd: 2, valueLabel: 'hidden_rms' }))));

  out.push(card('三个 SFT run 的关键指标', '表格是图表的「兜底视图」：颜色之外，数值本身也能读',
    table('sft-runs', [
      { key: 'label', label: 'run', cls: 'name' },
      { key: 'note', label: '说明', cls: 'dim' },
      { key: 'steps', label: '步数', num: true, fmt: (v) => fmtInt(v) },
      { key: 'loss_first', label: '首 loss', num: true, fmt: (v) => fmtNum(v) },
      { key: 'loss_last', label: '末 loss', num: true, fmt: (v) => fmtNum(v) },
      { key: 'val_loss_first', label: '首 val', num: true, fmt: (v) => fmtNum(v) },
      { key: 'val_loss_last', label: '末 val', num: true, fmt: (v) => fmtNum(v) },
      { key: 'val_loss_min', label: '最优 val', num: true, fmt: (v) => fmtNum(v) },
      { key: 'val_loss_min_step', label: '最优步', num: true, fmt: (v) => fmtK(v) },
      { key: 'tokens_per_sec', label: 'tok/s', num: true, fmt: (v) => fmtK(v) },
      { key: 'gpu_mem_mb', label: '显存 MB', num: true, fmt: (v) => fmtK(v) },
    ], runs)));
  return stack(out);
}

/** SFT 单个 run 的优化动力学小卡（四个参数组同轴）。 */
function optCardFor(r, kind) {
  const spec = kind === 'gate'
    ? { gate_mean: { label: '门控均值', short: 'mean', color: seriesColor(0), },
        gate_std: { label: '门控标准差', short: 'std', color: seriesColor(1) },
        gate_sat_lo: { label: '低饱和比例', short: 'sat_lo', color: seriesColor(6) },
        gate_sat_hi: { label: '高饱和比例', short: 'sat_hi', color: seriesColor(3) } }
    : { update_ratio_attn: { label: '注意力', short: 'attn', color: seriesColor(0) },
        update_ratio_ffn: { label: '前馈', short: 'ffn', color: seriesColor(1) },
        update_ratio_embed: { label: '词嵌入', short: 'embed', color: seriesColor(2) },
        update_ratio_norm: { label: '归一化', short: 'norm', color: seriesColor(3) } };
  const src = kind === 'gate' ? r.gate_curves : r.opt_curves;
  return curveCard(`${r.label} · ${kind === 'gate' ? '门控' : '相对更新量'}`, r.note, src, spec,
    { height: 150, yTickNd: kind === 'gate' ? 3 : 6 });
}

/* ======================================================================== */
/*  页面：强化学习                                                           */
/* ======================================================================== */
//: 各算法「专属」的日志列 —— 这正是「不同算法记录不同 log 信息」的落点
const ALGO_FIELDS = {
  dpo: [['preference_acc', '偏好对准确率', (v) => fmtPct(v)], ['reward_margin', '偏好间隔', (v) => fmtSigned(v, 4)],
        ['preference_loss', '偏好损失', (v) => fmtNum(v, 4)]],
  ipo: [['preference_acc', '偏好对准确率', (v) => fmtPct(v)], ['reward_margin', '偏好间隔', (v) => fmtSigned(v, 4)]],
  simpo: [['preference_acc', '偏好对准确率', (v) => fmtPct(v)], ['reward_margin', '长度归一化间隔', (v) => fmtSigned(v, 4)]],
  cpo: [['preference_acc', '偏好对准确率', (v) => fmtPct(v)], ['reward_margin', '偏好间隔', (v) => fmtSigned(v, 4)]],
  orpo: [['preference_acc', '偏好对准确率', (v) => fmtPct(v)], ['reward_margin', 'odds-ratio 间隔', (v) => fmtSigned(v, 4)]],
  kto: [['preference_acc', '偏好对准确率', (v) => fmtPct(v)], ['reward_margin', '参考策略间隔', (v) => fmtSigned(v, 4)]],
  grpo: [['reward', '平均奖励', (v) => fmtNum(v, 4)], ['group_zero_std', '组内奖励零方差比例', (v) => fmtPct(v)],
         ['clipfrac', '被裁剪 token 比例', (v) => fmtPct(v)], ['kl_ref', '相对参考模型 KL', (v) => fmtNum(v, 4)],
         ['response_len', '回复长度', (v) => fmtNum(v, 1)]],
  dapo: [['reward', '平均奖励', (v) => fmtNum(v, 4)], ['group_zero_std', '过滤组比例', (v) => fmtPct(v)],
         ['clipfrac', '被裁剪 token 比例', (v) => fmtPct(v)], ['kl_ref', '相对参考模型 KL', (v) => fmtNum(v, 4)],
         ['response_len', '回复长度', (v) => fmtNum(v, 1)]],
  rloo: [['reward', '平均奖励', (v) => fmtNum(v, 4)], ['group_zero_std', '组内奖励零方差比例', (v) => fmtPct(v)],
         ['clipfrac', '被裁剪 token 比例', (v) => fmtPct(v)], ['kl_ref', '相对参考模型 KL', (v) => fmtNum(v, 4)],
         ['response_len', '回复长度', (v) => fmtNum(v, 1)]],
  ppo: [['reward', '平均奖励', (v) => fmtNum(v, 4)], ['kl_ref', '相对参考模型 KL', (v) => fmtNum(v, 4)],
        ['clipfrac', '被裁剪 token 比例', (v) => fmtPct(v)], ['policy_loss', '策略损失', (v) => fmtNum(v)],
        ['grad_norm', '梯度范数', (v) => fmtNum(v, 3)]],
  agent: [['reward', '整轮结算奖励', (v) => fmtNum(v, 4)], ['turns_mean', '平均工具调用轮数', (v) => fmtNum(v, 2)],
          ['valid_call_rate', '合法工具调用比例', (v) => fmtPct(v)], ['pass_rate', '任务通过率', (v) => fmtPct(v)],
          ['response_len', '轨迹长度', (v) => fmtNum(v, 1)]],
};

//: 算法专属曲线的显示规格。**一个算法只画它自己那一族**，所以这里的 key 可以与
//: 通用族重名 —— 它们不会出现在同一张卡里（PPO 的 ``clipfrac`` 走 gen_curves，
//: 这张表只负责 ``algo_curves`` 里真正只有该算法才有的列）。
//:
//: 写成**函数**而不是常量：颜色必须等到渲染时再取 ``seriesColor()``，否则主题在
//: 脚本加载之后才切换的话，取到的会是浅色面那一套（``token()`` 读的是当前计算样式）。
function algoCurveSpec() {
  return {
    // 离线偏好：没有 rollout，因此这一族全是 chosen/rejected 对上的量
    preference_loss: { label: '偏好损失', short: 'preference_loss', color: seriesColor(0) },
    dpo_loss:        { label: 'DPO 损失', short: 'dpo_loss', color: seriesColor(0) },
    reward_margin:   { label: '偏好间隔（选中 − 拒绝）', short: 'margin', color: seriesColor(1) },
    preference_acc:  { label: '偏好对准确率', short: 'pref_acc', color: seriesColor(3) },
    // GRPO：组内奖励是否退化成常数
    group_reward_zero_std: { label: '组内奖励零方差比例', short: 'zero_std', color: seriesColor(6) },
    // PPO：critic 侧 + KL 早停 + 双学习率
    critic_loss:     { label: 'critic 损失', short: 'critic', color: seriesColor(0) },
    value_loss:      { label: 'value 损失（与 critic 同源）', short: 'value', color: seriesColor(1), dash: true },
    approx_kl:       { label: 'approx KL', short: 'approx_kl', color: seriesColor(2) },
    kl_early_stop:   { label: 'KL 早停触发次数（本步）', short: 'early_stop', color: seriesColor(6) },
    actor_lr:        { label: 'Actor 学习率', short: 'actor_lr', color: seriesColor(4) },
    critic_lr:       { label: 'Critic 学习率', short: 'critic_lr', color: seriesColor(5) },
    // Agentic：工具调用行为
    pass_rate:       { label: '任务通过率', short: 'pass', color: seriesColor(3) },
    unfinished_rate: { label: '未完成比例', short: 'unfinished', color: seriesColor(6) },
    turns_mean:      { label: '平均对话轮数', short: 'turns', color: seriesColor(0) },
    tool_calls_mean: { label: '平均工具调用次数', short: 'calls', color: seriesColor(1) },
    valid_call_rate: { label: '工具调用合法率', short: 'valid', color: seriesColor(2) },
    tool_gap_mean:   { label: '调用数与参考答案的偏离', short: 'gap', color: seriesColor(4) },
  };
}

function renderRl(d) {
  const runs = (d.rl && d.rl.runs) || [];
  const out = [];
  if (!runs.length) {
    out.push(card('没有 RL 记录', '需要 <code>test/log/rl/*_metrics.csv</code>。', mkEl('div', 'empty-note', '—')));
    return stack(out);
  }
  const archs = ['dense', 'moe'];
  const arch = state.rlArch;

  out.push(notice('四种算法共享同一个环境与奖励口径，但 <b>各自记录的日志列不同</b>：'
    + 'DPO 记偏好准确率与间隔，GRPO/CISPO 记组内零方差比例与裁剪率，PPO 记 critic 与 KL 早停，'
    + 'Agentic 记工具轮数与调用合法率。下面的「算法专属指标」区块逐算法列出。', ''));

  const seg = mkEl('div');
  seg.style.cssText = 'display:flex;gap:6px';
  for (const a of archs) {
    const b = mkEl('button', `btn${a === arch ? ' primary' : ' ghost'}`,
      `<span style="display:inline-block;width:9px;height:9px;border-radius:3px;background:${ARCH_COLOR[a]}"></span>${a === 'dense' ? 'Dense' : 'MoE'}`);
    b.addEventListener('click', () => { state.rlArch = a; rerender(); });
    seg.appendChild(b);
  }
  out.push(card('分面：当前架构', 'Dense 与 MoE 分开看 —— 换架构会整体抬高或压低奖励水平，叠在一张图里会掩盖算法差异',
    seg));

  const facet = runs.filter((r) => r.arch === arch && r.steps);

  // 奖励曲线：4 条算法（同一架构内），图例 + 直标
  const mkSeries = (field, label, fmt) => facet.filter((r) => r[field] && r[field].value.length).map((r) => ({
    key: r.algo, name: `${r.algo_label} ${label}`, shortName: r.algo_label,
    color: algoColor(r.algo), points: dencePoints(r[field]),
  }));

  const hidden = new Set();
  const rewardSeries = mkSeries('reward', '奖励', fmtNum);
  const w1 = mkEl('div');
  w1.appendChild(chartHost((h) => window.Viz.lineChart(h, rewardSeries.filter((s) => !hidden.has(s.key)), {
    height: 260, xLabel: 'step', yLabel: 'reward', yTickNd: 4, xTickFmt: (v) => fmtK(v),
  }), 260));
  w1.appendChild(legendHost(rewardSeries, hidden, (k) => { hidden.has(k) ? hidden.delete(k) : hidden.add(k); rerender(); }));
  out.push(card(`平均奖励 · ${arch === 'dense' ? 'Dense' : 'MoE'}`, '同一奖励尺度的算法可以直接叠看', w1));

  // 回复长度 + 通过率：量纲不同 → 两张图并排，绝不双轴
  const lenSeries = mkSeries('response_len', '回复长度', fmtNum);
  const passSeries = mkSeries('pass_rate', '通过率', fmtPct);
  const side = [];
  if (lenSeries.length) {
    const hh = new Set();
    const w = mkEl('div');
    w.appendChild(chartHost((h) => window.Viz.lineChart(h, lenSeries.filter((s) => !hh.has(s.key)),
      { height: 200, yTickFmt: (v) => fmtK(v), yLabel: 'tokens', xTickFmt: (v) => fmtK(v), endLabels: false }), 200));
    w.appendChild(legendHost(lenSeries, hh, (k) => { hh.has(k) ? hh.delete(k) : hh.add(k); rerender(); }));
    side.push(card('回复长度', '长度暴涨通常是奖励被「写长」投机的信号', w));
  }
  if (passSeries.length) {
    const hh = new Set();
    const w = mkEl('div');
    w.appendChild(chartHost((h) => window.Viz.lineChart(h, passSeries.filter((s) => !hh.has(s.key)),
      { height: 200, yTickFmt: (v) => fmtPct(v, 0), yLabel: '通过率', xTickFmt: (v) => fmtK(v), endLabels: false }), 200));
    w.appendChild(legendHost(passSeries, hh, (k) => { hh.has(k) ? hh.delete(k) : hh.add(k); rerender(); }));
    side.push(card('任务通过率', '奖励的「硬」版本：规则判定的对错', w));
  }
  if (side.length) out.push(grid('c2', side));

  // 墙钟：时间轴
  const timed = facet.filter((r) => r.wall_hours != null);
  if (timed.length) {
    out.push(card(`墙钟耗时 · ${arch === 'dense' ? 'Dense' : 'MoE'}`,
      '同一步数下不同算法的实际开销差异（rollout 与 update 的比例不同）',
      timeline(timed.map((r) => ({
        name: r.algo_label, hours: r.wall_hours, color: algoColor(r.algo), status: r.status,
        detail: `${r.steps} 步`,
      })))));
  }

  // ---- 奖励分解：堆叠面积（本轮重点）----
  // 「奖励涨了」有两种完全不同的原因：模型真的更好了（rew_rm 涨），
  // 或者学会了把答案写长（rew_len 涨）。折线叠画看不出「谁贡献了多少」，
  // 堆叠面积才同时给出总量与占比。Agent 的 rew_* 列实测全空 → 这一区块自动不出现。
  const withParts = facet.filter((r) => r.reward_parts && Object.keys(r.reward_parts).length);
  if (withParts.length) {
    const REW_SPEC = [
      ['rew_rm', 'RM 打分', 4], ['rew_len', '长度分', 0], ['rew_think_len', '思考长度分', 1],
      ['rew_think_close', '思考闭合分', 2], ['rew_rep', '重复惩罚（为负）', 6],
    ];
    const blocks = withParts.map((r) => {
      const series = REW_SPEC.filter(([k]) => r.reward_parts[k]).map(([k, name, slot]) => ({
        key: k, name, color: seriesColor(slot),
        points: (r.reward_parts[k].value || []).map((y, i) => ({ x: r.reward_parts[k].step[i], y })),
      }));
      const wrap = mkEl('div');
      wrap.appendChild(chartHost((h) => window.Viz.stackedArea(h, series, {
        height: 230, yLabel: 'reward 分项', xTickFmt: (v) => fmtK(v), yTickNd: 3,
        totalLine: { name: '合计奖励', shortName: '合计', color: token('--ink'), points: dencePoints(r.reward) },
        unit: '', xUnit: 'step', tipNd: 4,
      }), 230));
      wrap.appendChild(legendHost(series.map((s) => ({ key: s.key, name: s.name, color: s.color })), null, () => {}));
      return card(`${r.algo_label} · 奖励构成`, '正分项向上堆、惩罚项向下堆；黑线是实测总奖励。'
        + '若总量在涨而 RM 分（深色带）没涨、只有长度分在涨 → 是长度投机', wrap);
    });
    out.push(subTitle('奖励分解 · 涨的到底是哪一项'));
    out.push(grid('c2', blocks));
  }

  // ---- 生成本身的健康度：看的是「模型写成什么样」，与算法无关 ----
  // 这一族只随 rollout 的形状变化，所以把有该指标的在线算法叠在同一张图里：
  // 同一架构、同一环境、同一奖励口径，谁把输出写得越来越不像话，一眼能看出来。
  // 量纲不同 → 分四张卡（绝不双轴）；每张卡里一个算法一条线，身份靠固定色位 + 图例。
  const algoCols = (key) => {
    const cols = {};
    const spec = {};
    for (const r of facet) {
      const c = (r.gen_curves || {})[key];
      if (!c) continue;
      cols[r.algo] = c;
      spec[r.algo] = { label: r.algo_label, short: r.algo_label, color: algoColor(r.algo) };
    }
    return { cols, spec };
  };
  const genCards = [];
  {
    // EOS 率与截断率是同一枚硬币的两面（量纲相同、和 ≈ 1）→ 合成一张双系列卡
    const { cols: ec, spec: es } = algoCols('eos_rate');
    const { cols: tc, spec: ts } = algoCols('trunc_rate');
    const cols = {}; const spec = {};
    for (const a of Object.keys(ec)) { cols[`${a}__eos`] = ec[a]; spec[`${a}__eos`] = es[a]; }
    for (const a of Object.keys(tc)) {
      cols[`${a}__trunc`] = tc[a];
      spec[`${a}__trunc`] = Object.assign({}, ts[a], { label: `${ts[a].label} 截断率`, dash: true });
    }
    genCards.push(curveCard('正常收尾（EOS）率 vs 被截断比例',
      '实线 = EOS 率，虚线 = 截断率；同一算法的两条应当互补（和 ≈ 1）。EOS 率往下掉 = 模型越来越常'
      + '写不完就被 max_gen_len 截断 —— 此时长度分与 RM 分都会失真', cols, spec,
    { height: 200, yTickNd: 3 }));
    const ppl = algoCols('perplexity');
    genCards.push(curveCard('策略对自身输出的困惑度', '困惑度暴跌是熵塌缩的前兆'
      + '（输出退化成同几句话），暴涨则说明策略已经跑到训练分布之外', ppl.cols, ppl.spec,
    { height: 180, yTickNd: 3 }));
    const ratio = algoCols('ratio_mean');
    genCards.push(curveCard('importance ratio 均值', '新旧策略的比值应当贴着 1；持续偏离说明采样用的'
      + '策略与正在更新的策略已经不同步（off-policy 程度加深）。'
      + '这一列只有 GRPO / CISPO 记录（PPO 与 Agent 记的是同族的 approx_kl / kl_ref）',
    ratio.cols, ratio.spec, { height: 180, yTickNd: 4 }));
    const gt = algoCols('gen_tokens');
    genCards.push(curveCard('每步生成 token 数', '本步 rollout 实际生成的 token 总数；'
      + '与吞吐（tok/s）合看能区分「算得变慢」与「写得变短」', gt.cols, gt.spec,
    { height: 180, yTickFmt: (v) => fmtK(v), yTickNd: 0 }));
  }
  if (genCards.filter(Boolean).length) {
    out.push(subTitle('生成本身的健康度（与算法无关，看的是「模型写成什么样」）'));
    out.push(grid('c2', genCards.filter(Boolean)));
  }

  // ---- 算法专属曲线（PPO 的 critic / Agent 的工具轮次 / DPO 的偏好）----
  const algoCurveBlocks = facet.filter((r) => r.algo_curves && Object.keys(r.algo_curves).length);
  if (algoCurveBlocks.length) {
    out.push(subTitle('算法专属曲线 · 这一族只有该算法才有'));
    const algoSpec = algoCurveSpec();
    out.push(grid('c2', algoCurveBlocks.map((r) => {
      const spec = {};
      for (const k of r.algo_curve_keys || []) spec[k] = algoSpec[k] || { label: k };
      return curveCard(`${r.algo_label} · 专属指标`, r.algo_note, r.algo_curves, spec,
        { height: 210, yTickNd: 4 });
    })));
    // PPO 的 KL 早停：**阈值不在 CSV 里**（它来自 configs/rl.yaml 的 early_stop_kl，
    // 随算法参数而变，因此不能在图里画成一条固定的阈值线）。能画的是「触发了几次」。
    const ppo = facet.find((r) => r.algo === 'ppo');
    if (ppo && (ppo.algo_curves || {}).kl_early_stop) {
      const cur = ppo.algo_curves.kl_early_stop;
      const fired = cur.value.filter((v) => v > 0).length;
      out.push(notice(`PPO 在 ${cur.value.length} 个采样点里有 <b>${fired}</b> 次触发 KL 早停`
        + `（最新一点 kl_early_stop = ${fmtNum(cur.value[cur.value.length - 1], 3)}）。`
        + '触发说明这一步的策略偏离参考模型过快，该 step 的策略更新被跳过。'
        + '阈值本身来自训练配置 <code>early_stop_kl</code>，不在逐 step 指标里，因此图上不画阈值线。',
      fired ? 'warn' : ''));
    }
  }

  // ---- 算法 × 指标家族覆盖矩阵：一图回答「谁记了什么 log」----
  const cov = d.coverage;
  if (cov && (cov.families || []).length) {
    const rows = cov.families.map((f) => f.family);
    const cols = cov.algos.map((a) => (cov.algo_labels.find((x) => x.key === a) || {}).label || a);
    const values = cov.families.map((f) => f.values.map((v) => (v > 0 ? v : null)));
    const raw = cov.families.map((f) => cov.algos.map((a) => {
      const o = f.by_algo[a] || { hit: 0, of: f.total };
      return o.hit / Math.max(1, o.of);
    }));
    const totalCols = cov.algos.map((a) => (cov.total_cols || {})[a] || 0);
    out.push(card('算法 × 指标家族 覆盖矩阵',
      '上半部分是所有训练共有的基础采集，下半部分展示离线偏好与在线 RL 的专属差异。'
      + '格子里是「该家族的代表列里有几成真的出现在该算法的 CSV 表头里」。'
      + `各算法 CSV 的列数：${cov.algos.map((a, i) => `${cols[i]} ${totalCols[i]}`).join(' · ')} 列 —— `
      + '不同算法的目标与训练循环不同，这张图给出实际记录差异的全貌',
      chartHost((h) => window.Viz.heatmap(h, rows, cols, values, {
        raw, rowH: 20, labelW: 96, colW: 108, colLabelH: 20, showValues: true,
        valueLabel: '覆盖度', hint: '灰格 = 该算法完全没记这一族',
        rawFmt: fmtPct, cellFmt: (v) => v.toFixed(2),
      }), 340)));

    // 专属列清单：表格是矩阵的兜底视图（矩阵只给了「有几成」，这里给具体列名）
    const exRows = cov.algos.map((a) => ({
      algo: (cov.algo_labels.find((x) => x.key === a) || {}).label || a,
      total: totalCols[cov.algos.indexOf(a)],
      n: ((cov.exclusive || {})[a] || []).length,
      cols: ((cov.exclusive || {})[a] || []).join(', ') || '—',
    }));
    out.push(card('各算法独有的日志列', '只在该算法出现、其它算法没有的列 —— 这就是「针对不同算法记录不同 log」的确切含义',
      table('rl-cov', [
        { key: 'algo', label: '算法', cls: 'name' },
        { key: 'total', label: 'CSV 列数', num: true },
        { key: 'n', label: '独有列数', num: true },
        { key: 'cols', label: '独有列', cls: 'dim' },
      ], exRows)));
  }

  // 各算法专属指标：这是「不同算法记录不同 log」的正面展示
  out.push(subTitle('算法专属指标（tail100 均值）'));
  const algoCards = ALGO_ORDER.map((a) => {
    const rs = runs.filter((r) => r.algo === a);
    if (!rs.length) return null;
    const body = mkEl('div');
    const fields = ALGO_FIELDS[a] || [];
    body.appendChild(kv(fields.map(([key, label, fmt]) => {
      const vals = rs.map((r) => {
        const v = r[`${key}_tail`];
        return v == null ? '—' : fmt(v);
      });
      return [`${label}`, `${rs.map((r) => r.arch[0].toUpperCase()).join(' / ')} → ${vals.join(' / ')}`, true];
    })));
    const note = (rs[0] || {}).algo_note || '';
    return card(`${(rs[0] || {}).algo_label || a}`, note, body);
  }).filter(Boolean);
  out.push(grid('c2', algoCards));

  // 全量表
  out.push(card('RL 全部 run 指标', '空值 = 该算法不产出这一列（不是 0）',
    table('rl-all', [
      { key: 'key', label: 'run', cls: 'name' },
      { key: 'algo_label', label: '算法' },
      { key: 'arch', label: '架构', cell: (td, v) => td.appendChild(badge(v, 'mute')) },
      { key: 'status', label: '状态', cell: (td, v) => td.appendChild(statusBadge(v === 'OK' ? 'ok' : (v === 'FAIL' ? 'failed' : v))) },
      { key: 'steps', label: 'step', num: true },
      { key: 'wall_hours', label: '小时', num: true, fmt: (v) => fmtNum(v, 2) },
      { key: 'reward_first', label: '首奖励', num: true, fmt: (v) => fmtNum(v) },
      { key: 'reward_last', label: '末奖励', num: true, fmt: (v) => fmtNum(v) },
      { key: 'response_len_tail', label: '长度', num: true, fmt: (v) => fmtNum(v, 1) },
      { key: 'pass_rate_tail', label: '通过率', num: true, fmt: (v) => fmtPct(v) },
      { key: 'preference_acc_tail', label: '偏好准确率', num: true, fmt: (v) => fmtPct(v) },
      { key: 'kl_ref_tail', label: 'KL', num: true, fmt: (v) => fmtNum(v) },
      { key: 'clipfrac_tail', label: '裁剪率', num: true, fmt: (v) => fmtPct(v) },
      { key: 'eos_rate_tail', label: 'EOS 率', num: true, fmt: (v) => fmtPct(v) },
      { key: 'trunc_rate_tail', label: '截断率', num: true, fmt: (v) => fmtPct(v) },
      { key: 'group_zero_std_tail', label: '零方差组', num: true, fmt: (v) => fmtPct(v) },
      { key: 'turns_mean_tail', label: '轮数', num: true, fmt: (v) => fmtNum(v, 2) },
      { key: 'valid_call_rate_tail', label: '合法调用', num: true, fmt: (v) => fmtPct(v) },
      { key: 'gpu_mem_peak_mb', label: '峰值显存 MB', num: true, fmt: (v) => fmtK(v) },
    ], runs)));
  return stack(out);
}

/* ======================================================================== */
/*  页面：架构对比                                                           */
/* ======================================================================== */
function renderSweep(d) {
  const cfgs = (d.sweep && d.sweep.configs) || [];
  const base = (d.sweep && d.sweep.baseline_name) || '';
  const out = [];
  if (!cfgs.length) {
    out.push(card('没有架构 sweep 结果', '需要 <code>test/storage/report/summary.csv</code>。',
      mkEl('div', 'empty-note', '—')));
    return stack(out);
  }
  out.push(notice(`共 ${cfgs.length} 组配置，基线是 <code>${escapeHtml(base)}</code>。`
    + '所有配置同数据、同 token 预算、同 seed，因此 <b>Δ loss 可以直接读作「结构改动值不值」</b>。', ''));

  const baseRow = cfgs.find((c) => c.name === base) || cfgs[0];

  // Hero：一个最有信息量的主数字 —— 最优配置相对基线的收益
  const best = cfgs[0];
  const hero = mkEl('div', 'hero');
  const bm = mkEl('div', 'hero-main');
  bm.appendChild(mkEl('div', 'hero-num',
    `${best.delta_vs_base != null ? fmtSigned(best.delta_vs_base, 4) : '—'}<span class="unit">Δ loss vs 基线</span>`));
  bm.appendChild(mkEl('div', 'hero-cap',
    `最优配置 <b>${escapeHtml(best.name)}</b> —— ${escapeHtml(best.desc || '')}。`
    + `末 loss ${fmtNum(best.loss)}，基线 ${fmtNum(baseRow.loss)}。负值代表比基线更好。`));
  hero.appendChild(bm);
  const heroRight = mkEl('div', 'hero-split');
  heroRight.appendChild(chartHost((h) => window.Viz.barChart(h,
    [{ label: '基线', value: baseRow.loss, text: fmtNum(baseRow.loss), color: token('--ink-3') },
     { label: '最优', value: best.loss, text: fmtNum(best.loss), color: token('--accent') }],
    { labelW: 52, rowH: 20, valueW: 66, minRef: baseRow.loss, tickFmt: (v) => fmtNum(v, 2), valueLabel: '末 loss' }), 76));
  hero.appendChild(heroRight);
  out.push(hero);

  out.push(tiles([
    tile('配置数', String(cfgs.length), { unit: '组' }),
    tile('优于基线', String(cfgs.filter((c) => c.delta_vs_base != null && c.delta_vs_base < 0).length), { unit: '组' }),
    tile('最快', (() => { const f = [...cfgs].sort((a, b) => (b.tps || 0) - (a.tps || 0))[0]; return f ? fmtK(f.tps) : '—'; })(),
      { unit: 'tok/s', foot: '按 median 吞吐' }),
    tile('最省显存', (() => { const f = [...cfgs].filter((c) => c.peak_mem_mb).sort((a, b) => a.peak_mem_mb - b.peak_mem_mb)[0]; return f ? fmtK(f.peak_mem_mb) : '—'; })(),
      { unit: 'MB' }),
    tile('KV cache 最小', (() => { const f = [...cfgs].filter((c) => c.cache_kb_per_token).sort((a, b) => a.cache_kb_per_token - b.cache_kb_per_token)[0]; return f ? fmtNum(f.cache_kb_per_token, 1) : '—'; })(),
      { unit: 'kB/token' }),
  ]));

  // Δ loss vs 基线：正负两个方向 → 发散条（这是本页信息密度最高的一张图）
  const deltaItems = cfgs.filter((c) => c.delta_vs_base != null && c.name !== base)
    .sort((a, b) => a.delta_vs_base - b.delta_vs_base)
    .map((c) => ({
      label: c.name.replace(/^t\d-/, ''),
      value: c.delta_vs_base,
      text: fmtSigned(c.delta_vs_base, 4),
      sub: c.desc || '',
    }));
  out.push(card('相对基线的 loss 变化', '向左（蓝）= 比基线更好，向右（红）= 更差；长度按 |Δ| 等比',
    chartHost((h) => window.Viz.barChart(h, deltaItems, {
      labelW: 210, rowH: 17, gap: 3, symmetric: true, valueW: 82,
      tickFmt: (v) => fmtSigned(v, 2), valueLabel: 'Δ loss',
    }), 340)));

  // 权衡散点：吞吐 × loss，带帕累托前缘
  const pts = cfgs.filter((c) => c.tps && c.loss != null).map((c) => ({
    x: c.tps, y: c.loss, label: c.name, ci: c.name === base ? 7 : 0,
    color: c.name === base ? token('--ink-3') : token('--accent'),
    sub: c.desc || '', r: c.name === base ? 6 : 4.5,
  }));
  if (pts.length > 2) {
    out.push(card('吞吐 × loss 的权衡', '右上角是「又慢又差」，左下角是「又快又好」。虚线是帕累托前缘',
      chartHost((h) => window.Viz.scatterChart(h, pts, {
        height: 300, pareto: true, xName: '吞吐', yName: '末 loss',
        xTickFmt: (v) => fmtK(v), yTickFmt: (v) => fmtNum(v, 3), xLabel: 'tok/s', yLabel: 'loss',
      }), 300)));
  }

  // 热力图：配置 × 指标（按列归一化配色，数值原样显示）
  const metrics = [
    ['loss', '末 loss', (v) => fmtNum(v, 3)],
    ['tps', '吞吐 tok/s', (v) => fmtK(v)],
    ['peak_mem_mb', '峰值显存 MB', (v) => fmtK(v)],
    ['params_total_m', '总参 M', (v) => fmtNum(v, 1)],
    ['params_act_m', '激活 M', (v) => fmtNum(v, 1)],
    ['cache_kb_per_token', 'KV kB/tok', (v) => fmtNum(v, 1)],
    ['grad_norm', '梯度范数', (v) => fmtNum(v, 3)],
  ];
  const rowNames = cfgs.map((c) => c.name.replace(/^t\d-/, ''));
  const norm = [];
  const raw = [];
  for (const c of cfgs) {
    const nr = [], rr = [];
    for (const [k, , fmt] of metrics) {
      const v = c[k];
      rr.push(v == null ? null : v);
      nr.push(v == null ? null : v);
    }
    norm.push(nr); raw.push(rr);
  }
  // 按列归一化：每列 (v-min)/(max-min)。这样不同量纲能在同一张图里比较排序；
  // 但显示的值始终是原始值（见 heatmap 的 raw 参数）。
  for (let j = 0; j < metrics.length; j++) {
    const col = norm.map((r) => r[j]).filter((v) => v != null);
    const lo = Math.min(...col), hi = Math.max(...col);
    for (let i = 0; i < norm.length; i++) {
      if (norm[i][j] == null) continue;
      norm[i][j] = hi > lo ? (norm[i][j] - lo) / (hi - lo) : 0.5;
    }
  }
  out.push(card('配置 × 指标矩阵', '颜色按「列内排序」着色（每列各自归一化），格子里写的是原始值 —— 颜色只用来找异常，数值才是结论',
    chartHost((h) => window.Viz.heatmap(h, rowNames, metrics.map((m) => m[1]), norm, {
      raw, mode: 'seq', rowH: 19, labelW: 176, colW: 62, colLabelH: 38,
      showValues: true, valueLabel: '值',
      rawFmt: (v) => (v >= 1000 ? fmtK(v) : v >= 10 ? v.toFixed(1) : v.toFixed(3)),
      cellFmt: (v) => v.toFixed(2),
    }), 380)));

  // 表视图
  out.push(card('全部配置明细', null,
    table('sweep-all', [
      { key: 'name', label: '配置', cls: 'name' },
      { key: 'tier', label: '层级', cell: (td, v) => td.appendChild(badge(v, 'mute')) },
      { key: 'desc', label: '说明', cls: 'dim' },
      { key: 'loss', label: '末 loss', num: true, fmt: (v) => fmtNum(v, 4) },
      { key: 'delta_vs_base', label: 'Δ vs 基线', num: true, fmt: (v) => v == null ? '—' : fmtSigned(v, 4) },
      { key: 'steps_to_target', label: '达标步数', num: true, fmt: (v) => fmtK(v) },
      { key: 'tps', label: 'tok/s', num: true, fmt: (v) => fmtK(v) },
      { key: 'peak_mem_mb', label: '显存 MB', num: true, fmt: (v) => fmtK(v) },
      { key: 'params_total_m', label: '总参 M', num: true, fmt: (v) => fmtNum(v, 1) },
      { key: 'params_act_m', label: '激活 M', num: true, fmt: (v) => fmtNum(v, 1) },
      { key: 'cache_kb_per_token', label: 'KV kB/tok', num: true, fmt: (v) => fmtNum(v, 1) },
      { key: 'grad_norm', label: '梯度范数', num: true, fmt: (v) => fmtNum(v, 4) },
    ], cfgs)));
  return stack(out);
}

/* ======================================================================== */
/*  页面：标准评测                                                           */
/* ======================================================================== */
/** 任务分组的展示名与顺序。分组是「哪个阶段的产出」的口径，不是视觉装饰。 */
const EVAL_GROUPS = [
  { key: 'core', label: '基础能力' },
  { key: 'sft', label: 'SFT 产出' },
  { key: 'rl', label: 'RL 产出' },
];

/**
 * 评测页。数据层只给「权重 × 任务」的点估计，这里负责把它读成结论：
 *   - 档位过滤：quick 与 standard 题集不同，混在一起比会得出假结论；
 *   - 分组过滤：core / sft / rl 三组分别对应一个阶段的产出；
 *   - 相对基线：选定一个权重后，其余权重按「比它高/低多少」着色；
 *   - BPB 单独成图：它是 bits/token，与 0–1 的准确率不是同一个量纲；
 *   - SFT / RL 专属细项：ifeval 的三档通过率、agentic 的工具调用拆解。
 */
function renderEval(d) {
  const allModels = (d.eval && d.eval.models) || [];
  const allTasks = (d.eval && d.eval.tasks) || [];
  const rand = (d.eval && d.eval.random) || {};
  const out = [];
  if (!allModels.length) {
    out.push(card('没有评测结果', '需要 <code>test/storage/report/eval/summary.csv</code>。',
      mkEl('div', 'empty-note', '—')));
    return stack(out);
  }

  // ---- 过滤器：档位 + 分组 ----
  const depths = [...new Set(allModels.map((m) => m.depth).filter(Boolean))];
  const depth = depths.includes(state.evalDepth) ? state.evalDepth : 'all';
  const group = EVAL_GROUPS.some((g) => g.key === state.evalGroup) ? state.evalGroup : 'all';
  const base = allModels.some((m) => m.model === state.evalBase) ? state.evalBase : '';

  const models = depth === 'all' ? allModels : allModels.filter((m) => m.depth === depth);
  const tasks = group === 'all' ? allTasks : allTasks.filter((t) => t.group === group);

  const toolbar = mkEl('div', 'eval-toolbar');
  const pills = mkEl('div', 'pill-group');
  for (const [k, label] of [['all', '全部档位'], ...depths.map((x) => [x, `${x} 档`])]) {
    const b = mkEl('button', 'pill-btn', label);
    b.setAttribute('aria-pressed', String(k === depth));
    b.addEventListener('click', () => { state.evalDepth = k; rerender(); });
    pills.appendChild(b);
  }
  toolbar.appendChild(pills);

  const groupPills = mkEl('div', 'pill-group');
  for (const [k, label] of [['all', '全部任务'], ...EVAL_GROUPS.map((g) => [g.key, g.label])]) {
    const b = mkEl('button', 'pill-btn', label);
    b.setAttribute('aria-pressed', String(k === group));
    b.addEventListener('click', () => { state.evalGroup = k; rerender(); });
    groupPills.appendChild(b);
  }
  toolbar.appendChild(groupPills);

  const selBox = mkEl('div', 'delta-select-box', '<span>对比基线</span>');
  const sel = mkEl('select', 'delta-select');
  sel.appendChild(mkEl('option', null, '不对比'));
  sel.lastChild.value = '';
  for (const m of models) {
    const o = mkEl('option', null, m.model);
    o.value = m.model;
    if (m.model === base) o.selected = true;
    sel.appendChild(o);
  }
  sel.addEventListener('change', () => { state.evalBase = sel.value; rerender(); });
  selBox.appendChild(sel);
  toolbar.appendChild(selBox);
  out.push(toolbar);

  if (!models.length || !tasks.length) {
    out.push(card('这个组合没有结果', '换一个档位或分组。', mkEl('div', 'empty-note', '—')));
    return stack(out);
  }

  const mixedDepth = new Set(models.map((m) => m.depth).filter(Boolean)).size > 1;
  if (mixedDepth) {
    out.push(notice('表里同时有 <b>quick</b> 与 <b>standard</b> 两个档位 —— 档位不同 → 题集不同 → '
      + '<b>跨档位的行不能直接比</b>。点上面的档位切换只看其中一档。', 'warn'));
  }

  // ---- 统计牌：综合最强 + 每组各一个代表数字 ----
  const best = [...models].sort((a, b) => avgScore(b, tasks) - avgScore(a, tasks))[0];
  const tileItems = [
    tile('综合最强', best.model, { unit: '', foot: `${best.stage} · 平均 ${fmtPct(avgScore(best, tasks))}` }),
    tile('当前权重', String(models.length), { unit: '个', foot: `${tasks.length} 个任务` }),
  ];
  for (const g of EVAL_GROUPS) {
    const gt = allTasks.filter((t) => t.group === g.key);
    const winner = [...models].sort((a, b) => avgScore(b, gt) - avgScore(a, gt))[0];
    if (!winner || avgScore(winner, gt) < 0) continue;
    tileItems.push(tile(`${g.label}最强`, winner.model,
      { unit: '', foot: `平均 ${fmtPct(avgScore(winner, gt))}` }));
  }
  out.push(tiles(tileItems));

  // ---- 相对基线的差值表（只在选定基线时出现）----
  const baseModel = models.find((m) => m.model === base);
  if (baseModel) {
    out.push(card(`相对 ${base} 的差值`,
      '正数 = 比基线高。多选类的随机水平是 25~50%，所以 +1pp 已经是可观的变化；'
      + '生成类与 agentic 从 0 起算，差值本身就是绝对分',
      evalDeltaTable(models, tasks, baseModel, rand)));
  }

  // ---- 热力图 ----
  out.push(card('模型 × 任务得分矩阵',
    '颜色按列归一化（每列看自己的相对高低），格子里是真实得分。悬停格子看样本量 n 与 95% CI 半宽',
    evalHeatmap(models, tasks, rand)));

  // ---- BPB：唯一低方差主指标，单独成图 ----
  const bpbModels = models.filter((m) => m.bpb != null);
  if (bpbModels.length && (group === 'all' || group === 'core')) {
    const items = [...bpbModels].sort((a, b) => a.bpb - b.bpb).map((m) => ({
      label: m.model, value: m.bpb, text: m.bpb.toFixed(3),
      color: stageColor(m.stage), sub: `${m.stage} · ${m.depth || '?'} 档 · 越低越好`,
    }));
    out.push(card('WikiText BPB（越低越好）',
      '同一段 wikitext-2、同一 token 数，跨权重可以直接相减，不受采样噪声影响。'
      + '它与上面的准确率不是同一个量纲，所以单独画',
      chartHost((h) => window.Viz.barChart(h, items, {
        labelW: 188, rowH: 19, valueLabel: 'bits/token',
        tickFmt: (v) => v.toFixed(2),
      }), 160)));
  }

  // ---- SFT 产出：ifeval 三档通过率 ----
  if (group === 'all' || group === 'sft') {
    const sftCard = evalIfevalCard(models);
    if (sftCard) out.push(sftCard);
  }

  // ---- RL 产出：agentic 工具调用拆解 ----
  if (group === 'all' || group === 'rl') {
    const rlCard = evalAgenticCard(models);
    if (rlCard) out.push(rlCard);
  }

  // ---- 权重聚光灯 ----
  out.push(evalSpotlight(models, tasks, rand, base));

  // ---- 全量表 ----
  out.push(card('全部评测结果',
    '文本色标出「相对随机水平」：绿色 ≥1.4×、红色 <1.0×（即不如乱猜）。'
    + '格子下方的小字是 95% CI 半宽',
    evalTable(models, tasks, rand)));
  return stack(out);
}

/** 热力图：按列归一化着色，浮层带 n 与 CI。 */
function evalHeatmap(models, tasks, rand) {
  const rowNames = models.map((m) => m.model);
  const values = [], raw = [];
  for (const m of models) {
    const nr = [], rr = [];
    for (const t of tasks) {
      const v = m[t.key];
      rr.push(v == null ? null : v);
      nr.push(v == null ? null : v);
    }
    values.push(nr); raw.push(rr);
  }
  for (let j = 0; j < tasks.length; j++) {
    const col = values.map((r) => r[j]).filter((v) => v != null);
    if (!col.length) continue;
    const lo = Math.min(...col), hi = Math.max(...col);
    for (let i = 0; i < values.length; i++) {
      if (values[i][j] == null) continue;
      values[i][j] = hi > lo ? (values[i][j] - lo) / (hi - lo) : 0.5;
    }
  }
  return chartHost((h) => window.Viz.heatmap(h, rowNames, tasks.map((t) => t.label), values, {
    raw, rowH: 19, labelW: 168, colW: 92, colLabelH: 20, showValues: true,
    rawFmt: (v) => fmtPct(v, 1), cellFmt: (v) => v.toFixed(2),
    hint: `随机水平：${tasks.filter((t) => (rand[t.key] ?? 0) > 0)
      .map((t) => `${t.label.split(' ')[0]} ${fmtPct(rand[t.key], 0)}`).join(' · ')}`,
    tip: (i, j, rv) => {
      const m = models[i], t = tasks[j];
      const n = (m.n || {})[t.key];
      const ci = (m.ci95 || {})[t.key];
      return tipHead(m.model)
        + tipRow(stageColor(m.stage), t.label, fmtPct(rv))
        + (n ? `<div class="viz-row"><span class="viz-k">样本量</span><span class="viz-v">n=${n}</span></div>` : '')
        + (ci ? `<div class="viz-row"><span class="viz-k">95% CI 半宽</span><span class="viz-v">±${fmtPct(ci, 1)}</span></div>` : '')
        + `<div class="viz-row"><span class="viz-k">档位</span><span class="viz-v">${m.depth || '?'}</span></div>`;
    },
  }), 400);
}

/** 相对基线的差值表：每格是「该权重 − 基线」。 */
function evalDeltaTable(models, tasks, base, rand) {
  const rows = models.filter((m) => m.model !== base.model).map((m) => {
    const row = { model: m.model, stage: m.stage, is_moe: m.is_moe };
    for (const t of tasks) {
      const a = m[t.key], b = base[t.key];
      row[t.key] = (a == null || b == null) ? null : a - b;
    }
    return row;
  });
  const cols = [
    { key: 'model', label: '权重', cls: 'name', cell: (td, v, r) => {
      td.appendChild(badge(r.is_moe ? 'MoE' : 'Dense', 'mute'));
      td.appendChild(document.createTextNode(' ' + v));
    } },
    ...tasks.map((t) => ({
      key: t.key, label: t.label, num: true,
      cell: (td, v) => {
        td.classList.add('num');
        if (v == null) { td.textContent = '—'; td.classList.add('dim'); return; }
        const cls = v > 0.002 ? 'delta-pos' : v < -0.002 ? 'delta-neg' : 'delta-zero';
        const sign = v > 0 ? '+' : '';
        td.innerHTML = `<span class="${cls}">${sign}${fmtPct(v, 1)}</span>`;
      },
    })),
  ];
  return table('eval-delta', cols, rows, {});
}

/** IFEval 三档通过率：严格 / 宽松 / 指令级。只有 JSON sidecar 里有这三项。 */
function evalIfevalCard(models) {
  const rows = models.filter((m) => (m.details || {}).ifeval);
  if (!rows.length) return null;
  const keys = [
    { key: 'prompt_acc', label: '严格通过率', note: '整条 prompt 的每条指令都满足' },
    { key: 'prompt_acc_loose', label: '宽松通过率', note: '允许轻微格式偏差' },
    { key: 'inst_acc', label: '指令级通过率', note: '按单条指令计，不要求整条 prompt 全对' },
  ];
  const series = keys.map((k, i) => ({
    key: k.key, name: k.label, color: seriesColor(i),
    points: rows.map((m, idx) => ({ x: idx, y: m.details.ifeval[k.key] || 0 })),
  }));
  const host = mkEl('div');
  host.appendChild(chartHost((h) => window.Viz.lineChart(h, series, {
    height: 220, yLabel: '通过率', yTickFmt: (v) => fmtPct(v, 0),
    xTickFmt: (v) => (rows[v] || {}).model || '',
  }), 220));
  host.appendChild(legendHost(series, new Set(), () => {}));
  const skipped = rows[0].details.ifeval.n_skipped;
  return card('IFEval 指令跟随（三档口径）',
    `严格通过率是主指标；宽松与指令级用来看「差一点」还是「完全不会」。`
    + (skipped ? ` 每行跳过了 ${skipped} 道需要语言识别的题（本机没装 langdetect）。` : ''),
    host);
}

/** Agentic 工具调用拆解：只有跑过 standard 档的权重才有。 */
function evalAgenticCard(models) {
  const rows = models.filter((m) => (m.details || {}).agentic && m.details.agentic.pass_rate != null);
  if (!rows.length) return null;
  const keys = [
    { key: 'pass_rate', label: '答对率 pass_rate' },
    { key: 'tool_name_acc', label: '工具名命中率' },
    { key: 'arg_valid_rate', label: '参数合法率' },
    { key: 'valid_call_rate', label: '调用整体合法率' },
  ];
  const series = keys.map((k, i) => ({
    key: k.key, name: k.label, color: seriesColor(i),
    points: rows.map((m, idx) => ({ x: idx, y: m.details.agentic[k.key] || 0 })),
  }));
  const host = mkEl('div');
  host.appendChild(chartHost((h) => window.Viz.lineChart(h, series, {
    height: 220, yLabel: '比例', yTickFmt: (v) => fmtPct(v, 0),
    xTickFmt: (v) => (rows[v] || {}).model || '',
  }), 220));
  host.appendChild(legendHost(series, new Set(), () => {}));

  const facts = rows.map((m) => {
    const a = m.details.agentic;
    return `<div class="spotlight-fact"><div class="k">${escapeHtml(m.model)}</div>`
      + `<div class="v">${(a.turns_mean || 0).toFixed(2)} 轮 · ${(a.tool_calls_mean || 0).toFixed(2)} 次调用</div></div>`;
  }).join('');
  host.appendChild(mkEl('div', 'spotlight-facts', facts));
  return card('Agentic 工具调用拆解',
    'pass_rate 与训练控制台打的是同一个函数（trainer/algos/rl/agent_tools.py::calculate_rewards）。'
    + '工具名命中率高但参数合法率低，说明模型会「开口叫工具」但还不会填对参数',
    host);
}

/** 权重聚光灯：选一个权重看它在每个任务上相对随机水平的位置。 */
function evalSpotlight(models, tasks, rand, base) {
  const pick = models.find((m) => m.model === (state.evalFocus || base)) || models[0];
  const wrap = mkEl('div', 'spotlight-wrap');

  const items = tasks.filter((t) => pick[t.key] != null && t.key !== 'bpb').map((t) => {
    const v = pick[t.key];
    const ref = rand[t.key] ?? 0;
    return {
      label: t.label, value: v, text: fmtPct(v),
      color: ref > 0 && v < ref ? token('--danger') : stageColor(pick.stage),
      sub: ref > 0 ? `随机 ${fmtPct(ref, 0)}` : '',
    };
  });
  wrap.appendChild(chartHost((h) => window.Viz.barChart(h, items, {
    labelW: 150, rowH: 18, valueLabel: '得分',
    tickFmt: (v) => `${(v * 100).toFixed(0)}%`,
  }), 260));

  const meta = mkEl('div', 'spotlight-meta');
  const head = mkEl('div');
  const sel = mkEl('select', 'delta-select');
  for (const m of models) {
    const o = mkEl('option', null, `${m.model}（${m.stage}）`);
    o.value = m.model;
    if (m.model === pick.model) o.selected = true;
    sel.appendChild(o);
  }
  sel.addEventListener('change', () => { state.evalFocus = sel.value; rerender(); });
  head.appendChild(sel);
  meta.appendChild(head);

  const facts = [
    ['阶段', pick.stage],
    ['架构', pick.is_moe ? 'MoE' : 'Dense'],
    ['档位', pick.depth || '—'],
    ['综合', fmtPct(avgScore(pick, tasks))],
  ];
  const det = pick.details || {};
  if (det.bpb) facts.push(['BPB', `${det.bpb.bits_per_token.toFixed(3)} bit/tok`]);
  if (det.agentic) facts.push(['工具轮数', `${(det.agentic.turns_mean || 0).toFixed(2)}`]);
  meta.appendChild(mkEl('div', 'spotlight-facts', facts.map(([k, v]) =>
    `<div class="spotlight-fact"><div class="k">${escapeHtml(k)}</div><div class="v">${escapeHtml(String(v))}</div></div>`).join('')));
  wrap.appendChild(meta);
  return card('权重聚光灯', '选一个权重，看它在每个任务上离随机水平有多远（红色条 = 低于随机）', wrap);
}

/** 全量表：按阶段分组，格子带 CI 半宽。 */
function evalTable(models, tasks, rand) {
  const rows = [];
  let cur = null;
  for (const m of models) {
    if (m.stage !== cur) {
      cur = m.stage;
      rows.push({ _group: `${m.stage}（${models.filter((x) => x.stage === m.stage).length} 个）` });
    }
    rows.push({ ...m, _group: null });
  }
  const cols = [
    { key: 'model', label: '权重', cls: 'name', cell: (td, v, r) => {
      if (r._group) return;
      td.appendChild(badge(r.is_moe ? 'MoE' : 'Dense', 'mute'));
      td.appendChild(document.createTextNode(' ' + v));
      if (r.depth) td.appendChild(mkEl('span', 'ci-sub', r.depth));
    } },
    ...tasks.map((t) => ({
      key: t.key, label: t.label, num: true,
      cell: (td, v, r) => {
        td.classList.add('num');
        if (r._group) return;
        if (v == null) { td.textContent = '—'; td.classList.add('dim'); return; }
        const ref = rand[t.key] ?? 0;
        const rel = ref > 0 ? v / ref : (v > 0 ? 1.6 : 0);
        const c = rel >= 1.4 ? token('--ok') : rel >= 1.0 ? token('--ink') : token('--danger');
        const ci = (r.ci95 || {})[t.key];
        td.innerHTML = `<span style="color:${c}">${t.key === 'bpb' ? v.toFixed(3) : fmtPct(v)}</span>`
          + (ci ? `<span class="ci-sub">±${t.key === 'bpb' ? ci.toFixed(3) : fmtPct(ci, 1)}</span>` : '');
      },
    })),
  ];
  const inner = table('eval-all', cols, rows, {});
  const tb = inner.querySelector('tbody');
  [...tb.children].forEach((tr, i) => {
    if (rows[i]._group) {
      tr.className = 'group-row';
      tr.innerHTML = '';
      const td = mkEl('td', null, escapeHtml(rows[i]._group));
      td.colSpan = cols.length;
      tr.appendChild(td);
    }
  });
  const tw = mkEl('div');
  tw.appendChild(inner);
  return tw;
}
/* ======================================================================== */
/*  页面：资源与配置                                                         */
/* ======================================================================== */
/* ---------------- 起训练前检查表：配置要什么 × 机器有什么 ---------------- */

/** 配置期望的权重产物名（与 `trainer/eval.py::weight_path` 同一套命名规则）。 */
function expectedWeight(c) {
  const sw = (c.train || {}).save_weight;
  if (!sw) return null;
  const a = c.arch || {};
  return `${sw}_${a.hidden_size}${a.is_moe ? '_moe' : ''}.pth`;
}
/** `/api/models` 已发现的权重文件名集合（不含目录）。取不到就返回 null，不假装知道。 */
function knownWeights() {
  const ms = (state.models || {}).models;
  if (!Array.isArray(ms)) return null;
  const out = new Set();
  for (const m of ms) { const w = String(m.weight || ''); if (w) out.add(w.split('/').pop()); }
  return out;
}

/**
 * 「这套配置要多少显存」的实测值 —— **只报实测，不做外推**。
 *
 * 数据源是 `/api/experiments` 里各 metrics CSV 的 `gpu_mem_peak_mb`
 * （nvidia-smi 采样窗口内的最大值）。为什么不用参数量外推：本仓库实测的十几组
 * 「(batch, seq, arch) → 峰值」里，带 rollout 的 RL 与纯前反向的 SFT 根本不是
 * 同一个基函数（GRPO 的 batch=2 却要同时放 6 条采样序列），强行最小二乘会解出
 * 「token 越多越省显存」这种物理上不成立的系数。所以没有实测的配置就老实写「无实测」。
 */
function measuredPeak(d, c) {
  const sw = (c.train || {}).save_weight;
  const a = c.arch || {};
  if (sw === 'pretrain_moe') {
    const mb = ((d.pretrain || {}).moe || {}).gpu_mem_peak_mb;
    if (mb) return { mb, label: '预训练 MoE' };
  }
  const sft = {};
  for (const x of ((d.sft || {}).runs || [])) sft[x.key] = x;
  const SFT_BY_WEIGHT = { full_sft: 'sft_dense', full_sft_moe: 'sft_moe', full_sft_moe_full: 'sft_moe_full' };
  const k = SFT_BY_WEIGHT[sw];
  if (k && sft[k] && sft[k].gpu_mem_peak_mb) {
    return { mb: sft[k].gpu_mem_peak_mb, label: `SFT · ${sft[k].label}` };
  }
  // RL 配置一份 YAML 覆盖 4 个算法、峰值各不相同 → 报区间，不挑一个数冒充
  if (c.name === 'rl' || c.name === 'rl_moe') {
    const arch = a.is_moe ? 'moe' : 'dense';
    const vs = ((d.rl || {}).runs || [])
      .filter((x) => x.arch === arch && x.gpu_mem_peak_mb != null)
      .map((x) => x.gpu_mem_peak_mb);
    if (vs.length) return { mb: Math.max(...vs), lo: Math.min(...vs), label: `RL ${arch} · ${vs.length} 个算法` };
  }
  return null;
}

/**
 * 起训练前检查表：把「这套配置要什么」与「机器现在有什么」摆在一行里。
 *
 * 每一项结论都是**图标 + 文字**（`badge` 里带了 ✓ / ! / ✕），不靠颜色单独表意。
 * 数据全部来自已有的 `/api/catalog`、`/api/models`、`/api/resources`、`/api/experiments`，
 * 服务端零改动、不写任何文件。
 */
function preflightCard(d, cat, r) {
  const stage = ((cat.configs || {}).stage || [])
    .filter((c) => (c.data || {}).path || (c.train || {}).save_weight);
  if (!stage.length) return null;

  const have = knownWeights();
  const gpus = (r.gpus || []).filter((g) => g.mem_free_mb != null);
  const best = gpus.slice().sort((a, b) => b.mem_free_mb - a.mem_free_mb)[0] || null;

  const rows = stage.map((c) => {
    const dt = c.data || {}, a = c.arch || {};
    const peak = measuredPeak(d, c);
    const wf = expectedWeight(c);
    const wExists = wf ? (have ? have.has(wf) : null) : null;
    const rowsTxt = dt.exists ? rowsText(dt.rows, dt.rows_exact) : null;
    const notes = [];
    if (peak) notes.push(`实测来源：${peak.label}${peak.lo != null && peak.lo !== peak.mb ? `（${fmtInt(peak.lo)}~${fmtInt(peak.mb)}）` : ''}`);
    const algos = Object.keys(c.algo_blocks || {});
    if (algos.length) notes.push(`专属超参：${algos.join(' / ')}`);

    // 结论：数据缺 → 不可用；显存实测超过当前最大空闲 → 显存不足；其余缺项 → 需注意
    let text = '可用', cls = 'ok';
    if (dt.path && !dt.exists) { text = '不可用'; cls = 'bad'; }
    else if (peak && best && peak.mb > best.mem_free_mb) { text = '显存不足'; cls = 'bad'; }
    else if (!dt.path || !peak || (wf && !wExists)) { text = '需注意'; cls = 'warn'; }

    return {
      name: c.name, is_moe: a.is_moe, path: c.name + (a.is_moe ? ' moe' : ' dense'),
      data_ok: dt.exists ? 1 : 0, data_path: dt.path || '', data_rows: rowsTxt,
      peak_mb: peak ? peak.mb : null, free_mb: best ? best.mem_free_mb : null,
      wf: wf || '', w_exists: wExists, verdict: text, vcls: cls,
      note: notes.join('；'),
    };
  });

  const cols = [
    { key: 'name', label: '配置', cls: 'name', cell: (td, v, row) => {
      td.appendChild(document.createTextNode(String(v)));
      td.appendChild(document.createTextNode(' '));
      td.appendChild(badge(row.is_moe ? 'MoE' : 'Dense', row.is_moe ? 'acc' : 'mute'));
    } },
    { key: 'data_ok', label: '训练数据', cell: (td, _v, row) => {
      if (!row.data_path) { td.appendChild(badge('无数据段', 'mute')); return; }
      if (!row.data_ok) { td.appendChild(badge('✕ 文件不存在', 'bad')); return; }
      td.appendChild(badge(`✓ 就绪 · ${row.data_rows} 行`, 'ok'));
    } },
    { key: 'wf', label: '权重产物', cls: 'dim', cell: (td, _v, row) => {
      if (!row.wf) { td.appendChild(badge('按算法分别产出', 'mute')); return; }
      if (row.w_exists === null) { td.appendChild(badge('权重清单不可用', 'mute')); return; }
      td.appendChild(row.w_exists ? badge(`✓ ${row.wf}`, 'ok') : badge(`! 尚无 ${row.wf}（训练后生成）`, 'warn'));
    } },
    { key: 'peak_mb', label: '实测峰值显存', num: true,
      fmt: (v) => (v == null ? null : `${fmtInt(v)} MB`) },
    { key: 'free_mb', label: '当前最大空闲', num: true,
      fmt: (v) => (v == null ? null : `${fmtInt(v)} MB`) },
    { key: 'verdict', label: '结论', cell: (td, v, row) => td.appendChild(
      badge(`${row.vcls === 'ok' ? '✓' : row.vcls === 'warn' ? '!' : '✕'} ${v}`, row.vcls)) },
    { key: 'note', label: '口径备注', cls: 'dim' },
  ];

  const body = mkEl('div');
  body.appendChild(table('res-preflight', cols, rows));
  body.appendChild(mkEl('div', 'empty-note',
    best ? `当前空闲显存最大的是 GPU ${best.index}（${escapeHtml(best.name)}），空闲 ${fmtK(best.mem_free_mb)} MB —— `
      + '「实测峰值显存」一列就是与它比对；峰值来自训练 CSV 里 nvidia-smi 采样窗口内的最大值，'
      + '含同卡上的其它进程。'
      : '没有读到空闲显存，无法做显存比对。'));
  return card('起训练前检查表', '数据路径/行数来自 <code>/api/catalog</code>；权重名按 '
    + '<code>save_weight + _&lt;hidden&gt;[_moe].pth</code> 在 <code>/api/models</code> 已发现的产物里比对'
    + '（只说明磁盘上有同名文件，不代表这套配置跑过）；<b>显存一列只报实测峰值，不做参数量外推</b>'
    + ' —— 带 rollout 的 RL 与纯前反向的 SFT 不是同一个基函数，外推会给出不成立的数。',
    body);
}

function renderResources(d) {
  const r = state.resources;
  const cat = state.catalog;
  const out = [];
  if (!r) { out.push(card('正在采集资源…', null, mkEl('div', 'empty-note', '—'))); return stack(out); }

  const gpus = r.gpus || [];
  out.push(notice('本页每 5 秒自动刷新一次，数据来自 <code>nvidia-smi</code> 与 <code>/proc</code>。'
    + '训练控制台用它判断「现在有没有空卡」。', ''));

  // GPU 卡
  if (gpus.length) {
    out.push(subTitle(`GPU · ${gpus.length} 张`));
    out.push(grid('c3', gpus.map((g) => {
      const box = mkEl('div', 'gpu-card');
      const top = mkEl('div', 'gpu-top');
      top.appendChild(mkEl('span', 'idx', `GPU ${g.index}`));
      top.appendChild(mkEl('span', 'nm', escapeHtml(g.name)));
      box.appendChild(top);
      // 显存：meter（占用/总量）
      if (g.mem_total_mb) {
        const used = g.mem_used_mb ?? 0;
        box.appendChild(meterRow('显存', `${fmtInt(used)} / ${fmtInt(g.mem_total_mb)} MB`,
          used / g.mem_total_mb * 100, memColor(used / g.mem_total_mb)));
      }
      if (g.util_pct != null) {
        box.appendChild(meterRow('利用率', `${g.util_pct.toFixed(0)}%`, g.util_pct, utilColor(g.util_pct)));
      }
      const facts = mkEl('div', 'gpu-facts');
      for (const [k, v] of [['温度', g.temp_c != null ? `${g.temp_c}°C` : null],
                            ['功耗', g.power_w != null ? `${g.power_w.toFixed(0)}/${g.power_limit_w ? g.power_limit_w.toFixed(0) : '?'} W` : null],
                            ['空闲', g.mem_free_mb != null ? `${fmtK(g.mem_free_mb)} MB` : null]]) {
        if (v) facts.appendChild(mkEl('span', null, `${k} <b>${escapeHtml(v)}</b>`));
      }
      box.appendChild(facts);
      return box;
    })));
  } else {
    out.push(notice('没有检测到 GPU（nvidia-smi 不可用且 torch 未识别到 CUDA 设备）。训练会退化到 CPU。', 'warn'));
  }

  // 系统：内存 / CPU / 磁盘
  const m = r.memory || {}, cpu = r.cpu || {}, disk = r.disk || {};
  const sys = mkEl('div');
  const mw = mkEl('div', 'meter-wrap');
  if (m.total_gb) mw.appendChild(meterRow('内存', `${m.used_gb} / ${m.total_gb} GB`, m.used_pct, memColor(m.used_pct / 100)));
  if (cpu.load_pct != null) mw.appendChild(meterRow('CPU 负载', `load1 ${cpu.load1} · ${cpu.cores} 核`, cpu.load_pct, utilColor(cpu.load_pct)));
  else if (cpu.cores) mw.appendChild(meterRow('CPU 核心', `${cpu.cores} 核`, 0, token('--ink-3')));
  if (disk.total_gb) mw.appendChild(meterRow('磁盘', `剩 ${disk.free_gb} / ${disk.total_gb} GB`, disk.used_pct, memColor(disk.used_pct / 100)));
  sys.appendChild(mw);
  out.push(grid('c2', [
    card('系统负载', '内存 / CPU / 磁盘（磁盘指实验产物所在分区）', sys),
    card('软件环境', '决定能用哪些 dtype 与算子', kv([
      ['Python', (r.software || {}).python],
      ['torch', (r.software || {}).torch],
      ['CUDA', (r.software || {}).cuda],
      ['cuDNN', (r.software || {}).cudnn],
      ['bf16 支持', (r.software || {}).bf16_supported === undefined ? null : (r.software.bf16_supported ? '是' : '否')],
    ])),
  ]));

  // 起训练前检查表：把「这套配置要什么」和「机器现在有什么」摆在一起
  if (cat && r) {
    const check = preflightCard(d, cat, r);
    if (check) out.push(check);
  }

  // 正在跑的进程
  const jobs = r.jobs || [];
  const jw = mkEl('div');
  if (!jobs.length) {
    jw.appendChild(mkEl('div', 'empty-note', '当前没有 <code>trainer/train.py</code> 进程在跑 —— 可以安全地起新训练。'));
  } else {
    jw.appendChild(table('res-jobs', [
      { key: 'pid', label: 'PID', num: true },
      { key: 'algo', label: '算法', cell: (td, v) => td.appendChild(badge(v, 'acc')) },
      { key: 'run_name', label: '实验名', cls: 'name' },
      { key: 'device', label: '设备' },
      { key: 'elapsed_sec', label: '已运行', num: true, fmt: (v) => fmtDur(v) },
      { key: 'rss_mb', label: '内存 MB', num: true, fmt: (v) => fmtK(v) },
    ], jobs));
  }
  out.push(card('正在运行的训练进程', '进程按 <code>trainer/train.py</code> 识别；起新任务前请确认卡是空的', jw));

  // 配置库
  if (cat) {
    const stage = (cat.configs || {}).stage || [];
    const sweep = (cat.configs || {}).sweep || [];
    out.push(subTitle(`可用配置 · 阶段 ${stage.length} 组 / 扫描 ${sweep.length} 组`));
    out.push(grid('c3', stage.map(cfgCard)));
    if (sweep.length) {
      out.push(card('架构扫描预设', `来自 <code>${escapeHtml((cat.configs.sweep[0] || {}).path || '').replace(/\/[^/]+$/, '')}</code>`,
        grid('c3', sweep.map(cfgCard))));
    }
    const datasets = cat.datasets || [];
    if (datasets.length) {
      out.push(card('本地数据集', '扫描 <code>dataset/**/*.jsonl</code>',
        table('res-data', [
          { key: 'name', label: '文件', cls: 'name' },
          { key: 'tag', label: '用途', cell: (td, v) => v ? td.appendChild(badge(v, 'mute')) : td.appendChild(document.createTextNode('—')) },
          { key: 'size_gb', label: '大小 GB', num: true, fmt: (v) => fmtNum(v, 3) },
          { key: 'path', label: '路径', cls: 'dim' },
        ], datasets)));
    }
  }
  return stack(out);
}
function cfgCard(c) {
  const a = c.arch || {}, t = c.train || {}, dt = c.data || {};
  const box = mkEl('div', 'cfgcard');
  const top = mkEl('div', 'cfgcard-top');
  top.appendChild(mkEl('span', 'nm', escapeHtml(c.name)));
  top.appendChild(mkEl('span', 'spacer'));
  top.appendChild(badge(a.is_moe ? 'MoE' : 'Dense', a.is_moe ? 'acc' : 'mute'));
  box.appendChild(top);
  box.appendChild(mkEl('div', 'path', escapeHtml(c.path)));
  box.appendChild(mkEl('div', 'desc', escapeHtml(c.arch.summary)));
  const chips = mkEl('div', 'slot-chips');
  for (const [k, v] of Object.entries(a.slots || {})) {
    const lbl = { norm: 'norm', positional_encoding: 'pos', attention: 'attn', feedforward: 'ffn' }[k] || k;
    chips.appendChild(mkEl('span', 'slot-chip', `<span class="k">${lbl}</span><span class="v">${escapeHtml(String(v.type))}</span>`));
  }
  box.appendChild(chips);
  const facts = [];
  if (t.batch_size) facts.push(`batch ${t.batch_size}`);
  if (t.epochs) facts.push(`${t.epochs} ep`);
  if (t.learning_rate) facts.push(`lr ${sci(t.learning_rate)}`);
  if (dt.max_seq_len) facts.push(`seq ${dt.max_seq_len}`);
  if (dt.rows) facts.push(`${dt.rows_exact === false ? '≈' : ''}${fmtK(dt.rows)} 行`);
  if (dt.path) facts.push(dt.exists ? '数据就绪' : '数据缺失');
  if (facts.length) box.appendChild(mkEl('div', 'gpu-facts', facts.map((f) => `<span>${escapeHtml(f)}</span>`).join('')));
  return box;
}
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
// 状态色只在「阈值」语义下使用（高占用 = 需注意），不与分类色混用
const memColor = (f) => f >= 0.92 ? token('--critical') : f >= 0.75 ? token('--warning') : token('--accent');
const utilColor = (p) => p >= 90 ? token('--critical') : p >= 60 ? token('--warning') : token('--good');

/* ======================================================================== */
/*  页面：日志浏览                                                           */
/* ======================================================================== */
function renderLogs(d) {
  const files = d.logs || [];
  const out = [];
  if (!files.length) {
    out.push(card('没有可浏览的日志', '扫描 <code>test/log/</code> 下的 .log / .csv / .status / .json / .md。',
      mkEl('div', 'empty-note', '—')));
    return stack(out);
  }
  out.push(notice('左侧是 <code>test/log/</code> 下的原始文件，右侧按需读取（默认只取前 200 kB）。'
    + '读取接口只允许该目录下的白名单后缀，路径越界会被拒绝。', ''));

  const groups = {};
  for (const f of files) (groups[f.group] = groups[f.group] || []).push(f);

  const split = mkEl('div', 'logsplit');
  const list = mkEl('div', 'loglist');
  for (const [g, items] of Object.entries(groups)) {
    list.appendChild(mkEl('div', 'grp', `${escapeHtml(g)} · ${items.length}`));
    for (const f of items) {
      const b = mkEl('button', 'logitem',
        `<span class="nm">${escapeHtml(f.name)}</span><span class="sz">${escapeHtml(fmtBytes(f.size))}</span>`);
      b.type = 'button';
      if (state.logPath === f.path) b.setAttribute('aria-current', 'true');
      b.addEventListener('click', () => openLog(f.path));
      list.appendChild(b);
    }
  }
  split.appendChild(list);

  const view = mkEl('div', 'logview');
  const head = mkEl('div', 'logview-head');
  const pathEl = mkEl('span', 'path', escapeHtml(state.logPath || '（未选择文件）'));
  head.appendChild(pathEl);
  head.appendChild(mkEl('span', 'spacer'));
  const meta = mkEl('span', 'chip', state.logSize ? fmtBytes(state.logSize) : '');
  head.appendChild(meta);
  const reload = mkEl('button', 'btn ghost', '重新载入');
  reload.addEventListener('click', () => { if (state.logPath) openLog(state.logPath, true); });
  head.appendChild(reload);
  const copy = mkEl('button', 'btn ghost', '复制路径');
  copy.addEventListener('click', () => {
    if (!state.logPath) return;
    navigator.clipboard?.writeText(state.logPath).then(() => toast('已复制路径'), () => toast('复制失败', true));
  });
  head.appendChild(copy);
  view.appendChild(head);

  const body = mkEl('pre', 'logview-body');
  if (!state.logPath) body.appendChild(mkEl('div', 'empty-note', '从左侧选一个文件开始浏览'));
  else body.innerHTML = highlight(state.logText);
  view.appendChild(body);
  split.appendChild(view);
  out.push(split);
  pendingLogBody = body;
  return stack(out);
}
let pendingLogBody = null;

function openLog(path, force) {
  state.logPath = path;
  state.logOffset = 0;
  state.logText = '';
  fetch(`/api/logs/content?path=${encodeURIComponent(path)}&offset=0&limit=200000`)
    .then((r) => r.json())
    .then((j) => {
      if (j.error) { toast(j.error, true); return; }
      state.logText = j.text || '';
      state.logOffset = j.offset || 0;
      state.logSize = j.size || 0;
      rerender();
      requestAnimationFrame(() => { if (pendingLogBody) pendingLogBody.scrollTop = 0; });
    })
    .catch((e) => toast(String(e), true));
}

/** 日志高亮：只做「一眼扫到异常」这一步，不做语法着色（那会变成噪音）。 */
function highlight(text) {
  if (!text) return '';
  return escapeHtml(text)
    .replace(/^(\s*)(Traceback \(most recent call last\):.*)$/gm, '$1<span class="hl-err">$2</span>')
    .replace(/\b(Error|Exception|FAILED|Killed|OOM|CUDA out of memory)\b/g, '<span class="hl-err">$1</span>')
    .replace(/\b(Warning|WARN|deprecated)\b/g, '<span class="hl-warn">$1</span>')
    .replace(/\b(OK|success|完成)\b/g, '<span class="hl-ok">$1</span>');
}

/* ======================================================================== */
/*  路由与生命周期                                                           */
/* ======================================================================== */
const PAGES = {
  library:  { title: '实验库',     sub: (d) => `${(d.eval.models || []).length} 个权重 · ${(d.logs || []).length} 份日志`, render: renderLibrary },
  pipeline: { title: '全流程',     sub: () => 'tokenizer → 预训练 → 对齐 → 强化 → 评测 → 部署', render: renderPipeline },
  runs:     { title: '训练登记',   sub: (d) => `${(d.runs && d.runs.total) || 0} 次 run`, render: renderRuns },
  pretrain: { title: '预训练',     sub: () => 'Dense / MoE 收敛与 MoE 健康度', render: renderPretrain },
  sft:      { title: '监督微调',   sub: (d) => `${((d.sft || {}).runs || []).length} 组对照`, render: renderSft },
  rl:       { title: '强化学习',   sub: () => '4 算法 × 2 架构', render: renderRl },
  sweep:    { title: '架构对比',   sub: (d) => `${(((d.sweep || {}).configs) || []).length} 组配置`, render: renderSweep },
  eval:     { title: '标准评测',   sub: (d) => `${((d.eval || {}).tasks || []).length} 任务`, render: renderEval },
  resources:{ title: '资源与配置', sub: () => '实时占用 + 可用配置', render: renderResources },
  logs:     { title: '日志浏览',   sub: (d) => `${(d.logs || []).length} 个文件`, render: renderLogs },
};

function rerender() {
  pendingCharts = [];
  const host = document.getElementById('lab-scroll');
  const page = PAGES[state.page];
  host.innerHTML = '';
  if (!state.data) { host.appendChild(card('正在读取实验数据…', null, mkEl('div', 'empty-note', '只读扫描，不会写入任何文件'))); return; }
  try {
    host.appendChild(page.render(state.data));
  } catch (e) {
    host.appendChild(card('这一页渲染出错', `${state.page}`,
      mkEl('pre', 'cmdbox err', `${e && e.stack ? e.stack : e}`)));
  }
  // 图表是在挂载前构建的：那时 host.clientWidth 还是 0，各图只能退回默认宽度
  // （折线图会被 preserveAspectRatio 拉到满宽，热力图会缩成中间一小条）。
  // 挂载后用真实宽度重建一遍 —— 上面 pendingCharts 就是为此积累的。
  requestAnimationFrame(() => {
    pendingCharts.forEach((f) => f());
    document.querySelectorAll('.tile-spark').forEach((h) => { h.dataset.done = '1'; });
  });
}

function showPage(name, push) {
  if (!PAGES[name]) name = 'library';
  state.page = name;
  document.querySelectorAll('.lab-nav button').forEach((b) => {
    if (b.dataset.page === name) b.setAttribute('aria-current', 'true');
    else b.removeAttribute('aria-current');
  });
  const page = PAGES[name];
  document.getElementById('lab-title').textContent = page.title;
  document.getElementById('lab-sub').textContent = state.data ? page.sub(state.data) : '—';
  if (push !== false) history.replaceState(null, '', `#${name}`);
  document.getElementById('lab-scroll').scrollTop = 0;
  rerender();
  if (name === 'resources') startResourcePoll(); else stopResourcePoll();
}

/* 资源轮询（只在资源页跑，避免无谓请求） */
let resTimer = null;
function startResourcePoll() {
  stopResourcePoll();
  const tick = () => fetch('/api/resources').then((r) => r.json()).then((j) => {
    state.resources = j;
    if (state.page === 'resources') rerender();
    updateLive();
  }).catch(() => {});
  tick();
  resTimer = setInterval(tick, 5000);
}
function stopResourcePoll() { if (resTimer) { clearInterval(resTimer); resTimer = null; } }

/** 顶栏的实时摘要：GPU 占用 + 正在跑的任务数。 */
function updateLive() {
  const el = document.getElementById('lab-live');
  if (!el) return;
  const r = state.resources;
  if (!r) { el.textContent = '资源不可用'; return; }
  const g = (r.gpus || [])[0];
  const jobs = (r.jobs || []).length;
  const parts = [];
  if (g && g.util_pct != null) parts.push(`GPU ${g.util_pct.toFixed(0)}%`);
  if (g && g.mem_used_pct != null) parts.push(`显存 ${g.mem_used_pct.toFixed(0)}%`);
  if (r.memory && r.memory.used_pct != null) parts.push(`内存 ${r.memory.used_pct.toFixed(0)}%`);
  parts.push(jobs ? `${jobs} 个训练在跑` : '无训练进程');
  el.innerHTML = `<span class="live-dot"></span>${escapeHtml(parts.join(' · '))}`;
}

/* ------------------------------- toast --------------------------------- */
function toast(msg, bad) {
  let wrap = document.querySelector('.toast-wrap');
  if (!wrap) { wrap = mkEl('div', 'toast-wrap'); document.body.appendChild(wrap); }
  const t = mkEl('div', `toast${bad ? ' bad' : ' ok'}`, escapeHtml(msg));
  wrap.appendChild(t);
  setTimeout(() => { t.style.opacity = '0'; setTimeout(() => t.remove(), 260); }, 2600);
}

/* ------------------------------- 主题 ---------------------------------- */
function applyTheme(t) {
  document.documentElement.dataset.theme = t;
  try { localStorage.setItem('mm-theme', t); } catch (e) { /* 隐私模式下忽略 */ }
  rerender();          // 颜色从 CSS 变量读，主题变了必须重画
}
function initTheme() {
  let t = 'light';
  try { t = localStorage.getItem('mm-theme') || (matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'); } catch (e) { /* ignore */ }
  document.documentElement.dataset.theme = t;
}

/* ------------------------------- 启动 ---------------------------------- */
function load() {
  return fetch('/api/experiments').then((r) => r.json()).then((j) => {
    state.data = j;
    const n = (j.logs || []).length;
    document.getElementById('lab-stat').innerHTML =
      `${escapeHtml((j.meta || {}).generated_at || '')}<br>${escapeHtml((j.meta || {}).artifact_root || '')}`;
    const nb = document.getElementById('nav-runs');
    if (nb) nb.textContent = String((j.runs && j.runs.total) || '');
    const nl = document.getElementById('nav-logs');
    if (nl) nl.textContent = String(n);
    const sub = document.getElementById('lab-sub');
    if (sub && PAGES[state.page]) sub.textContent = PAGES[state.page].sub(j);
  });
}

function boot() {
  initTheme();
  document.getElementById('lab-nav').addEventListener('click', (ev) => {
    const b = ev.target.closest('button[data-page]');
    if (b) showPage(b.dataset.page);
  });
  document.getElementById('lab-theme').addEventListener('click', () => {
    applyTheme(document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark');
  });
  const refresh = document.getElementById('lab-refresh');
  refresh.addEventListener('click', () => {
    refresh.classList.add('loading');
    refresh.querySelector('.btn-label, span, svg');
    Promise.all([
      load(),
      fetch('/api/catalog').then((r) => r.json()).then((j) => { state.catalog = j; }).catch(() => {}),
      fetch('/api/models').then((r) => r.json()).then((j) => { state.models = j; }).catch(() => {}),
      fetch('/api/resources').then((r) => r.json()).then((j) => { state.resources = j; updateLive(); }).catch(() => {}),
    ]).then(() => { showPage(state.page, false); toast('数据已刷新'); })
      .catch((e) => toast(String(e), true))
      .finally(() => refresh.classList.remove('loading'));
  });
  const hash = (location.hash || '').replace('#', '');
  showPage(PAGES[hash] ? hash : 'library', false);
  Promise.all([
    load(),
    fetch('/api/catalog').then((r) => r.json()).then((j) => { state.catalog = j; }).catch(() => {}),
    fetch('/api/models').then((r) => r.json()).then((j) => { state.models = j; }).catch(() => {}),
    fetch('/api/resources').then((r) => r.json()).then((j) => { state.resources = j; updateLive(); }).catch(() => {}),
  ]).then(() => showPage(state.page, false))
    .catch((e) => {
      document.getElementById('lab-scroll').innerHTML = '';
      document.getElementById('lab-scroll').appendChild(
        card('无法读取实验数据', null, mkEl('pre', 'cmdbox err', String(e))));
    });
}
window.addEventListener('hashchange', () => showPage((location.hash || '').replace('#', '') || 'library', false));
document.addEventListener('DOMContentLoaded', boot);
})();
