/* docs/play/*.html 共通：マウス（またはタッチ）でチップを空欄へ入れる・ブロックを並べる．
 *
 *   Play.slots(levelEl, chips)    空欄（.slot[data-a]）と仕分けの箱（.bin[data-bin]）に札を入れる
 *   Play.parsons(levelEl, spec)   コードの行を並べ，字下げを合わせる
 *   Play.choice(levelEl)          選択問題（button[data-ok] が正解）
 *
 * 操作：ドラッグして離す．または，札をクリックして選び → 入れたい所をクリック．
 */
(function () {
  "use strict";

  function store(key, value) {
    try {
      if (value === undefined) return localStorage.getItem(key);
      localStorage.setItem(key, value);
    } catch (e) { /* 保存できない環境でも遊べる */ }
    return null;
  }

  function shuffle(list) {
    const a = list.slice();
    for (let i = a.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [a[i], a[j]] = [a[j], a[i]];
    }
    return a;
  }

  function markCleared(level, cleared) {
    level.classList.toggle("is-cleared", cleared);
    if (cleared) store("play:" + location.pathname + ":" + level.id, "1");
  }

  function restoreCleared(level) {
    if (store("play:" + location.pathname + ":" + level.id) === "1") level.classList.add("is-cleared");
  }

  function controls(level, onCheck, onReset) {
    const box = document.createElement("div");
    box.className = "controls";
    box.innerHTML = '<button class="primary">答え合わせ</button><button>やり直し</button><span class="result"></span>';
    const hints = document.createElement("ul");
    hints.className = "hints";
    level.appendChild(box);
    level.appendChild(hints);
    const [check, reset] = box.querySelectorAll("button");
    const result = box.querySelector(".result");
    check.addEventListener("click", () => {
      const r = onCheck();
      result.textContent = r.ok === r.total ? `全問正解！（${r.total} / ${r.total}）` : `${r.ok} / ${r.total} 正解．赤い所を直してもう一度`;
      result.classList.toggle("all", r.ok === r.total);
      hints.innerHTML = "";
      [...new Set(r.hints)].forEach((h) => {
        const li = document.createElement("li");
        li.textContent = h;
        hints.appendChild(li);
      });
      markCleared(level, r.ok === r.total);
    });
    reset.addEventListener("click", () => {
      onReset();
      result.textContent = "";
      hints.innerHTML = "";
    });
    return () => { result.textContent = ""; hints.innerHTML = ""; };
  }

  /* ドラッグとクリックの共通処理．動かしたら drag，動かさずに離したら click． */
  function pointerize(el, handlers) {
    el.addEventListener("pointerdown", (ev) => {
      if (ev.button !== 0 || ev.target.closest("button")) return;
      ev.preventDefault();
      const startX = ev.clientX, startY = ev.clientY;
      const rect = el.getBoundingClientRect();
      let ghost = null;
      const move = (e) => {
        if (!ghost && Math.hypot(e.clientX - startX, e.clientY - startY) > 5) {
          ghost = el.cloneNode(true);
          ghost.classList.add("ghost");
          ghost.classList.remove("selected", "ok", "ng");
          ghost.style.width = rect.width + "px";
          document.body.appendChild(ghost);
          el.classList.add("dragging");
        }
        if (ghost) {
          ghost.style.left = e.clientX - (startX - rect.left) + "px";
          ghost.style.top = e.clientY - (startY - rect.top) + "px";
          handlers.over && handlers.over(e.clientX, e.clientY);
        }
      };
      const up = (e) => {
        document.removeEventListener("pointermove", move);
        document.removeEventListener("pointerup", up);
        if (ghost) {
          ghost.remove();
          el.classList.remove("dragging");
          handlers.drop(document.elementFromPoint(e.clientX, e.clientY), e.clientX, e.clientY);
        } else {
          handlers.click && handlers.click();
        }
      };
      document.addEventListener("pointermove", move);
      document.addEventListener("pointerup", up);
    });
  }

  /* ------------------------------------------------------------------ 空欄と箱 */
  function slots(level, chips) {
    const pool = document.createElement("div");
    pool.className = "pool";
    level.querySelector(".pool-here").appendChild(pool);
    const targets = [...level.querySelectorAll(".slot, .bin")];
    let selected = null;

    const clearMarks = () => {
      level.querySelectorAll(".ok, .ng").forEach((e) => e.classList.remove("ok", "ng"));
      clearMsg();
    };
    const select = (chip) => {
      if (selected) selected.classList.remove("selected");
      selected = chip === selected ? null : chip;
      if (selected) selected.classList.add("selected");
    };
    const place = (chip, target) => {
      if (!target) return;
      if (target.classList.contains("slot") && target.firstElementChild && target.firstElementChild !== chip) {
        pool.appendChild(target.firstElementChild);   // 入っていた札は置き場へ戻す
      }
      target.appendChild(chip);
      select(null);
      clearMarks();
    };
    const targetAt = (node) => node && (node.closest(".slot, .bin, .pool") || null);
    const inLevel = (t) => t && level.contains(t);
    let hovered = null;

    const build = () => {
      pool.innerHTML = "";
      targets.forEach((t) => t.querySelectorAll(".chip").forEach((c) => c.remove()));
      shuffle(chips).forEach((spec) => {
        const chip = document.createElement("span");
        chip.className = "chip" + (spec.code ? " code" : "");
        chip.textContent = spec.t;
        chip.dataset.v = spec.v || spec.t;
        if (spec.bin) chip.dataset.bin = spec.bin;
        if (spec.hint) chip.dataset.hint = spec.hint;
        pointerize(chip, {
          click: () => select(chip),
          over: (x, y) => {
            const t = targetAt(document.elementFromPoint(x, y));
            if (hovered && hovered !== t) hovered.classList.remove("hover");
            hovered = inLevel(t) && t !== pool ? t : null;
            if (hovered) hovered.classList.add("hover");
          },
          drop: (node) => {
            if (hovered) hovered.classList.remove("hover");
            hovered = null;
            const t = targetAt(node);
            if (inLevel(t)) place(chip, t);
          },
        });
        pool.appendChild(chip);
      });
    };

    [...targets, pool].forEach((t) => t.addEventListener("click", (e) => {
      if (selected && !e.target.closest(".chip")) place(selected, t);
    }));

    const clearMsg = controls(level, () => {
      let ok = 0, total = 0;
      const hints = [];
      level.querySelectorAll(".slot").forEach((s) => {
        total++;
        const c = s.querySelector(".chip");
        const answers = s.dataset.a.split("|");
        const good = c && answers.includes(c.dataset.v);
        s.classList.add(good ? "ok" : "ng");
        if (good) ok++;
        else if (s.dataset.hint) hints.push(s.dataset.hint);
      });
      level.querySelectorAll(".chip[data-bin]").forEach((c) => {
        total++;
        const bin = c.closest(".bin");
        const good = bin && bin.dataset.bin === c.dataset.bin;
        if (bin) c.classList.add(good ? "ok" : "ng");
        if (good) ok++;
        else if (bin && c.dataset.hint) hints.push(c.dataset.hint);
      });
      if (level.querySelectorAll(".bin").length && pool.querySelector(".chip[data-bin]")) hints.push("置き場にまだ札が残っている");
      return { ok, total, hints };
    }, () => { build(); clearMarks(); markCleared(level, false); });

    build();
    restoreCleared(level);
  }

  /* ------------------------------------------------------------------ 行を並べる */
  function parsons(level, spec) {
    const wrap = level.querySelector(".parsons");
    wrap.innerHTML = '<div class="plist pool-list"><div class="ttl">ブロック置き場（クリックで右の最後に入る）</div></div>' +
                     '<div class="plist answer"><div class="ttl"></div><div class="pbody"></div></div>';
    const poolList = wrap.querySelector(".pool-list");
    const answer = wrap.querySelector(".answer");
    const body = answer.querySelector(".pbody");   // ブロックが入る所（外枠の行は固定）
    const frame = spec.frame || { head: [], tail: [], indent: null };   // indent を決めると字下げは自動
    answer.querySelector(".ttl").textContent = frame.indent === null
      ? "ここに上から順に並べる（◀ ▶ で字下げ，× で戻す）" : "外枠の中に上から順に並べる（× で戻す）";
    const fixedLine = (t) => {
      const d = document.createElement("div");
      d.className = "pfixed";
      d.textContent = t;
      return d;
    };
    frame.head.forEach((t) => answer.insertBefore(fixedLine(t), body));
    frame.tail.forEach((t) => answer.appendChild(fixedLine(t)));
    if (frame.indent !== null) wrap.classList.add("no-indent");
    const line = document.createElement("div");
    line.className = "drop-line";

    const setIndent = (b, n) => {
      b.dataset.indent = Math.max(0, Math.min(4, n));
      b.querySelector(".txt").style.paddingLeft = b.dataset.indent * 2 + "ch";
    };
    const clearMarks = () => {
      wrap.querySelectorAll(".ok, .ng").forEach((e) => e.classList.remove("ok", "ng"));
      clearMsg();
      if (spec.onChange) spec.onChange([...body.querySelectorAll(".pblock")].map((x) => x.dataset.t));   // 並べ替えるたびに知らせる
    };
    const insertIndex = (y) => {
      const blocks = [...body.querySelectorAll(".pblock:not(.dragging)")];
      const i = blocks.findIndex((b) => {
        const r = b.getBoundingClientRect();
        return y < r.top + r.height / 2;
      });
      return i === -1 ? blocks.length : i;
    };
    const putAt = (b, idx) => {
      const blocks = [...body.querySelectorAll(".pblock")].filter((x) => x !== b);
      if (idx >= blocks.length) body.appendChild(b);
      else body.insertBefore(b, blocks[idx]);
      if (frame.indent !== null) setIndent(b, frame.indent);
      clearMarks();
    };
    const back = (b) => { setIndent(b, 0); poolList.appendChild(b); clearMarks(); };

    const make = (text) => {
      const b = document.createElement("div");
      b.className = "pblock";
      b.dataset.t = text;
      b.innerHTML = '<span class="txt"></span><span class="ind"><button title="字下げを浅く">◀</button><button title="字下げを深く">▶</button><button title="置き場へ戻す">×</button></span>';
      b.querySelector(".txt").textContent = text;
      const [l, r, x] = b.querySelectorAll("button");
      l.addEventListener("click", () => { setIndent(b, +b.dataset.indent - 1); clearMarks(); });
      r.addEventListener("click", () => { setIndent(b, +b.dataset.indent + 1); clearMarks(); });
      x.addEventListener("click", () => back(b));
      setIndent(b, 0);
      pointerize(b, {
        click: () => { if (b.parentElement === poolList) putAt(b, Infinity); },
        over: (x2, y) => {
          const over = document.elementFromPoint(x2, y);
          if (over && answer.contains(over)) {
            const blocks = [...body.querySelectorAll(".pblock:not(.dragging)")];
            const i = insertIndex(y);
            if (i >= blocks.length) body.appendChild(line); else body.insertBefore(line, blocks[i]);
          } else line.remove();
        },
        drop: (node, x2, y) => {
          const inAnswer = node && (node === line || answer.contains(node));   // 線の上で離しても並びに入れる
          line.remove();
          if (inAnswer) putAt(b, insertIndex(y));
          else if (node && poolList.contains(node)) back(b);
        },
      });
      return b;
    };

    const build = () => {
      wrap.querySelectorAll(".pblock").forEach((b) => b.remove());
      shuffle(spec.lines.map((l) => l.t).concat(spec.distractors || [])).forEach((t) => poolList.appendChild(make(t)));
    };

    const clearMsg = controls(level, () => {
      const blocks = [...body.querySelectorAll(".pblock")];
      let ok = 0;
      const hints = [];
      blocks.forEach((b, i) => {
        const want = spec.lines[i];
        const goodText = want && want.t === b.dataset.t;
        const good = goodText && want.i === +b.dataset.indent;
        b.classList.add(good ? "ok" : "ng");
        if (good) ok++;
        else if (goodText) hints.push(`${i + 1} 行目は字下げが違う（ループやwithの中に入る行は1段深い）`);
        else if ((spec.distractors || []).includes(b.dataset.t)) hints.push(`「${b.dataset.t.trim()}」は使わないブロック`);
      });
      if (blocks.length < spec.lines.length) hints.push(`あと ${spec.lines.length - blocks.length} 行足りない`);
      (spec.hints || []).forEach((h) => { if (ok < spec.lines.length) hints.push(h); });
      return { ok: blocks.length > spec.lines.length ? Math.min(ok, spec.lines.length - 1) : ok, total: spec.lines.length, hints };
    }, () => { build(); clearMarks(); markCleared(level, false); });

    build();
    if (spec.onChange) spec.onChange([]);
    restoreCleared(level);
  }

  /* ------------------------------------------------------------------ 選択問題 */
  function choice(level) {
    const buttons = [...level.querySelectorAll(".choices button")];
    const explain = level.querySelector(".explain");
    buttons.forEach((b) => b.addEventListener("click", () => {
      buttons.forEach((x) => x.classList.remove("ok", "ng"));
      const good = b.hasAttribute("data-ok");
      b.classList.add(good ? "ok" : "ng");
      explain.classList.toggle("show", good);
      if (!good) explain.classList.remove("show");
      markCleared(level, good);
    }));
    restoreCleared(level);
  }

  window.Play = { slots, parsons, choice };
})();
