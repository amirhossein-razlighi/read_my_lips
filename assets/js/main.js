/* Read My Lips project page: players, transcript panels with per-word strips, baseline comparisons, selection explorer,
   result charts, section progress nav. Data: window.RML_AUTHORS (data/authors.js), window.RML_DATA (data/samples.js). */
(function () {
  "use strict";
  var D = window.RML_DATA || { cards: [], babble: [] };
  var A = window.RML_AUTHORS || { authors: [], affiliations: [] };
  var DUR = 4.8;
  var players = [];

  function $(s, r) { return (r || document).querySelector(s); }
  function $$(s, r) { return Array.prototype.slice.call((r || document).querySelectorAll(s)); }
  function h(tag, attrs, kids) {
    var e = document.createElement(tag);
    if (attrs) Object.keys(attrs).forEach(function (k) {
      var v = attrs[k];
      if (v === null || v === undefined || v === false) return;
      if (k === "class") e.className = v;
      else if (k === "html") e.innerHTML = v;
      else if (k === "text") e.textContent = v;
      else if (k.slice(0, 2) === "on") e.addEventListener(k.slice(2), v);
      else e.setAttribute(k, v === true ? "" : v);
    });
    (kids || []).forEach(function (c) { if (c) e.appendChild(typeof c === "string" ? document.createTextNode(c) : c); });
    return e;
  }
  function esc(s) { return String(s).replace(/[&<>"]/g, function (c) { return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]; }); }
  var byId = {};
  D.cards.forEach(function (c) { byId[c.id] = c; });
  function kindOf(key, chosen) {
    if (key === "truth") return "orig";
    if (key === "rml" || key === "ours" || (chosen && key === chosen)) return "ours";
    if (key === "input") return "input";
    return "other";
  }

  /* ---------------------------------------------------------------- authors */
  (function authors() {
    var el = $("#authors"), af = $("#affils");
    if (!el) return;
    el.innerHTML = A.authors.map(function (a) {
      var n = a.url ? '<a href="' + esc(a.url) + '">' + esc(a.name) + "</a>" : esc(a.name);
      return '<span class="au">' + n + "<sup>" + (a.affil || []).join(",") + "</sup></span>";
    }).join(" ");
    af.innerHTML = A.affiliations.map(function (x, i) { return '<span class="af"><sup>' + (i + 1) + "</sup>" + esc(x) + "</span>"; }).join(" ");
    if (A.note) { var n = $("#affilNote"); n.textContent = A.note; n.hidden = false; }
  })();

  /* ---------------------------------------------------------------- theme (persists; storage may be unavailable) */
  (function theme() {
    var b = $("#themeToggle");
    function current() {
      var t = document.documentElement.getAttribute("data-theme");
      if (t) return t;
      return window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
    }
    b.addEventListener("click", function () {
      var next = current() === "dark" ? "light" : "dark";
      document.documentElement.setAttribute("data-theme", next);
      try { localStorage.setItem("rml-theme", next); } catch (e) {}
    });
  })();

  /* ---------------------------------------------------------------- header: section progress */
  (function nav() {
    var n = $(".nav"), cur = $("#navCurrent");
    var ids = $$(".secnav a").map(function (a) { return a.getAttribute("data-sec"); });
    var secs = ids.map(function (id) { return document.getElementById(id); });
    var fills = ids.map(function (id) { return $$('[data-sec="' + id + '"] b'); });
    var links = $$(".secnav a");
    var ticking = false;
    function update() {
      ticking = false;
      n.classList.toggle("scrolled", window.scrollY > 8);
      var y = window.scrollY + n.offsetHeight + 1, ya = window.scrollY + window.innerHeight * 0.3, active = -1;
      secs.forEach(function (s, i) {
        if (!s) return;
        var top = s.offsetTop, hgt = s.offsetHeight;
        var p = Math.max(0, Math.min(1, (y - top) / hgt));
        if (window.innerHeight + window.scrollY >= document.body.scrollHeight - 2) p = 1;
        fills[i].forEach(function (b) { b.style.width = (p * 100).toFixed(1) + "%"; });
        if (ya >= top && ya < top + hgt) active = i;
      });
      links.forEach(function (a, i) { a.classList.toggle("active", i === active); });
      if (cur) cur.textContent = active >= 0 ? links[active].textContent : "";
    }
    window.addEventListener("scroll", function () { if (!ticking) { ticking = true; requestAnimationFrame(update); } }, { passive: true });
    window.addEventListener("resize", update);
    update();
  })();

  /* ---------------------------------------------------------------- lazy thumbnails */
  var thumbIO = "IntersectionObserver" in window ? new IntersectionObserver(function (es) {
    es.forEach(function (e) {
      if (e.isIntersecting) { var t = e.target; t.style.backgroundImage = "url('" + t.dataset.bg + "')"; thumbIO.unobserve(t); }
    });
  }, { rootMargin: "400px 0px" }) : null;
  function lazyBg(el, url) {
    if (!url) return;
    if (!thumbIO) { el.style.backgroundImage = "url('" + url + "')"; return; }
    el.dataset.bg = url; thumbIO.observe(el);
  }

  /* ---------------------------------------------------------------- player */
  var ICON_PLAY = '<svg viewBox="0 0 24 24" class="i-play"><path d="M7 4.5v15l13-7.5z"/></svg>';
  var ICON_PAUSE = '<svg viewBox="0 0 24 24" class="i-pause"><path d="M6.5 4.5h4v15h-4zM13.5 4.5h4v15h-4z"/></svg>';
  function fmt(t) { return Math.max(0, t || 0).toFixed(1) + " s"; }

  /* tracks: [{key, file, label, kind}] ; every clip sits in the same 16:9 box (object-fit: contain) */
  function Player(opts) {
    var self = this;
    this.tracks = {};
    (opts.tracks || []).forEach(function (t) { if (t.file) self.tracks[t.key] = t; });
    this.key = null;
    this.pending = 0;
    this.onchange = opts.onchange || function () {};
    this.video = h("video", { playsinline: true, "webkit-playsinline": true, preload: "none", "aria-label": opts.label || "Sample video" });
    var poster = opts.poster ? h("img", { class: "poster", src: opts.poster, alt: "", loading: "lazy", decoding: "async" }) : null;
    this.now = h("span", { class: "now", "aria-hidden": "true" });
    var big = h("button", { class: "bigplay", type: "button", "aria-label": "Play", html: '<svg viewBox="0 0 24 24"><path d="M7 4.5v15l13-7.5z"/></svg>' });
    var screen = h("div", { class: "screen" }, [this.video, poster, this.now, big]);
    this.pbtn = h("button", { class: "pbtn", type: "button", "aria-label": "Play or pause", html: ICON_PLAY + ICON_PAUSE });
    this.prog = h("div", { class: "prog" });
    this.head = h("div", { class: "head" });
    var track = h("div", { class: "track" });
    (opts.bands || []).forEach(function (b) {
      track.appendChild(h("div", { class: "band", style: "left:" + (100 * b[0] / DUR) + "%;width:" + (100 * (b[1] - b[0]) / DUR) + "%" }));
    });
    track.appendChild(this.prog);
    this.timeline = h("div", { class: "timeline", role: "slider", tabindex: "0", "aria-label": "Seek", "aria-valuemin": "0", "aria-valuemax": String(DUR), "aria-valuenow": "0" }, [track, this.head]);
    if (opts.bands && opts.bands.length) {
      var b0 = opts.bands[0];
      this.timeline.appendChild(h("span", { class: "bandlabel", style: "left:" + (100 * (b0[0] + b0[1]) / 2 / DUR) + "%", text: opts.bandLabel || "silenced" }));
    }
    this.time = h("span", { class: "ptime", text: "0.0 s" });
    this.el = h("div", { class: "player" }, [screen, h("div", { class: "bar" }, [this.pbtn, this.timeline, this.time])]);

    var v = this.video;
    function toggle() { if (v.paused || v.ended) self.play(); else v.pause(); }
    big.addEventListener("click", toggle);
    this.pbtn.addEventListener("click", toggle);
    v.addEventListener("click", toggle);
    v.addEventListener("playing", function () { self.el.classList.add("shown"); self.el.classList.remove("loading"); });
    v.addEventListener("seeked", function () { if (v.readyState >= 2) self.el.classList.add("shown"); });
    v.addEventListener("play", function () {
      self.el.classList.add("playing", "started");
      players.forEach(function (p) { if (p !== self && !p.video.paused) p.video.pause(); });
      self.tick();
    });
    v.addEventListener("pause", function () { self.el.classList.remove("playing"); });
    v.addEventListener("ended", function () { self.el.classList.remove("playing"); });
    v.addEventListener("timeupdate", function () { self.draw(); });
    v.addEventListener("waiting", function () { self.el.classList.add("loading"); });
    function seekFromEvent(e) {
      var r = self.timeline.getBoundingClientRect();
      self.seek(Math.min(1, Math.max(0, (e.clientX - r.left) / r.width)) * self.dur());
    }
    var dragging = false;
    this.timeline.addEventListener("pointerdown", function (e) { dragging = true; self.timeline.setPointerCapture(e.pointerId); seekFromEvent(e); });
    this.timeline.addEventListener("pointermove", function (e) { if (dragging) seekFromEvent(e); });
    this.timeline.addEventListener("pointerup", function () { dragging = false; });
    this.timeline.addEventListener("keydown", function (e) {
      if (e.key === "ArrowRight") { self.seek((v.currentTime || 0) + 0.25); e.preventDefault(); }
      else if (e.key === "ArrowLeft") { self.seek((v.currentTime || 0) - 0.25); e.preventDefault(); }
      else if (e.key === " " || e.key === "Enter") { toggle(); e.preventDefault(); }
    });
    players.push(this);
    if (opts.initial) this.set(opts.initial);
  }
  Player.prototype.dur = function () { var d = this.video.duration; return isFinite(d) && d > 0 ? d : DUR; };
  Player.prototype.seek = function (t) {
    t = Math.max(0, Math.min(this.dur() - 0.02, t));
    if (!this.video.getAttribute("src")) { this.pending = t; this.drawAt(t); return; }
    try { this.video.currentTime = t; } catch (e) {}
    this.drawAt(t);
  };
  Player.prototype.drawAt = function (t) {
    var p = 100 * Math.min(1, t / this.dur());
    this.prog.style.width = p + "%";
    this.head.style.left = p + "%";
    this.time.textContent = fmt(t);
    this.timeline.setAttribute("aria-valuenow", t.toFixed(2));
  };
  Player.prototype.draw = function () { this.drawAt(this.video.currentTime || 0); };
  Player.prototype.tick = function () {
    var self = this;
    if (self._raf) return;
    (function loop() {
      self.draw();
      self._raf = !self.video.paused && !self.video.ended ? requestAnimationFrame(loop) : null;
    })();
  };
  Player.prototype.play = function () {
    var v = this.video;
    if (!v.getAttribute("src")) {
      var t = this.tracks[this.key];
      if (!t) return;
      v.src = t.file;
      v.preload = "auto";
      var at = this.pending || 0;
      if (at > 0) v.addEventListener("loadedmetadata", function () { try { v.currentTime = at; } catch (e) {} }, { once: true });
    }
    if (v.ended) { try { v.currentTime = 0; } catch (e) {} }
    var p = v.play();
    if (p && p.catch) p.catch(function () {});
    this.el.classList.add("started");
  };
  /* switch the soundtrack, keeping the playback position and the play state */
  Player.prototype.set = function (key) {
    var t = this.tracks[key];
    if (!t) return false;
    var v = this.video, had = !!v.getAttribute("src");
    var time = had ? (v.currentTime || 0) : this.pending || 0, playing = had && !v.paused && !v.ended;
    this.key = key;
    this.now.textContent = t.label || key;
    this.el.setAttribute("data-kind", t.kind === "input" ? "other" : t.kind || "other");
    if (had && v.getAttribute("src") !== t.file) {
      v.preload = "auto";
      v.src = t.file;
      v.addEventListener("loadedmetadata", function () {
        try { v.currentTime = Math.min(time, (isFinite(v.duration) ? v.duration : DUR) - 0.03); } catch (e) {}
        if (playing) { var p = v.play(); if (p && p.catch) p.catch(function () {}); }
      }, { once: true });
      v.load();
    }
    this.onchange(key);
    return true;
  };

  /* ---------------------------------------------------------------- transcripts */
  /* marks: [{w, op: ok|sub|ins|del|gap, s: inside the damaged span}]
     opt.soft: on span cards, errors outside the span (often clipped edge words) are muted and edge deletions hidden */
  function wordsHTML(marks, opt) {
    opt = opt || {};
    if (!marks) return null;
    if (!marks.length) return "";
    var out = [], i = 0;
    while (i < marks.length) {
      var m = marks[i];
      if (m.op === "gap" || m.op === "del") {
        var j = i, op = m.op, inSpan = 0;
        while (j < marks.length && marks[j].op === op) { inSpan += marks[j].s; j++; }
        var n = j - i;
        if (op === "gap") {
          out.push('<span class="gap" style="width:' + Math.min(9, 1.6 + 0.75 * (n - 1)).toFixed(2) + 'em" title="' + n + " silenced word" + (n > 1 ? "s" : "") + '"></span>');
        } else if (!(opt.soft && !inSpan)) {
          out.push('<span class="miss">' + n + " word" + (n > 1 ? "s" : "") + " missing</span>");
        }
        i = j;
        continue;
      }
      var cls = ["w"];
      if (m.s) cls.push("span");
      if (m.op === "sub" || m.op === "ins") cls.push(opt.soft && !m.s ? "soft" : "err");
      out.push('<span class="' + cls.join(" ") + '">' + esc(m.w) + "</span>");
      i++;
    }
    return out.join(" ");
  }
  /* one block per reference word: correct (filled), wrong (red), missing (outlined); insertions count as errors only */
  function stripHTML(marks, kind) {
    if (!marks || !marks.length) return "";
    var html = marks.filter(function (m) { return m.op !== "ins"; }).map(function (m) {
      var cls = m.op === "ok" ? (kind === "input" ? "in" : "ok") : m.op === "sub" ? "sub" : m.op === "gap" ? "hole" : "del";
      var g = Math.max(2, Math.min(10, (m.w || "").length || 3));
      return '<i class="blk ' + cls + (m.s ? " s" : "") + '" style="--g:' + g + '"></i>';
    }).join("");
    return '<span class="strip" aria-hidden="true">' + html + "</span>";
  }
  function errCount(marks, spanOnly) {
    if (!marks) return null;
    return marks.filter(function (m) { return (m.op === "sub" || m.op === "ins" || m.op === "del") && (!spanOnly || m.s); }).length;
  }
  /* o: {key, name, sub, kind, marks, note, span (bool), count (bool), disabled} */
  function panel(o) {
    var li = h("li", { class: "pnl" + (o.disabled ? " disabled" : ""), "data-key": o.key, "data-kind": o.kind,
      tabindex: o.disabled ? null : "0", role: o.disabled ? null : "button", "aria-pressed": o.disabled ? null : "false" });
    var cnt = "";
    if (o.count && o.marks && o.marks.length && o.kind !== "orig") {
      var n = errCount(o.marks, o.span);
      cnt = o.span ? (n === 0 ? '<span class="pnl-count exact"><b>exact</b></span>' : '<span class="pnl-count"><b class="bad">' + n + "</b>wrong in span</span>")
        : '<span class="pnl-count"><b class="' + (n ? "bad" : "") + '">' + n + "</b>wrong</span>";
    }
    var words = wordsHTML(o.marks, { soft: o.span });
    var text = words === null || words === "" ? '<span class="note">' + esc(o.note || "") + "</span>" : words;
    li.innerHTML = '<span class="pnl-head"><span class="pnl-name">' + esc(o.name) + "</span>" + (o.sub ? '<span class="pnl-sub">' + esc(o.sub) + "</span>" : "") + cnt + "</span>" +
      (o.strip === false ? "" : stripHTML(o.marks, o.kind)) + '<span class="pnl-text">' + text + "</span>";
    return li;
  }
  function bindSwitch(root, player) {
    $$("[data-key]", root).forEach(function (el) {
      if (el.classList.contains("disabled")) return;
      function go() { player.set(el.getAttribute("data-key")); }
      el.addEventListener("click", go);
      if (el.tagName !== "BUTTON") el.addEventListener("keydown", function (e) { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); go(); } });
    });
  }
  function markActive(root, key) {
    $$("[data-key]", root).forEach(function (x) {
      var on = x.getAttribute("data-key") === key;
      x.classList.toggle("active", on);
      if (x.hasAttribute("aria-pressed")) x.setAttribute("aria-pressed", on ? "true" : "false");
    });
  }
  function chips(items) {
    /* items: [{key, label, kind, disabled}] ; a "vs" separator after Ours, as in the video */
    var row = h("div", { class: "chips", role: "group", "aria-label": "Soundtrack" });
    items.forEach(function (it, i) {
      if (it.disabled) return;
      row.appendChild(h("button", { type: "button", "data-key": it.key, "data-kind": it.kind, "aria-pressed": "false", text: it.label }));
      if (it.kind === "ours" && i === 0 && items.length > 2) row.appendChild(h("span", { class: "vs", text: "vs" }));
    });
    return row;
  }
  var WAVE = '<svg viewBox="0 0 18 14" aria-hidden="true"><rect x="0" y="5" width="2" height="4" rx="1"/><rect x="4" y="2" width="2" height="10" rx="1"/>' +
    '<rect x="8" y="0" width="2" height="14" rx="1"/><rect x="12" y="3" width="2" height="8" rx="1"/><rect x="16" y="5.5" width="2" height="3" rx="1"/></svg>';
  function condBadge(text) { return h("span", { class: "cond", html: WAVE + esc(text) }); }

  /* ---------------------------------------------------------------- sample cards */
  function hasSpan(c) { return c.span && c.span.length > 0; }
  function card(c) {
    var span = hasSpan(c), t = {};
    c.tracks.forEach(function (x) { t[x.key] = x; });
    var order = ["truth", "input", "rml"];
    var tracks = order.map(function (k) { return { key: k, file: t[k].file, label: t[k].label, kind: kindOf(k) }; });
    var p = new Player({ tracks: tracks, poster: c.poster, bands: c.bands, initial: "rml", bandLabel: c.group === "noisy" ? "noise" : "silenced", label: c.cond });
    var list = h("ol", { class: "panels" });
    order.forEach(function (k) {
      var x = t[k];
      list.appendChild(panel({ key: k, name: x.label, kind: kindOf(k), marks: x.marks, note: x.note, span: span, count: k === "rml" || (k === "input" && !span),
        sub: k === "input" && x.marks && x.marks.length && !span && c.group !== "silent" ? "heard by Whisper" : null }));
    });
    bindSwitch(list, p);
    p.onchange = function (k) { markActive(list, k); };
    p.onchange(p.key);
    var head = h("div", { class: "card-head" }, [condBadge(c.cond), h("span", { class: "route", html: "Ours: <b>" + esc(c.chosen_label) + "</b>" })]);
    var why = h("a", { class: "why", href: "#selection", text: "All candidates and why this one", onclick: function () { explorerSelect(c.id); } });
    return h("article", { class: "card", id: "card-" + c.id }, [head, h("p", { class: "desc", text: c.desc }), p.el, list, why]);
  }
  (function samples() {
    var films = $("#filmCards");
    D.cards.forEach(function (c) {
      if (c.group === "film") films.appendChild(card(c));
      else { var g = $('.grid.cards[data-group="' + c.group + '"]'); if (g) g.appendChild(card(c)); }
    });
  })();

  /* ---------------------------------------------------------------- tabs */
  function tabs(listSel) {
    var list = $(listSel);
    if (!list) return;
    var btns = $$('[role="tab"]', list);
    function sel(b) {
      btns.forEach(function (x) {
        var on = x === b;
        x.setAttribute("aria-selected", on ? "true" : "false");
        x.tabIndex = on ? 0 : -1;
        var panelEl = document.getElementById(x.getAttribute("aria-controls"));
        if (panelEl) panelEl.hidden = !on;
      });
      players.forEach(function (p) { if (!p.video.paused && p.el.offsetParent === null) p.video.pause(); });
    }
    btns.forEach(function (b, i) {
      b.addEventListener("click", function () { sel(b); });
      b.addEventListener("keydown", function (e) {
        var j = e.key === "ArrowRight" ? i + 1 : e.key === "ArrowLeft" ? i - 1 : null;
        if (j === null) return;
        j = (j + btns.length) % btns.length; btns[j].focus(); sel(btns[j]); e.preventDefault();
      });
    });
  }
  tabs("#sampleTabs"); tabs("#blTabs");

  /* ---------------------------------------------------------------- baseline comparisons */
  function comparison(root, items) {
    /* items: [{id, poster, caption, sub, typical, bands, cond, ref, rows:[{key, name, sub, file, marks, kind}], span}] */
    if (!root || !items.length) return;
    var picker = h("div", { class: "picker", role: "listbox", "aria-label": "Choose a clip" });
    var body = h("div", { class: "compare-body" });
    root.appendChild(picker); root.appendChild(body);
    var current = null;
    function show(it, btn) {
      if (current) current.video.pause();
      $$(".pick", picker).forEach(function (b) { b.setAttribute("aria-selected", b === btn ? "true" : "false"); });
      body.innerHTML = "";
      var tracks = it.rows.map(function (r) { return { key: r.key, file: r.file, label: r.name, kind: r.kind }; });
      var p = new Player({ tracks: tracks, poster: it.poster, bands: it.bands, initial: it.rows.filter(function (r) { return r.kind === "ours"; })[0].key, label: it.caption });
      var ch = chips([it.rows.filter(function (r) { return r.kind === "ours"; })[0]].concat(it.rows.filter(function (r) { return r.kind !== "ours"; }))
        .map(function (r) { return { key: r.key, label: r.name, kind: r.kind, disabled: !r.file }; }));
      var list = h("ol", { class: "panels" });
      it.rows.forEach(function (r) {
        list.appendChild(panel({ key: r.key, name: r.name, sub: r.sub, kind: r.kind, marks: r.marks, note: r.file ? "" : "no output for this clip",
          span: it.span, count: r.kind !== "orig" && !(it.span && r.kind === "input"), disabled: !r.file }));
      });
      bindSwitch(list, p); bindSwitch(ch, p);
      p.onchange = function (k) { markActive(list, k); markActive(ch, k); };
      p.onchange(p.key);
      var left = h("div", { class: "compare-main" }, [h("div", { class: "card-head", style: "margin-bottom:12px" }, [condBadge(it.cond)]), p.el, ch]);
      var side = h("div", { class: "compare-side" }, [h("p", { class: "ref", html: "<b>Reference:</b> " + esc(it.ref) }), list]);
      body.appendChild(left); body.appendChild(side);
      current = p;
    }
    items.forEach(function (it, i) {
      var th = h("span", { class: "thumb" });
      lazyBg(th, it.poster);
      var b = h("button", { class: "pick", type: "button", role: "option", "aria-selected": "false" }, [th,
        h("span", { class: "cap", html: esc(it.caption) + (it.typical ? ' <span class="badge typical">typical</span>' : "") + (it.sub ? "<em>" + esc(it.sub) + "</em>" : "") })]);
      b.addEventListener("click", function () { show(it, b); });
      picker.appendChild(b);
      if (i === 0) show(it, b);
    });
  }
  function ordered(rows) {
    /* Original, Input, baselines, Ours: the order of the video */
    var rank = function (r) { return r.kind === "orig" ? 0 : r.kind === "input" ? 1 : r.kind === "ours" ? 3 : 2; };
    return rows.slice().sort(function (a, b) { return rank(a) - rank(b); });
  }
  (function baselines() {
    comparison($('.compare[data-set="babble"]'), D.babble.map(function (b, i) {
      return { id: b.id, poster: b.poster, caption: "Clip " + (i + 1), sub: b.typical ? "median case" : "best case", typical: b.typical,
        bands: [], cond: b.cond, ref: b.ref.join(" "), span: false,
        rows: ordered(b.rows.map(function (r) { return { key: r.key, name: r.name, sub: r.sub, file: r.file, marks: r.marks, kind: kindOf(r.key) }; })) };
    }));
    function fromCards(filter, span) {
      return D.cards.filter(function (c) { return c.baselines && filter(c); }).map(function (c) {
        var t = {}; c.tracks.forEach(function (x) { t[x.key] = x; });
        var rows = [{ key: "truth", name: "Original", file: t.truth.file, marks: t.truth.marks, kind: "orig" },
          { key: "input", name: "Input", file: t.input.file, marks: t.input.marks, kind: "input" }];
        c.baselines.forEach(function (b) { rows.push({ key: b.key, name: b.name, sub: b.sub, file: b.file, marks: b.marks, kind: "other" }); });
        rows.push({ key: "rml", name: "Ours", file: t.rml.file, marks: t.rml.marks, kind: "ours" });
        return { id: c.id, poster: c.poster, caption: c.cond, sub: "", bands: c.bands, cond: c.cond, ref: c.ref.join(" "), rows: rows, span: span };
      });
    }
    comparison($('.compare[data-set="noise"]'), fromCards(function (c) { return c.group === "noisy"; }, false));
    comparison($('.compare[data-set="silenced"]'), fromCards(function (c) { return c.group === "silenced"; }, true));
  })();

  /* ---------------------------------------------------------------- selection explorer */
  var explorerSelect = function () {};
  (function explorer() {
    var picker = $("#selPicker"), host = $("#selPlayer"), tbody = $("#selTable tbody"), summary = $("#selSummary");
    if (!picker) return;
    var order = ["film", "silenced", "noisy", "silent", "generated"];
    var cards = D.cards.slice().sort(function (a, b) { return order.indexOf(a.group) - order.indexOf(b.group); });
    var current = null, buttons = {};
    function f1(x) { return x === null || x === undefined ? "–" : (Math.round(x * 10) / 10).toFixed(1); }
    function show(c) {
      if (current) current.video.pause();
      Object.keys(buttons).forEach(function (k) { buttons[k].setAttribute("aria-selected", k === c.id ? "true" : "false"); });
      var t = {}; c.tracks.forEach(function (x) { t[x.key] = x; });
      var tracks = [{ key: "input", file: t.input.file, label: "Input", kind: "input" }];
      c.candidates.forEach(function (k) { if (k.file) tracks.push({ key: k.key, file: k.file, label: k.name + " · " + k.mode, kind: k.chosen ? "ours" : "other" }); });
      tracks.push({ key: "truth", file: t.truth.file, label: "Original", kind: "orig" });
      host.innerHTML = "";
      var p = new Player({ tracks: tracks, poster: c.poster, bands: c.bands, initial: c.chosen, bandLabel: c.group === "noisy" ? "noise" : "silenced", label: c.cond });
      var ch = chips([{ key: c.chosen, label: "Selected", kind: "ours" }, { key: "input", label: "Input", kind: "input" }, { key: "truth", label: "Original", kind: "orig" }]);
      host.appendChild(h("div", { class: "card-head", style: "margin-bottom:12px" }, [condBadge(c.cond)]));
      host.appendChild(p.el); host.appendChild(ch);
      host.appendChild(h("p", { class: "fine", text: c.desc + " Reference: “" + c.ref.join(" ") + "”" }));
      tbody.innerHTML = "";
      var span = hasSpan(c), maxEv = 1, maxW = 0;
      c.candidates.forEach(function (k) {
        if (k.audio_ev !== null) maxEv = Math.max(maxEv, Math.abs(k.audio_ev));
        if (k.lip_ev !== null) maxEv = Math.max(maxEv, Math.abs(k.lip_ev));
        if (k.weight !== null) maxW = Math.max(maxW, k.weight);
      });
      function evCell(v) {
        if (v === null) return '<td class="num">–</td>';
        return '<td class="num">' + f1(v) + '<span class="mini"><i style="width:' + (Math.max(0, v) / maxEv * 100) + '%"></i></span></td>';
      }
      c.candidates.forEach(function (k) {
        var tr = h("tr", { "data-key": k.file ? k.key : null, class: (k.voter_only ? "voter" : "") + (k.file ? " playable" : "") + (k.chosen ? " sel" : "") });
        var res = k.correct ? (k.correct === "exact" ? "span exact" : k.correct === "wrong" ? "span wrong" : k.correct) : "";
        var good = res === "span exact" || res === "WER 0%";
        tr.innerHTML =
          '<td class="cfg">' + (k.file ? '<span class="pi"><svg viewBox="0 0 24 24"><path d="M7 4.5v15l13-7.5z"/></svg></span>' : "") + esc(k.name) +
          "<small>" + esc(k.mode) + "</small>" + (k.chosen ? ' <span class="badge chosen">selected</span>' : "") + "</td>" +
          '<td class="tw">' + (wordsHTML(k.marks, { soft: span }) || '<span class="note">(no speech)</span>') +
          (res ? '<span class="res"><span class="badge' + (good ? " good" : "") + '">' + esc(res) + "</span></span>" : "") + "</td>" +
          evCell(k.audio_ev) + evCell(k.lip_ev) +
          (k.weight === null ? '<td class="num"><span class="nested" title="Already contained in L’s input, so it does not vote separately">nested</span></td>'
            : '<td class="num">' + k.weight.toFixed(2) + '<span class="mini w"><i style="width:' + (maxW ? k.weight / maxW * 100 : 0) + '%"></i></span></td>') +
          '<td class="num">' + (k.risk === null ? "–" : k.risk.toFixed(3)) + "</td>";
        if (k.file) tr.addEventListener("click", function () { p.set(k.key); });
        tbody.appendChild(tr);
      });
      bindSwitch(ch, p);
      p.onchange = function (key) { markActive(tbody, key); markActive(ch, key); };
      p.onchange(p.key);
      var sel = c.candidates.filter(function (k) { return k.chosen; })[0];
      var others = c.candidates.filter(function (k) { return !k.voter_only && !k.chosen && k.risk !== null; }).sort(function (a, b) { return a.risk - b.risk; });
      var nOut = c.candidates.filter(function (k) { return !k.voter_only; }).length;
      var s = "RML produced <b>" + nOut + " outputs</b>. The selected one, <b>" + esc(sel.name) + "</b> (" + esc(sel.mode) + "), has the lowest risk, " + sel.risk.toFixed(3);
      if (others.length) {
        var nx = others[0];
        s += Math.abs(nx.risk - sel.risk) < 1e-9 ? "; it ties with " + esc(nx.name) + " (" + esc(nx.mode) + "), and ties go to the configuration that uses more components." :
          ", against " + nx.risk.toFixed(3) + " for the runner-up, " + esc(nx.name) + " (" + esc(nx.mode) + ").";
      } else s += ".";
      var keepEv = c.candidates.filter(function (k) { return /kept/.test(k.mode) && k.audio_ev !== null; }).map(function (k) { return k.audio_ev; });
      if (/silence/.test(sel.mode)) s += " The outputs that replace the damaged audio with silence follow the lips and agree with each other and with the text voters, so the noisy audio is switched off without any threshold on the noise level.";
      else if (span || c.group === "generated") s += " The outputs that keep the audible speech and speak L’s transcript agree best with the voters, and both the ear and the eye support them.";
      else if (keepEv.length && Math.max.apply(null, keepEv) > 20) s += " The ear still finds strong support in the input audio, so the outputs that keep it win the vote.";
      summary.innerHTML = s;
      current = p;
    }
    cards.forEach(function (c) {
      var th = h("span", { class: "thumb" });
      lazyBg(th, c.poster);
      var lab = c.group === "film" ? "Film" : c.group === "generated" ? "Generated" : c.group === "silent" ? "Fully silent" : c.group === "noisy" ? "Noisy" : "Silenced";
      var b = h("button", { class: "pick", type: "button", role: "option", "aria-selected": "false" }, [th, h("span", { class: "cap", html: esc(lab) + "<em>" + esc(c.cond.replace(/^film · /, "")) + "</em>" })]);
      b.addEventListener("click", function () { show(c); });
      buttons[c.id] = b;
      picker.appendChild(b);
    });
    var first = cards.filter(function (c) { return c.id === "ps22"; })[0] || cards[0];
    if (first) show(first);
    explorerSelect = function (id) {
      var c = byId[id];
      if (!c) return;
      show(c);
      var b = buttons[id];
      if (b && b.scrollIntoView) b.scrollIntoView({ block: "nearest", inline: "center" });
    };
  })();

  /* ---------------------------------------------------------------- charts */
  var tip = $("#tooltip");
  function showTip(e, html) { tip.innerHTML = html; tip.hidden = false; moveTip(e); }
  function moveTip(e) {
    var x = e.clientX + 14, y = e.clientY + 14, r = tip.getBoundingClientRect();
    if (x + r.width > window.innerWidth - 8) x = e.clientX - r.width - 14;
    if (y + r.height > window.innerHeight - 8) y = e.clientY - r.height - 14;
    tip.style.left = x + "px"; tip.style.top = y + "px";
  }
  function hideTip() { tip.hidden = true; }

  function barChart(el, cfg) {
    /* cfg: {title, sub, unit, lower (bool), max, views:[{key,label}], rows:[{label, small, v:{key:value}, ours, ref, note}], foot} */
    var view = cfg.views[0].key, tableOn = false;
    var head = h("div", { class: "chart-head" }, [h("div", { class: "chart-title", html: esc(cfg.title) + "<small>" + esc(cfg.sub) + "</small>" })]);
    var seg = null;
    if (cfg.views.length > 1) {
      seg = h("div", { class: "chip-seg", role: "group", "aria-label": "Condition" });
      cfg.views.forEach(function (vw) {
        seg.appendChild(h("button", { type: "button", "aria-pressed": vw.key === view ? "true" : "false", text: vw.label,
          onclick: function () { view = vw.key; $$("button", seg).forEach(function (b) { b.setAttribute("aria-pressed", b.textContent === vw.label ? "true" : "false"); }); draw(); } }));
      });
      head.appendChild(seg);
    }
    var bars = h("div", { class: "bars", role: "img" });
    var table = h("div", { hidden: true });
    var tbtn = h("button", { class: "linkbtn", type: "button", text: "Show as table", onclick: function () {
      tableOn = !tableOn; table.hidden = !tableOn; bars.hidden = tableOn; tbtn.textContent = tableOn ? "Show as chart" : "Show as table"; } });
    var foot = h("div", { class: "chart-foot" }, [h("span", { text: cfg.foot || "" }), tbtn]);
    el.appendChild(head); el.appendChild(bars); el.appendChild(table); el.appendChild(foot);
    // table (all views)
    var th = "<tr><th>Method</th>" + cfg.views.map(function (v) { return "<th>" + esc(v.label) + "</th>"; }).join("") + "</tr>";
    table.innerHTML = "<table>" + th + cfg.rows.map(function (r) {
      return "<tr><td>" + esc(r.label + (r.small ? " " + r.small : "")) + "</td>" + cfg.views.map(function (v) { var x = r.v[v.key]; return "<td>" + (x === undefined ? "–" : x.toFixed(1)) + "</td>"; }).join("") + "</tr>";
    }).join("") + "</table>";
    var rowsEl = {};
    cfg.rows.forEach(function (r, i) {
      var bf = h("div", { class: "bf" });
      var row = h("div", { class: "brow" + (r.ours ? " ours" : "") + (r.ref ? " ref" : "") }, [
        h("span", { class: "bl", html: esc(r.label) + (r.small ? " <small>" + esc(r.small) + "</small>" : "") }),
        h("span", { class: "bt" }, [bf]), h("span", { class: "bv" })]);
      row.addEventListener("pointerenter", function (e) {
        var x = r.v[view];
        showTip(e, "<b>" + esc(r.label + (r.small ? " " + r.small : "")) + "</b><br>" + (x === undefined ? "not reported" : x.toFixed(1) + cfg.unit) + (r.note ? "<br>" + esc(r.note) : ""));
      });
      row.addEventListener("pointermove", moveTip);
      row.addEventListener("pointerleave", hideTip);
      rowsEl[i] = { row: row, bf: bf, bv: row.lastChild };
    });
    function draw() {
      var idx = cfg.rows.map(function (r, i) { return i; }).filter(function (i) { return cfg.rows[i].v[view] !== undefined; });
      idx.sort(function (a, b) { var x = cfg.rows[a].v[view], y = cfg.rows[b].v[view]; return cfg.lower ? x - y : y - x; });
      bars.innerHTML = "";
      var labelParts = [];
      idx.forEach(function (i) {
        var r = cfg.rows[i], x = r.v[view], o = rowsEl[i];
        o.bf.style.width = Math.min(100, 100 * x / cfg.max) + "%";
        o.bv.textContent = x.toFixed(1);
        bars.appendChild(o.row);
        labelParts.push(r.label + (r.small ? " " + r.small : "") + " " + x.toFixed(1) + cfg.unit);
      });
      bars.setAttribute("aria-label", cfg.title + ": " + labelParts.join("; "));
    }
    draw();
  }

  (function charts() {
    var sil = $("#chartSilenced");
    if (sil) barChart(sil, {
      title: "Exact span recovery (%)", sub: "higher is better · silenced test sets", unit: "%", lower: false, max: 70,
      views: [{ key: "p", label: "Pooled" }, { key: "t", label: "TalkVid" }, { key: "v", label: "VoxMM" }],
      foot: "Text: the transcript a method receives.",
      rows: [
        { label: "Ours", small: "with USR 2.0 as R", ours: true, v: { t: 62.7, v: 28.3, p: 42.7 }, note: "Same L and G, a stronger lip reader swapped in" },
        { label: "Ours", small: "with Auto-AVSR as R", ours: true, v: { t: 60.5, v: 22.9, p: 38.6 }, note: "The default pipeline R → L → G" },
        { label: "F5-TTS", small: "L's text + span", v: { t: 57.4, v: 20.2, p: 35.8 }, note: "Speech editor, told the words and where the span lies" },
        { label: "AlignDiT", small: "L's text + span", v: { t: 56.3, v: 20.9, p: 35.7 }, note: "Lip-synchronized generator, told the words and the span" },
        { label: "USR 2.0", small: "text", v: { t: 53.5, v: 23.0, p: 35.8 }, note: "Lip reader, scored on its text" },
        { label: "VoiceCraft", small: "L's text + span", v: { t: 51.8, v: 18.3, p: 32.3 } },
        { label: "Auto-AVSR", small: "text", v: { t: 51.0, v: 16.1, p: 30.7 }, note: "RML's default lip reader on its own" },
        { label: "Llama-AVSR", small: "text", v: { t: 47.7, v: 18.0, p: 30.4 } },
        { label: "LipVoicer", v: { t: 43.4, v: 17.3, p: 28.3 }, note: "Speech from the video alone" }
      ]
    });
    var noi = $("#chartNoisy");
    if (noi) barChart(noi, {
      title: "Word error rate (%)", sub: "lower is better · TalkVid + VoxMM, 273 windows per condition", unit: "%", lower: true, max: 110,
      views: [{ key: "avg", label: "Average" }, { key: "b10", label: "Babble −10" }, { key: "b5", label: "Babble −5" },
        { key: "ss", label: "SFX seen" }, { key: "su", label: "SFX unseen" }, { key: "tk", label: "Talker" }],
      foot: "† only windows where its face preprocessing succeeds.",
      rows: [
        { label: "Ours", ours: true, v: { b10: 46.5, b5: 29.8, ss: 22.9, su: 21.2, tk: 22.5, avg: 28.6 } },
        { label: "VisualVoice", small: "†", v: { b10: 80.5, b5: 41.8, ss: 28.7, su: 29.1, tk: 20.7, avg: 40.1 }, note: "Audio-visual separation" },
        { label: "Llama-AVSR", small: "text, AV", v: { b10: 57.8, b5: 34.3, ss: 30.7, su: 26.0, tk: 90.4, avg: 47.9 } },
        { label: "USR 2.0", small: "text, lips", v: { b10: 47.4, b5: 47.4, ss: 47.4, su: 47.4, tk: 47.4, avg: 47.4 }, note: "Reads the lips alone, so the noise does not matter" },
        { label: "AlignDiT", small: "R's text †", v: { b10: 50.5, b5: 50.5, ss: 50.5, su: 50.5, tk: 50.5, avg: 50.5 }, note: "Ignores the audio" },
        { label: "USR 2.0", small: "text, AV", v: { b10: 62.4, b5: 28.8, ss: 35.4, su: 29.0, tk: 104.3, avg: 52.0 } },
        { label: "LipVoicer", v: { b10: 52.9, b5: 52.9, ss: 52.9, su: 52.9, tk: 52.9, avg: 52.9 }, note: "Speech from the video alone" },
        { label: "Noisy input", ref: true, v: { b10: 91.8, b5: 52.9, ss: 47.7, su: 48.0, tk: 104.8, avg: 69.0 }, note: "Whisper on the damaged input" },
        { label: "Auto-AVSR", small: "text, AV", v: { b10: 82.1, b5: 47.7, ss: 68.7, su: 61.0, tk: 93.6, avg: 70.6 } },
        { label: "DeepFilterNet3", v: { b10: 92.5, b5: 70.6, ss: 51.6, su: 43.3, tk: 102.8, avg: 72.2 }, note: "Audio-only enhancement" },
        { label: "Sidon", v: { b10: 99.4, b5: 93.0, ss: 61.5, su: 64.8, tk: 103.1, avg: 84.4 }, note: "Audio-only restoration" }
      ]
    });
    var hard = $("#chartHard");
    if (hard) {
      var a = h("div"), b = h("div");
      var duo = h("div", { class: "duo" }, [a, b]);
      hard.appendChild(duo);
      barChart(a, {
        title: "Silenced: exact span recovery (%)", sub: "higher is better · 116 cuts", unit: "%", lower: false, max: 35,
        views: [{ key: "x", label: "" }], foot: "",
        rows: [
          { label: "Ours", ours: true, v: { x: 27.6 } },
          { label: "AlignDiT", small: "L's text *", v: { x: 23.6 } },
          { label: "F5-TTS", small: "L's text", v: { x: 23.3 } },
          { label: "Llama-AVSR", small: "*", v: { x: 13.3 } },
          { label: "USR 2.0", small: "*", v: { x: 12.0 } },
          { label: "LipVoicer", small: "*", v: { x: 10.8 } },
          { label: "Auto-AVSR", v: { x: 8.6 }, note: "RML's lip reader on its own" }
        ]
      });
      barChart(b, {
        title: "Noisy: word error rate (%)", sub: "lower is better · 220 windows", unit: "%", lower: true, max: 100,
        views: [{ key: "x", label: "" }], foot: "",
        rows: [
          { label: "Ours", ours: true, v: { x: 29.9 } },
          { label: "VisualVoice", small: "*", v: { x: 41.6 }, note: "Face detection succeeds on 90 of 220 windows" },
          { label: "SAM Audio", small: "text prompt", v: { x: 53.7 } },
          { label: "Damaged input", ref: true, v: { x: 54.1 } },
          { label: "USR 2.0", small: "*", v: { x: 56.3 } },
          { label: "Llama-AVSR", small: "*", v: { x: 60.3 } },
          { label: "Auto-AVSR", small: "*", v: { x: 72.5 } },
          { label: "LipVoicer", small: "*", v: { x: 92.8 } },
          { label: "AlignDiT", small: "R's text *", v: { x: 97.0 } }
        ]
      });
    }
  })();

  /* ---------------------------------------------------------------- bibtex copy */
  (function bib() {
    var b = $("#copyBib");
    if (!b) return;
    b.addEventListener("click", function () {
      var t = $("#bibText").textContent;
      function done() { b.textContent = "Copied"; setTimeout(function () { b.textContent = "Copy"; }, 1600); }
      if (navigator.clipboard && navigator.clipboard.writeText) navigator.clipboard.writeText(t).then(done, function () {});
      else { var r = document.createRange(); r.selectNodeContents($("#bibText")); var s = window.getSelection(); s.removeAllRanges(); s.addRange(r); try { document.execCommand("copy"); done(); } catch (e) {} }
    });
  })();
})();
