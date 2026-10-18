// 地铁陪伴 - 前端聊天逻辑
// 纯原生 JS，无构建依赖。与后端 /api/v1/chat 对接。
// 会话记录使用 localStorage 持久化，可恢复、可清空；兼容并容错旧数据/异常数据。

(function () {
  "use strict";

  // 后端地址：
  //   - 默认同源（相对路径 ""）：页面由后端/隧道同一域名提供时，API 也走该域名，
  //     外部访问者不会错误地请求到自己设备的 localhost。
  //   - 可用 URL 参数 ?api=http://host:port 覆盖（如本地前端单独跑在 8080 时）。
  var API_PARAM = new URLSearchParams(location.search).get("api");
  var API_BASE = API_PARAM || (location.protocol.indexOf("http") === 0 ? "" : "http://127.0.0.1:8000");
  var HISTORY_KEY = "metro_chat_history";
  var SID_KEY = "metro_sid";
  var MAX_HISTORY = 100; // 本地最多保留条数，防止无限增长

  var messagesEl = document.getElementById("messages");
  var inputEl = document.getElementById("input");
  var sendBtn = document.getElementById("send");
  var clearBtn = document.getElementById("clear");
  var statusEl = document.getElementById("api-status");
  var welcomeEl = document.getElementById("welcome");
  var errorBanner = document.getElementById("error-banner");
  var quickEntriesEl = document.getElementById("quick-entries");

  // 内存中的对话记录：[{role:"user"|"bot", text, data?}]
  var history = [];
  // 待重试的文本（发送失败时保留，供"重试"按钮复用）
  var pendingRetryText = null;
  // 用户是否正在手动向上查看历史（为 true 时不强制滚动到底部）
  var userPaused = false;

  // ---------- 工具 ----------
  function el(tag, cls, text) {
    var e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text != null) e.textContent = text;
    return e;
  }

  function isNearBottom() {
    // 距离底部小于 120px 视为"在底部附近"，允许自动滚动
    return messagesEl.scrollHeight - messagesEl.scrollTop - messagesEl.clientHeight < 120;
  }

  function scrollBottom(force) {
    if (force || !userPaused) {
      messagesEl.scrollTop = messagesEl.scrollHeight;
    }
  }

  function showError(msg, retryText) {
    pendingRetryText = retryText || null;
    errorBanner.textContent = "出错了：" + msg;
    if (retryText) {
      var btn = el("button", "retry-link", "重试");
      btn.type = "button";
      btn.addEventListener("click", function () {
        errorBanner.classList.remove("show");
        sendText(retryText);
      });
      errorBanner.appendChild(btn);
    }
    errorBanner.classList.add("show");
    setTimeout(function () {
      if (errorBanner.textContent.indexOf(msg) >= 0) errorBanner.classList.remove("show");
    }, 8000);
  }

  function hideError() {
    errorBanner.classList.remove("show");
    errorBanner.textContent = "";
    pendingRetryText = null;
  }

  function updateWelcome() {
    if (history.length > 0) {
      welcomeEl.classList.add("hidden");
    } else {
      welcomeEl.classList.remove("hidden");
    }
  }

  function pct(n) {
    if (typeof n !== "number" || isNaN(n)) return "—";
    return Math.round(n * 100) + "%";
  }

  function safeStr(v) { return (typeof v === "string" && v) ? v : ""; }

  // ---------- localStorage 容错读写 ----------
  function loadHistory() {
    var raw = null;
    try { raw = localStorage.getItem(HISTORY_KEY); } catch (e) { /* 隐私模式等不可用 */ }
    if (!raw) return [];
    try {
      var arr = JSON.parse(raw);
      if (!Array.isArray(arr)) return [];
      var ok = [];
      for (var i = 0; i < arr.length; i++) {
        var it = arr[i];
        if (it && (it.role === "user" || it.role === "bot") && typeof it.text === "string") {
          ok.push({ role: it.role, text: it.text, data: it.data || null });
        }
      }
      return ok;
    } catch (e) {
      return [];
    }
  }

  function saveHistory() {
    try {
      var trimmed = history.slice(-MAX_HISTORY);
      localStorage.setItem(HISTORY_KEY, JSON.stringify(trimmed));
    } catch (e) { /* 存储失败（满/隐私模式）静默忽略 */ }
  }

  function getSid() {
    try {
      var s = localStorage.getItem(SID_KEY);
      if (s) return s;
    } catch (e) {}
    var s = "s" + Math.random().toString(36).slice(2, 10);
    try { localStorage.setItem(SID_KEY, s); } catch (e) {}
    return s;
  }

  // ---------- 消息渲染 ----------
  function renderAnalysis(data) {
    data = data || {};
    var box = el("div", "analysis");

    var row1 = el("div", "row");
    row1.appendChild(el("span", "label", "情绪"));
    if (data.emotion) {
      row1.appendChild(el("span", "tag", safeStr(data.emotion) + " " + pct(data.emotion_confidence)));
    } else {
      row1.appendChild(el("span", "tag", "未知"));
    }
    box.appendChild(row1);

    var row2 = el("div", "row");
    row2.appendChild(el("span", "label", "强度"));
    if (data.emotion_intensity) {
      var t = el("span", "tag", safeStr(data.emotion_intensity));
      if (data.emotion_intensity === "高") t.classList.add("urgent");
      row2.appendChild(t);
    } else {
      row2.appendChild(el("span", "tag", "未知"));
    }
    if (Array.isArray(data.intensity_evidence) && data.intensity_evidence.length) {
      row2.appendChild(el("span", "label", data.intensity_evidence.map(safeStr).join("；")));
    }
    box.appendChild(row2);

    var row3 = el("div", "row");
    row3.appendChild(el("span", "label", "事件"));
    if (data.event) {
      var et = el("span", "tag", safeStr(data.event) + " " + pct(data.event_confidence));
      if (safeStr(data.event).indexOf("突发事件") >= 0) et.classList.add("urgent");
      row3.appendChild(et);
    } else {
      row3.appendChild(el("span", "tag", "未知"));
    }
    box.appendChild(row3);

    var decision = data.decision || {};
    if (decision.strategy) {
      var row4 = el("div", "row");
      row4.appendChild(el("span", "label", "策略"));
      var st = el("span", "tag", safeStr(decision.strategy));
      if (decision.needs_escalation) st.classList.add("urgent");
      row4.appendChild(st);
      if (decision.needs_escalation) row4.appendChild(el("span", "tag warn", "已升级人工"));
      box.appendChild(row4);
    }
    if (Array.isArray(decision.actions) && decision.actions.length) {
      var row5 = el("div", "row");
      row5.appendChild(el("span", "label", "建议"));
      var ul = el("ul", "actions");
      decision.actions.forEach(function (a) { ul.appendChild(el("li", null, safeStr(a))); });
      row5.appendChild(ul);
      box.appendChild(row5);
    }

    return box;
  }

  function renderMessage(item) {
    var wrap = el("div", item.role === "user" ? "msg msg-user" : "msg msg-bot");
    var text = item.role === "user" ? item.text : (item.text || "（无回复内容）");
    wrap.appendChild(el("div", "bubble", text));
    if (item.role === "bot" && item.data) {
      try { wrap.appendChild(renderAnalysis(item.data)); } catch (e) { /* 损坏数据不崩溃 */ }
    }
    messagesEl.appendChild(wrap);
    return wrap;
  }

  function redrawAll() {
    messagesEl.innerHTML = "";
    history.forEach(renderMessage);
    scrollBottom(true);
    updateWelcome();
  }

  function appendUser(text) {
    history.push({ role: "user", text: text });
    saveHistory();
    renderMessage(history[history.length - 1]);
    scrollBottom();
    updateWelcome();
  }

  function appendBot(text, data) {
    history.push({ role: "bot", text: text, data: data || null });
    saveHistory();
    renderMessage(history[history.length - 1]);
    scrollBottom();
  }

  function appendLoading() {
    var wrap = el("div", "msg msg-bot");
    var bubble = el("div", "bubble loading");
    var dots = el("span", "dots");
    dots.appendChild(el("span"));
    dots.appendChild(el("span"));
    dots.appendChild(el("span"));
    bubble.appendChild(dots);
    bubble.appendChild(el("span", "loading-text", "正在想怎么陪你聊聊……"));
    wrap.appendChild(bubble);
    messagesEl.appendChild(wrap);
    scrollBottom();
    return wrap;
  }

  function clearChat() {
    history = [];
    saveHistory();
    redrawAll();
    hideError();
  }

  // ---------- textarea 自适应高度 ----------
  function autoResize() {
    inputEl.style.height = "auto";
    inputEl.style.height = Math.min(inputEl.scrollHeight, 140) + "px";
  }

  // ---------- 与后端交互 ----------
  function setStatus(text, cls) {
    statusEl.textContent = text;
    statusEl.className = "status" + (cls ? " " + cls : "");
  }

  var HEALTH_RETRY = 3;
  function checkHealth(attempt) {
    attempt = attempt || 1;
    fetch(API_BASE + "/health")
      .then(function (r) { return r.ok; })
      .then(function (ok) {
        if (ok) {
          setStatus("已连接", "ok");
          sendBtn.disabled = false;
          if (quickEntriesEl) quickEntriesEl.querySelectorAll(".chip").forEach(function (c) { c.disabled = false; });
        } else if (attempt < HEALTH_RETRY) {
          setTimeout(function () { checkHealth(attempt + 1); }, 800);
        } else {
          setStatus("后端未连接", "err");
          sendBtn.disabled = true;
        }
      })
      .catch(function () {
        if (attempt < HEALTH_RETRY) {
          setTimeout(function () { checkHealth(attempt + 1); }, 800);
        } else {
          setStatus("后端未连接（请先启动后端）", "err");
          sendBtn.disabled = true;
        }
      });
  }

  function sendText(text) {
    text = (text || "").trim();
    if (!text) {
      showError("消息内容不能为空");
      return;
    }
    if (sendBtn.disabled) return; // 发送期间防止重复提交

    hideError();
    appendUser(text);
    inputEl.value = "";
    autoResize();
    sendBtn.disabled = true;
    if (quickEntriesEl) quickEntriesEl.querySelectorAll(".chip").forEach(function (c) { c.disabled = true; });

    var loadingEl = appendLoading();

    fetch(API_BASE + "/api/v1/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: text, session_id: "web-" + getSid() }),
    })
      .then(function (resp) {
        if (!resp.ok) {
          return resp.text().then(function (t) {
            throw new Error("HTTP " + resp.status + (t ? "：" + t.slice(0, 200) : ""));
          });
        }
        return resp.json();
      })
      .then(function (data) {
        if (loadingEl && loadingEl.parentNode) loadingEl.parentNode.removeChild(loadingEl);
        appendBot((data && data.reply) || "（无回复内容）", data);
      })
      .catch(function (err) {
        if (loadingEl && loadingEl.parentNode) loadingEl.parentNode.removeChild(loadingEl);
        showError((err && err.message) || "网络错误，请稍后重试", text);
      })
      .finally(function () {
        sendBtn.disabled = false;
        if (quickEntriesEl) quickEntriesEl.querySelectorAll(".chip").forEach(function (c) { c.disabled = false; });
        inputEl.focus();
      });
  }

  function send() {
    sendText(inputEl.value);
  }

  // ---------- 事件绑定 ----------
  sendBtn.addEventListener("click", send);
  clearBtn.addEventListener("click", function () {
    if (history.length === 0 && !welcomeEl.classList.contains("hidden")) return;
    clearChat();
  });

  inputEl.addEventListener("keydown", function (e) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      send();
    }
  });
  inputEl.addEventListener("input", autoResize);

  // 用户手动滚动：向上离开底部时暂停自动滚动，回到底部附近时恢复
  messagesEl.addEventListener("scroll", function () {
    userPaused = !isNearBottom();
  });

  // 快捷入口：点击后通过正常发送流程发出自然语言请求
  if (quickEntriesEl) {
    quickEntriesEl.addEventListener("click", function (e) {
      var chip = e.target.closest(".chip");
      if (!chip || chip.disabled) return;
      var text = chip.getAttribute("data-text");
      if (text) sendText(text);
    });
  }

  // 启动：恢复历史
  history = loadHistory();
  redrawAll();

  checkHealth();
  setInterval(checkHealth, 15000);
  inputEl.focus();
  autoResize();
})();
