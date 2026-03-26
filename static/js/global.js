/* ============================================
   澳洲联通 - 全局 JS 组件系统
   ============================================ */

// =========================================
// TOAST NOTIFICATION SYSTEM
// =========================================
const Toast = (() => {
  let container = null;

  function ensureContainer() {
    if (!container) {
      container = document.createElement('div');
      container.className = 'toast-container';
      document.body.appendChild(container);
    }
    return container;
  }

  function getIcon(type) {
    const icons = {
      success: '✓',
      error: '✕',
      warning: '!',
      info: 'ℹ'
    };
    return icons[type] || icons.info;
  }

  function getTitle(type) {
    const titles = {
      success: '成功',
      error: '错误',
      warning: '警告',
      info: '提示'
    };
    return titles[type] || titles.info;
  }

  /**
   * Show a toast notification
   * @param {string} message - The message to display
   * @param {string} type - 'success' | 'error' | 'warning' | 'info'
   * @param {object} options - { title, duration, closable }
   */
  function show(message, type = 'info', options = {}) {
    const c = ensureContainer();
    const duration = options.duration || (type === 'error' ? 6000 : 4000);
    const title = options.title || getTitle(type);
    const closable = options.closable !== false;

    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    toast.style.position = 'relative';
    toast.innerHTML = `
      <div class="toast-icon">${getIcon(type)}</div>
      <div class="toast-body">
        <div class="toast-title">${title}</div>
        <div class="toast-message">${escapeHtml(message)}</div>
      </div>
      ${closable ? '<button class="toast-close" aria-label="关闭">&times;</button>' : ''}
      <div class="toast-progress" style="animation-duration: ${duration}ms"></div>
    `;

    if (closable) {
      toast.querySelector('.toast-close').addEventListener('click', () => removeToast(toast));
    }

    c.appendChild(toast);

    const timer = setTimeout(() => removeToast(toast), duration);
    toast._timer = timer;

    // Limit max visible toasts
    const toasts = c.querySelectorAll('.toast:not(.toast-exit)');
    if (toasts.length > 5) {
      removeToast(toasts[0]);
    }

    return toast;
  }

  function removeToast(toast) {
    if (!toast || toast.classList.contains('toast-exit')) return;
    clearTimeout(toast._timer);
    toast.classList.add('toast-exit');
    setTimeout(() => toast.remove(), 300);
  }

  return {
    show,
    success: (msg, opts) => show(msg, 'success', opts),
    error: (msg, opts) => show(msg, 'error', opts),
    warning: (msg, opts) => show(msg, 'warning', opts),
    info: (msg, opts) => show(msg, 'info', opts),
  };
})();


// =========================================
// MODAL / DIALOG SYSTEM
// =========================================
const Dialog = (() => {

  function createOverlay() {
    const overlay = document.createElement('div');
    overlay.className = 'g-modal-overlay';
    return overlay;
  }

  function closeOverlay(overlay) {
    overlay.classList.add('modal-exit');
    setTimeout(() => overlay.remove(), 200);
  }

  /**
   * Show a confirmation dialog
   * @param {string} message - The message
   * @param {object} options - { title, confirmText, cancelText, type }
   * @returns {Promise<boolean>}
   */
  function confirm(message, options = {}) {
    return new Promise((resolve) => {
      const title = options.title || '确认操作';
      const confirmText = options.confirmText || '确认';
      const cancelText = options.cancelText || '取消';
      const type = options.type || 'primary'; // primary, danger

      const overlay = createOverlay();
      overlay.innerHTML = `
        <div class="g-modal">
          <div class="g-modal-header">
            <div class="g-modal-title">${escapeHtml(title)}</div>
            <button class="g-modal-close" data-action="close">&times;</button>
          </div>
          <div class="g-modal-body">${escapeHtml(message)}</div>
          <div class="g-modal-footer">
            <button class="g-btn g-btn-secondary" data-action="cancel">${escapeHtml(cancelText)}</button>
            <button class="g-btn g-btn-${type}" data-action="confirm">${escapeHtml(confirmText)}</button>
          </div>
        </div>
      `;

      function handleAction(result) {
        closeOverlay(overlay);
        resolve(result);
      }

      overlay.querySelector('[data-action="confirm"]').addEventListener('click', () => handleAction(true));
      overlay.querySelector('[data-action="cancel"]').addEventListener('click', () => handleAction(false));
      overlay.querySelector('[data-action="close"]').addEventListener('click', () => handleAction(false));
      overlay.addEventListener('click', (e) => { if (e.target === overlay) handleAction(false); });

      document.addEventListener('keydown', function onKey(e) {
        if (e.key === 'Escape') {
          document.removeEventListener('keydown', onKey);
          handleAction(false);
        }
      });

      document.body.appendChild(overlay);
      overlay.querySelector('[data-action="confirm"]').focus();
    });
  }

  /**
   * Show a prompt dialog
   * @param {string} message
   * @param {object} options - { title, defaultValue, placeholder, confirmText, cancelText }
   * @returns {Promise<string|null>}
   */
  function prompt(message, options = {}) {
    return new Promise((resolve) => {
      const title = options.title || '请输入';
      const defaultValue = options.defaultValue || '';
      const placeholder = options.placeholder || '';
      const confirmText = options.confirmText || '确认';
      const cancelText = options.cancelText || '取消';

      const overlay = createOverlay();
      overlay.innerHTML = `
        <div class="g-modal">
          <div class="g-modal-header">
            <div class="g-modal-title">${escapeHtml(title)}</div>
            <button class="g-modal-close" data-action="close">&times;</button>
          </div>
          <div class="g-modal-body">
            <div>${escapeHtml(message)}</div>
            <input class="g-modal-input" type="text" value="${escapeHtml(defaultValue)}" placeholder="${escapeHtml(placeholder)}">
          </div>
          <div class="g-modal-footer">
            <button class="g-btn g-btn-secondary" data-action="cancel">${escapeHtml(cancelText)}</button>
            <button class="g-btn g-btn-primary" data-action="confirm">${escapeHtml(confirmText)}</button>
          </div>
        </div>
      `;

      const input = overlay.querySelector('.g-modal-input');

      function handleAction(value) {
        closeOverlay(overlay);
        resolve(value);
      }

      overlay.querySelector('[data-action="confirm"]').addEventListener('click', () => handleAction(input.value));
      overlay.querySelector('[data-action="cancel"]').addEventListener('click', () => handleAction(null));
      overlay.querySelector('[data-action="close"]').addEventListener('click', () => handleAction(null));
      overlay.addEventListener('click', (e) => { if (e.target === overlay) handleAction(null); });

      input.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') handleAction(input.value);
      });

      document.addEventListener('keydown', function onKey(e) {
        if (e.key === 'Escape') {
          document.removeEventListener('keydown', onKey);
          handleAction(null);
        }
      });

      document.body.appendChild(overlay);
      setTimeout(() => { input.focus(); input.select(); }, 100);
    });
  }

  /**
   * Show an alert dialog (replaces window.alert)
   * @param {string} message
   * @param {object} options - { title, buttonText }
   * @returns {Promise<void>}
   */
  function alert(message, options = {}) {
    return new Promise((resolve) => {
      const title = options.title || '提示';
      const buttonText = options.buttonText || '确定';

      const overlay = createOverlay();
      overlay.innerHTML = `
        <div class="g-modal">
          <div class="g-modal-header">
            <div class="g-modal-title">${escapeHtml(title)}</div>
            <button class="g-modal-close" data-action="close">&times;</button>
          </div>
          <div class="g-modal-body">${escapeHtml(message)}</div>
          <div class="g-modal-footer">
            <button class="g-btn g-btn-primary" data-action="ok">${escapeHtml(buttonText)}</button>
          </div>
        </div>
      `;

      function handleClose() {
        closeOverlay(overlay);
        resolve();
      }

      overlay.querySelector('[data-action="ok"]').addEventListener('click', handleClose);
      overlay.querySelector('[data-action="close"]').addEventListener('click', handleClose);
      overlay.addEventListener('click', (e) => { if (e.target === overlay) handleClose(); });
      document.addEventListener('keydown', function onKey(e) {
        if (e.key === 'Escape' || e.key === 'Enter') {
          document.removeEventListener('keydown', onKey);
          handleClose();
        }
      });

      document.body.appendChild(overlay);
      overlay.querySelector('[data-action="ok"]').focus();
    });
  }

  return { confirm, prompt, alert };
})();


// =========================================
// LOADING SYSTEM
// =========================================
const Loading = (() => {
  let overlay = null;
  let count = 0;

  function show(text = '加载中...') {
    count++;
    if (overlay) {
      overlay.querySelector('.g-loading-text').textContent = text;
      return;
    }
    overlay = document.createElement('div');
    overlay.className = 'g-loading-overlay';
    overlay.innerHTML = `
      <div class="g-loading-spinner"></div>
      <div class="g-loading-text">${escapeHtml(text)}</div>
    `;
    document.body.appendChild(overlay);
  }

  function hide() {
    count = Math.max(0, count - 1);
    if (count > 0 || !overlay) return;
    overlay.classList.add('loading-exit');
    const el = overlay;
    overlay = null;
    setTimeout(() => el.remove(), 200);
  }

  function forceHide() {
    count = 0;
    hide();
  }

  /**
   * Set a button to loading state
   * @param {HTMLElement} btn
   * @param {boolean} loading
   */
  function button(btn, loading) {
    if (!btn) return;
    if (loading) {
      btn._originalContent = btn.innerHTML;
      btn.classList.add('loading');
      btn.disabled = true;
      const spinner = document.createElement('span');
      spinner.className = 'btn-spinner';
      btn.prepend(spinner);
    } else {
      btn.classList.remove('loading');
      btn.disabled = false;
      if (btn._originalContent) {
        btn.innerHTML = btn._originalContent;
        delete btn._originalContent;
      } else {
        const spinner = btn.querySelector('.btn-spinner');
        if (spinner) spinner.remove();
      }
    }
  }

  return { show, hide, forceHide, button };
})();


// =========================================
// FORM VALIDATION
// =========================================
const FormValidator = (() => {

  /**
   * Validate a single input
   * @param {HTMLElement} input
   * @param {object} rules - { required, minLength, maxLength, pattern, custom }
   * @returns {string|null} error message or null
   */
  function validateInput(input, rules = {}) {
    const value = input.value.trim();
    const errorEl = input.parentElement?.querySelector('.g-form-error');

    if (rules.required && !value) {
      setError(input, errorEl, rules.requiredMsg || '此字段为必填项');
      return rules.requiredMsg || '此字段为必填项';
    }

    if (rules.minLength && value.length < rules.minLength) {
      const msg = `至少输入 ${rules.minLength} 个字符`;
      setError(input, errorEl, msg);
      return msg;
    }

    if (rules.maxLength && value.length > rules.maxLength) {
      const msg = `最多输入 ${rules.maxLength} 个字符`;
      setError(input, errorEl, msg);
      return msg;
    }

    if (rules.pattern && !rules.pattern.test(value)) {
      const msg = rules.patternMsg || '格式不正确';
      setError(input, errorEl, msg);
      return msg;
    }

    if (rules.custom) {
      const msg = rules.custom(value);
      if (msg) {
        setError(input, errorEl, msg);
        return msg;
      }
    }

    clearError(input, errorEl);
    return null;
  }

  function setError(input, errorEl, msg) {
    input.classList.add('input-error');
    input.classList.remove('input-success');
    if (errorEl) {
      errorEl.textContent = msg;
      errorEl.classList.add('visible');
    }
  }

  function clearError(input, errorEl) {
    input.classList.remove('input-error');
    if (input.value.trim()) input.classList.add('input-success');
    if (errorEl) errorEl.classList.remove('visible');
  }

  /**
   * Attach live validation to a form
   * @param {HTMLFormElement} form
   * @param {object} fieldRules - { fieldName: rules }
   */
  function attachLive(form, fieldRules) {
    if (!form) return;
    Object.entries(fieldRules).forEach(([name, rules]) => {
      const input = form.querySelector(`[name="${name}"]`) || form.querySelector(`#${name}`);
      if (!input) return;
      input.addEventListener('blur', () => validateInput(input, rules));
      input.addEventListener('input', () => {
        if (input.classList.contains('input-error')) {
          validateInput(input, rules);
        }
      });
    });
  }

  /**
   * Validate all fields in a form
   * @param {HTMLFormElement} form
   * @param {object} fieldRules
   * @returns {boolean} true if all valid
   */
  function validateAll(form, fieldRules) {
    let valid = true;
    Object.entries(fieldRules).forEach(([name, rules]) => {
      const input = form.querySelector(`[name="${name}"]`) || form.querySelector(`#${name}`);
      if (!input) return;
      if (validateInput(input, rules)) valid = false;
    });
    return valid;
  }

  return { validateInput, attachLive, validateAll };
})();


// =========================================
// SMART POLLER (Phase 8 - Real-time updates)
// =========================================
const SmartPoller = (() => {
  const pollers = {};

  /**
   * Start a polling task
   * @param {string} id - Unique identifier
   * @param {Function} fn - Async function to execute
   * @param {number} interval - Interval in ms (default 30000)
   */
  function start(id, fn, interval = 30000) {
    stop(id);
    let timer;
    let paused = false;

    function run() {
      if (paused) return;
      fn().catch(err => console.warn(`[Poller ${id}]`, err));
      timer = setTimeout(run, interval);
    }

    // Visibility API: pause when hidden
    function onVisibility() {
      if (document.hidden) {
        paused = true;
        clearTimeout(timer);
      } else {
        paused = false;
        run();
      }
    }

    document.addEventListener('visibilitychange', onVisibility);

    pollers[id] = {
      stop: () => {
        clearTimeout(timer);
        document.removeEventListener('visibilitychange', onVisibility);
        delete pollers[id];
      }
    };

    run();
  }

  function stop(id) {
    if (pollers[id]) pollers[id].stop();
  }

  function stopAll() {
    Object.keys(pollers).forEach(stop);
  }

  return { start, stop, stopAll };
})();


// =========================================
// BROWSER NOTIFICATIONS (Phase 7)
// =========================================
const BrowserNotify = (() => {
  let permission = Notification?.permission || 'default';

  async function requestPermission() {
    if (!('Notification' in window)) return false;
    if (permission === 'granted') return true;
    const result = await Notification.requestPermission();
    permission = result;
    return result === 'granted';
  }

  function send(title, options = {}) {
    if (permission !== 'granted' || !document.hidden) return;
    try {
      const notif = new Notification(title, {
        icon: '/static/logo.png',
        badge: '/static/logo.png',
        ...options,
      });
      notif.onclick = () => {
        window.focus();
        notif.close();
        if (options.url) window.location.href = options.url;
      };
    } catch (e) {
      console.warn('Browser notification failed:', e);
    }
  }

  return { requestPermission, send };
})();


// =========================================
// HEARTBEAT (Phase 8)
// =========================================
const Heartbeat = (() => {
  let timer = null;

  function start(interval = 60000) {
    stop();
    function beat() {
      fetch('/api/heartbeat', { method: 'POST', headers: {'Content-Type': 'application/json'} }).catch(() => {});
    }
    beat();
    timer = setInterval(beat, interval);
  }

  function stop() {
    if (timer) clearInterval(timer);
    timer = null;
  }

  return { start, stop };
})();


// =========================================
// UTILITY FUNCTIONS
// =========================================
function escapeHtml(str) {
  if (!str) return '';
  const div = document.createElement('div');
  div.textContent = str;
  return div.innerHTML;
}

function formatDateTime(dateStr) {
  if (!dateStr) return '-';
  try {
    const d = new Date(dateStr.replace(' ', 'T'));
    return d.toLocaleString('zh-CN', { year: 'numeric', month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit' });
  } catch { return dateStr; }
}

function formatDate(dateStr) {
  if (!dateStr) return '-';
  try {
    const d = new Date(dateStr.replace(' ', 'T'));
    return d.toLocaleString('zh-CN', { year: 'numeric', month: '2-digit', day: '2-digit' });
  } catch { return dateStr; }
}

function timeAgo(dateStr) {
  if (!dateStr) return '';
  try {
    const d = new Date(dateStr.replace(' ', 'T'));
    const now = new Date();
    const diff = (now - d) / 1000;
    if (diff < 60) return '刚刚';
    if (diff < 3600) return `${Math.floor(diff / 60)} 分钟前`;
    if (diff < 86400) return `${Math.floor(diff / 3600)} 小时前`;
    if (diff < 604800) return `${Math.floor(diff / 86400)} 天前`;
    return formatDate(dateStr);
  } catch { return dateStr; }
}

function debounce(fn, delay = 300) {
  let timer;
  return function (...args) {
    clearTimeout(timer);
    timer = setTimeout(() => fn.apply(this, args), delay);
  };
}

/**
 * Helper to make API calls with loading and error handling
 */
async function apiCall(url, options = {}) {
  const { showLoading = false, loadingText, button: btn } = options;
  if (showLoading) Loading.show(loadingText);
  if (btn) Loading.button(btn, true);

  try {
    const resp = await fetch(url, options);
    const data = await resp.json();
    if (data.status === 'error') {
      Toast.error(data.message || '操作失败');
      return null;
    }
    return data;
  } catch (err) {
    Toast.error('网络请求失败: ' + err.message);
    return null;
  } finally {
    if (showLoading) Loading.hide();
    if (btn) Loading.button(btn, false);
  }
}


// =========================================
// MOBILE DRAWER
// =========================================
function initMobileDrawer() {
  const btn = document.querySelector('.g-mobile-menu-btn');
  const drawer = document.querySelector('.g-mobile-drawer');
  const backdrop = document.querySelector('.g-mobile-drawer-backdrop');
  const closeBtn = drawer?.querySelector('.g-mobile-drawer-close');

  if (!btn || !drawer) return;

  function open() {
    drawer.classList.add('open');
    if (backdrop) backdrop.classList.add('open');
    document.body.style.overflow = 'hidden';
  }

  function close() {
    drawer.classList.remove('open');
    if (backdrop) backdrop.classList.remove('open');
    document.body.style.overflow = '';
  }

  btn.addEventListener('click', open);
  if (backdrop) backdrop.addEventListener('click', close);
  if (closeBtn) closeBtn.addEventListener('click', close);
}


// =========================================
// NOTIFICATION POLLER (used across pages)
// =========================================
let _notifCallback = null;

function startNotificationPoller(callback) {
  _notifCallback = callback;
  SmartPoller.start('notifications', async () => {
    try {
      const resp = await fetch('/api/notifications');
      const data = await resp.json();
      if (data.status === 'success' && _notifCallback) {
        _notifCallback(data.notifications || []);
        // Browser push for unread critical notifications
        const unread = (data.notifications || []).filter(n => !n.read);
        if (unread.length > 0) {
          const critical = unread.find(n => n.type === 'critical');
          if (critical) {
            BrowserNotify.send('项目告警', {
              body: critical.message,
              url: `/projects/${critical.project_id}`
            });
          }
        }
      }
    } catch (e) { /* silent */ }
  }, 30000);
}


// =========================================
// VERSION CHANGELOG
// =========================================
const CHANGELOG = [
  {
    version: 'v0.4.0.2',
    date: '13/03/2026',
    changes: [
      '系统背景改为白色，字体改为黑色',
      '进度条颜色优化：完成区域显示绿色，异常部分显示红色',
      '新增虚拟项目以体现翻页功能',
    ],
  },
  {
    version: 'v0.4.0.1',
    date: '06/03/2026',
    changes: [
      '导出文件语言统一为中文',
      '系统名称以及描述改动',
      '项目名称前增加项目编号',
      '系统主页下方增加翻页键',
      '搜索栏改为精准搜索',
      '修改子项目分类',
      '项目分享版块增加「分享」功能，能够生成静态链接；如果是邮件分享也可以生成 HTML 分享页模式',
    ],
  },
];

function showChangelog() {
  const existing = document.querySelector('.changelog-overlay');
  if (existing) return;

  const overlay = document.createElement('div');
  overlay.className = 'changelog-overlay';

  const entriesHTML = CHANGELOG.map(entry => `
    <div style="margin-bottom: 8px;">
      <div class="changelog-title-wrap" style="margin-bottom:14px;">
        <span class="changelog-version-tag">${entry.version}</span>
        <span class="changelog-date">${entry.date}</span>
      </div>
      <div class="changelog-section-label">本次更新内容</div>
      <ul class="changelog-list">
        ${entry.changes.map(c => `
          <li class="changelog-item">
            <span class="changelog-item-dot"></span>
            <span>${c}</span>
          </li>
        `).join('')}
      </ul>
    </div>
  `).join('<hr style="border-color:#1e293b;margin:20px 0;">');

  overlay.innerHTML = `
    <div class="changelog-modal" role="dialog" aria-modal="true" aria-label="更新日志">
      <div class="changelog-header">
        <div>
          <div class="changelog-title">更新日志</div>
        </div>
        <button class="changelog-close" aria-label="关闭">&times;</button>
      </div>
      <div class="changelog-body">${entriesHTML}</div>
      <div class="changelog-footer">澳洲联通系统版本历史 · Australia Unicom</div>
    </div>
  `;

  function closeChangelog() {
    overlay.classList.add('modal-exit');
    setTimeout(() => overlay.remove(), 200);
  }

  overlay.querySelector('.changelog-close').addEventListener('click', closeChangelog);
  overlay.addEventListener('click', e => { if (e.target === overlay) closeChangelog(); });
  document.addEventListener('keydown', function escClose(e) {
    if (e.key === 'Escape') { closeChangelog(); document.removeEventListener('keydown', escClose); }
  });

  document.body.appendChild(overlay);
}


// =========================================
// AUTO-INIT
// =========================================
document.addEventListener('DOMContentLoaded', () => {
  initMobileDrawer();
  // Request browser notification permission after user interaction
  document.body.addEventListener('click', function reqNotif() {
    BrowserNotify.requestPermission();
    document.body.removeEventListener('click', reqNotif);
  }, { once: true });
  // Start heartbeat
  Heartbeat.start();
});
