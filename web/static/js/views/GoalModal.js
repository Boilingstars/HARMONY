const LABELS = { priorities: 'Приоритеты', money: 'Деньги' };

export function renderGoalModal(root, state, actions) {
  if (state.notice) {
    root.innerHTML = `
      <div class="modal-backdrop" data-close>
        <div class="modal" role="dialog">
          <h2>Сообщение</h2>
          <p class="empty">${escapeHtml(state.notice)}</p>
          <div class="actions">
            <button type="button" class="btn" data-close>Закрыть</button>
          </div>
        </div>
      </div>`;
    root.querySelectorAll('[data-close]').forEach((n) => {
      n.addEventListener('click', (e) => { if (e.target === n) actions.clearNotice(); });
    });
    return;
  }
  if (!state.pendingObjective) {
    root.innerHTML = '';
    return;
  }
  root.innerHTML = `
    <div class="modal-backdrop" data-close>
      <div class="modal" role="dialog">
        <h2>Смена цели</h2>
        <p>Действительно ли вы хотите поменять цель смены с шага ${state.step}?</p>
        <p class="goal-fromto">${LABELS[state.objective] || state.objective} → ${LABELS[state.pendingObjective] || state.pendingObjective}</p>
        <p class="goal-effect">Ожидаемый эффект от смены:</p>
        <p class="empty"> </p>
        <div class="actions">
          <button type="button" class="btn" data-close>Отмена</button>
          <button type="button" class="btn btn-go" data-ok>Сменить</button>
        </div>
      </div>
    </div>`;
  root.querySelectorAll('[data-close]').forEach((n) => {
    n.addEventListener('click', (e) => { if (e.target === n) actions.cancelObjective(); });
  });
  root.querySelector('[data-ok]').addEventListener('click', (e) => {
    e.stopPropagation();
    actions.confirmObjective();
  });
}

function escapeHtml(s) {
  return String(s)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}
