// ═══ View Controller ═══

const App = {
  currentView: 'setup',

  init() {
    document.querySelectorAll('.nav-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        const view = btn.dataset.view;
        if (!btn.disabled) App.showView(view);
      });
    });
  },

  showView(name) {
    App.currentView = name;

    document.querySelectorAll('.view').forEach(v => v.classList.remove('active'));
    document.getElementById(`view-${name}`).classList.add('active');

    document.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));
    document.querySelector(`.nav-btn[data-view="${name}"]`).classList.add('active');
  },

  enableNav(name) {
    document.getElementById(`nav-${name}`).disabled = false;
  },

  disableNav(name) {
    document.getElementById(`nav-${name}`).disabled = true;
  }
};

App.init();
