import { mountSplash } from './views/SplashScreen.js';

const splashRoot = document.getElementById('splash-root');
const consoleEl = document.getElementById('console');
if (consoleEl) consoleEl.hidden = true;

mountSplash(splashRoot, {
  async onStartShift(scenario) {
    const [{ snapshotFromScenario }, { startConsole }] = await Promise.all([
      import('./scenario.js'),
      import('./consoleApp.js'),
    ]);
    splashRoot.innerHTML = '';
    splashRoot.hidden = true;
    startConsole(snapshotFromScenario(scenario));
  },
});
