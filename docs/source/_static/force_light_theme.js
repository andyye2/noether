/*
 * # Copyright © 2025 Emmi AI GmbH. All rights reserved.
 */

(function () {
  try {
    const root = document.documentElement;

    // Force Furo's theme attributes
    root.dataset.theme = "light";
    root.dataset.mode = "light";

    // Persist it, so even on future visits it starts as light
    try {
      localStorage.setItem("theme", "light");
      localStorage.setItem("mode", "light");
    } catch (e) {
      // Safari Private might block localStorage; that's fine.
    }
  } catch (e) {
    // Fail silently, docs should still render.
  }
})();