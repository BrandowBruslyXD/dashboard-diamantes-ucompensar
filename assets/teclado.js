// Accesibilidad de dcc.Tabs: las pestañas de Dash son <div> sin foco.
// Las volvemos una lista de pestañas navegable con Tab, Enter/Espacio y flechas.
(function () {
  function preparar() {
    var lista = document.querySelector('.pestanas');
    if (!lista) return;
    lista.setAttribute('role', 'tablist');
    lista.setAttribute('aria-label', 'Secciones del dashboard');
    var tabs = lista.querySelectorAll('.tab');
    tabs.forEach(function (t, i) {
      var activa = t.classList.contains('tab--selected');
      t.setAttribute('role', 'tab');
      t.setAttribute('aria-selected', activa ? 'true' : 'false');
      t.setAttribute('tabindex', activa ? '0' : '-1');
      if (t.dataset.teclado) return;
      t.dataset.teclado = '1';
      t.addEventListener('keydown', function (e) {
        var todas = lista.querySelectorAll('.tab');
        if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); t.click(); }
        if (e.key === 'ArrowRight' || e.key === 'ArrowLeft') {
          e.preventDefault();
          var sig = todas[(i + (e.key === 'ArrowRight' ? 1 : todas.length - 1)) % todas.length];
          sig.focus(); sig.click();
        }
      });
    });
  }
  new MutationObserver(preparar).observe(document.documentElement, { childList: true, subtree: true, attributes: true, attributeFilter: ['class'] });
})();
